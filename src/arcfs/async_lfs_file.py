"""async_lfs_file.py

Async streamed file used by GitLabARCFileSystem.
"""

from __future__ import annotations

import io
from hashlib import sha256

import aiofiles
from fsspec.asyn import AbstractAsyncStreamedFile

from .transactions import commit_lfs_transaction

tempfile = aiofiles.tempfile


class AsyncLFSFile(AbstractAsyncStreamedFile):
    """
    Async fsspec streamed file backed by a temporary file and Git LFS commits.

    Reads lazily download the GitLab file into a temporary file. Writes update
    the temporary file and commit changed content through the LFS transaction
    workflow when the context manager exits successfully.
    """

    def __init__(
        self,
        fs,
        path,
        token,
        repo_id,
        ref,
        mode="rb",
        feature_branch=None,
        **kwargs,
    ):
        """
        Create an async streamed file that reads from GitLab and writes via LFS.

        Args:
            fs: Owning ``GitLabARCFileSystem`` instance.
            path: Repository-internal file path.
            token: GitLab token used for LFS upload authentication.
            repo_id: Numeric GitLab project id.
            ref: Branch, tag, or commit SHA to read from or base writes on.
            mode: File mode such as ``"rb"`` or ``"wb"``.
            feature_branch: Complete branch name used for writes.
            **kwargs: Additional fsspec streamed-file arguments.
        """
        super().__init__(fs=fs, path=path, mode=mode, **kwargs)
        self.path = path
        self.token = token
        self.repo_id = repo_id
        self.ref = ref
        self.mode = mode
        self.feature_branch = (
            feature_branch
            if feature_branch is not None
            else fs.feature_branch
        )

        self._tmp = None
        self._shasum = sha256()
        self._changed = False
        self._downloaded = False
        self.fs = fs

    async def _ensure_tmp(self):
        """
        Create the temporary backing file if it does not already exist.

        Returns:
            None.
        """
        if self._tmp is None:
            self._tmp = await tempfile.NamedTemporaryFile(mode="w+b", delete=True)

    async def __aenter__(self):
        """
        Enter the async context manager and prepare the backing file.

        Returns:
            This ``AsyncLFSFile`` instance, ready for async reads or writes.
        """
        await self._ensure_tmp()
        if "r" in self.mode and not self._downloaded:
            await self._download_from_gitlab()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        Exit the async context manager, committing changed data when successful.

        Args:
            exc_type: Exception type raised inside the context, or ``None``.
            exc: Exception instance raised inside the context, or ``None``.
            tb: Traceback for the exception, or ``None``.

        Returns:
            None.
        """
        try:
            if exc_type is None:
                await self._commit()
        finally:
            if self._tmp:
                await self._tmp.close()
                self._tmp = None
            self.closed = True

    async def _download_from_gitlab(self):
        """
        Download the remote GitLab file into the temporary backing file.

        Returns:
            None.
        """
        ref = self.ref or await self.fs.client.get_default_branch(self.repo_id)

        await self._ensure_tmp()
        async for chunk in self.fs.client.stream_file(
                repo_id=self.repo_id,
                path=self.path,
                ref=ref,
        ):
            await self._tmp.write(chunk)

        await self._tmp.seek(0)
        self._downloaded = True

    async def read(self, length=-1):
        """
        Read bytes from the temporary file, downloading first when needed.

        Args:
            length: Maximum number of bytes to read, or ``-1`` for the rest of
                the file.

        Returns:
            Bytes read from the file.
        """
        await self._ensure_tmp()
        if "r" in self.mode and not self._downloaded:
            await self._download_from_gitlab()
        return await self._tmp.read(length)

    async def write(self, data):
        """
        Write bytes to the temporary file and update the upload checksum.

        Args:
            data: ``bytes`` or ``str`` data to write. Strings are encoded as
                UTF-8 bytes before writing.

        Returns:
            Number of bytes written, as returned by the temporary file.
        """
        await self._ensure_tmp()
        if isinstance(data, str):
            data = data.encode()
        self._changed = True
        self._shasum.update(data)
        return await self._tmp.write(data)

    async def _commit(self):
        """
        Commit changed temporary-file content through the Git LFS workflow.

        Returns:
            None.
        """
        if not self._changed:
            return

        await self._ensure_tmp()
        await self._tmp.seek(0, io.SEEK_END)
        size = await self._tmp.tell()
        await self._tmp.seek(0)
        sha = self._shasum.hexdigest()

        repo = await self.fs.client.get_project_by_id(self.repo_id)
        if not repo:
            raise FileNotFoundError(f"Project id {self.repo_id} not found")

        await commit_lfs_transaction(
            client=self.fs.client,
            token=str(self.token or ""),
            repo=repo,
            base_branch=self.ref,
            final_path=self.path,
            sha=sha,
            size=size,
            data_stream=self._tmp,
            feature_branch=self.feature_branch,
            tmp_pointer_name=True,
            create_mr=True,
            mode="create" if "x" in self.mode else "overwrite",
        )

        if hasattr(self.fs, "_invalidate_after_write"):
            self.fs._invalidate_after_write(repo=repo, inside_path=self.path)

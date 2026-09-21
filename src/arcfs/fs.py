from __future__ import annotations

import asyncio
from typing import Optional
from uuid import uuid4

import aiofiles

import fsspec
from fsspec.asyn import AsyncFileSystem, sync
from .async_lfs_file import AsyncLFSFile
from .gitlab_client import GitLabClient
from .utils import norm_inside


# GitLab REST pagination documents a maximum ``per_page`` of 100.
GITLAB_MAX_PER_PAGE = 100


class GitLabARCFileSystem(AsyncFileSystem):
    """
    fsspec filesystem for GitLab repositories.

    Path forms supported:
      - Raw GitLab paths:
            group/subgroup/repo/path/to/file
      - Marker paths (internal / canonical):
            group/subgroup:-:repo/path/to/file

    Internally, everything resolves to:
        (repo_id, inside_path)
    """

    root_marker = ":-:"

    def __init__(
        self,
        base_url: str,
        token: str | None,
        asynchronous: bool = False,
        feature_branch: str | None = None,
        feature_branch_prefix: str | None = None,
        **kwargs,
    ):
        """
        Create a GitLab-backed fsspec filesystem.

        Args:
            base_url: Base GitLab instance URL.
            token: Private token used for GitLab API authentication, or
                ``None`` for unauthenticated requests.
            asynchronous: If True, expose async fsspec behavior; if False,
                fsspec wraps async methods for synchronous callers.
            feature_branch: Complete upload branch name. If omitted, one
                UUID-based name is generated for this filesystem instance.
            feature_branch_prefix: Compatibility alias for ``feature_branch``.
            **kwargs: Additional keyword arguments passed to
                ``AsyncFileSystem``.
        """
        super().__init__(asynchronous=asynchronous, **kwargs)

        self.client = GitLabClient(base_url, token)
        normalized_feature_branch = self._normalize_feature_branch(
            feature_branch,
            feature_branch_prefix,
        )
        self.feature_branch = (
            normalized_feature_branch
            if normalized_feature_branch is not None
            else f"run_results-{uuid4()}"
        )

        # Project cache:
        #   original_path -> {"id": int, "original_path": str}
        self.repos: dict[str, dict] = {}

        # Negative cache for failed raw-prefix probes
        self.not_repo: set[str] = set()

        # Root index state
        self._project_index_built: bool = False
        self._project_index_building: bool = False

    @staticmethod
    def _normalize_feature_branch(
        feature_branch: str | None,
        feature_branch_prefix: str | None,
        *,
        default: str | None = None,
    ) -> str | None:
        """Normalize the legacy branch-name alias at a filesystem boundary."""
        if (
            feature_branch is not None
            and feature_branch_prefix is not None
            and feature_branch != feature_branch_prefix
        ):
            raise ValueError(
                "feature_branch and feature_branch_prefix must match when both "
                "are provided"
            )
        if feature_branch is not None:
            return feature_branch
        if feature_branch_prefix is not None:
            return feature_branch_prefix
        return default

    async def _close(self):
        """
        Close the underlying GitLab client session.

        Returns:
            None.
        """
        await self.client.close()

    def close(self) -> None:
        """Close the underlying GitLab client session from synchronous code."""
        if self.asynchronous:
            raise RuntimeError("Use _close() with asynchronous=True filesystems.")

        sync(self.loop, self._close)

    async def _ensure_project_index(self, *, refresh: bool = False) -> None:
        """
        Ensure the in-memory GitLab project index is available.

        Builds an index of accessible projects keyed by ``original_path`` the first
        time it is needed. Subsequent calls are no-ops unless ``refresh=True`` is
        given.

        Args:
            refresh: If True, rebuild and replace the cached project index even if it
                already exists.

        Side effects:
            - Populates or replaces ``self.repos``.
            - Clears ``self.not_repo`` only on explicit refresh after a successful
              rebuild.
            - Sets internal flags to prevent concurrent rebuilds.

        Returns:
            None.
        """
        if self._project_index_built and not refresh:
            return

        if self._project_index_building:
            while self._project_index_building:
                await fsspec.asyn.asyncio.sleep(0.01)
            return

        self._project_index_building = True
        try:
            projects = await self.client.retrieve_root_level(
                per_page=100,
                simple=True,
            )
            self.repos.clear()
            self.repos.update({p["original_path"]: p for p in projects})
            if refresh:
                self.not_repo.clear()
            self._project_index_built = True
        finally:
            self._project_index_building = False

    # ------------------------------------------------------------------
    # Resolve helpers
    # ------------------------------------------------------------------
    def _resolve_from_cache(self, raw_path: str) -> Optional[tuple[dict, str]]:
        """
        Resolve a raw GitLab path using only the local repo cache.

        Args:
            raw_path: Raw filesystem path such as
                ``"group/sub/repo/path/to/file"``.

        Returns:
            ``(repo, inside_path)`` when a cached project prefix matches, where
            ``repo`` is the cached project dict and ``inside_path`` is the
            repository-internal path. Returns ``None`` when no cached project
            prefix matches.

        Uses longest-prefix matching against ``self.repos`` keys. E.G. if
        ``self.repos`` contains "group/sub/repo" and ``raw_path`` is
        "group/sub/repo/path/to/file", this returns (repo, "path/to/file").
        """
        parts = [p for p in raw_path.strip("/").split("/") if p]
        if not parts:
            return None

        for i in range(len(parts), 0, -1):
            candidate = "/".join(parts[:i])
            repo = self.repos.get(candidate)
            if repo:
                inside = "/".join(parts[i:])
                return repo, norm_inside(inside)
        return None

    async def _resolve_marker(self, path: str) -> tuple[dict, str]:
        """
        Resolve a canonical marker path into project metadata and inside path.

        Args:
            path: Marker path in the form ``"original_path:-:/inside"``.

        Returns:
            ``(repo, inside_path)`` where ``repo`` is a project dict and
            ``inside_path`` is normalized relative to the repository root.
        """
        original_path, rest = path.split(self.root_marker, 1)
        original_path = original_path.strip("/")
        inside = rest.lstrip("/")

        repo = self.repos.get(original_path)
        if not repo:
            repo = await self.client.get_project_by_path(original_path)
            if not repo:
                raise FileNotFoundError(original_path)
            self.repos[repo["original_path"]] = repo

        return repo, norm_inside(inside)

    async def _resolve_raw(self, path: str, *, refresh: bool = False) -> tuple[dict, str]:
        """
        Resolve a raw GitLab path by probing project prefixes and cache indexes.

        Args:
            path: Raw filesystem path such as
                ``"group/sub/repo/path/to/file"``.
            refresh: If True, rebuild the full project index if fallback index
                resolution is needed.

        Returns:
            ``(repo, inside_path)`` where ``repo`` is a project dict and
            ``inside_path`` is normalized relative to the repository root.
        """
        # 1) cache-only lookup (longest prefix)
        hit = self._resolve_from_cache(path)
        if hit is not None:
            return hit

        # 2) DIRECT lookup: the whole path might be the repo root
        proj = await self.client.get_project_by_path(path)
        if proj:
            self.repos[proj["original_path"]] = proj
            return proj, ""  # repo root = directory

        # 3) lazy probing for prefixes (namespace/project/inside/...)
        parts = [p for p in path.strip("/").split("/") if p]
        for i in range(len(parts), 0, -1):
            candidate = "/".join(parts[:i])
            if candidate in self.not_repo:
                continue

            proj = await self.client.get_project_by_path(candidate)
            if proj:
                self.repos[proj["original_path"]] = proj
                inside = "/".join(parts[i:])
                return proj, norm_inside(inside)

            self.not_repo.add(candidate)

        # 4) build full index once, retry
        await self._ensure_project_index(refresh=refresh)

        hit = self._resolve_from_cache(path)
        if hit is not None:
            return hit

        raise FileNotFoundError(path)

    async def _resolve(
        self,
        path: str,
        *,
        refresh: bool = False,
        **kwargs
    ) -> tuple[dict, str]:
        """
        Resolve any filesystem path into (repo, inside_path).

        Args:
            path: Raw GitLab path or marker path.
            refresh: If True, allow cache/index rebuilding while resolving.
            **kwargs: Extra fsspec keyword arguments accepted for compatibility
                with callers; currently not used by the resolver.

        Returns:
            ``(repo, inside_path)`` where ``repo`` is a project dict and
            ``inside_path`` is normalized relative to the repository root.
        """
        path = (path or "").strip().strip("/")
        if self.root_marker in path:
            return await self._resolve_marker(path)
        return await self._resolve_raw(path, refresh=refresh)

    # ------------------------------------------------------------------
    # (extended) fsspec API
    # ------------------------------------------------------------------
    async def _open_async_lfs_file(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        feature_branch: str | None = None,
        **kwargs,
    ):
        refresh = bool(kwargs.pop("refresh", False))
        ref = kwargs.pop("ref", None)
        if feature_branch is None:
            feature_branch = self.feature_branch

        repo, inside = await self._resolve(path, refresh=refresh)
        if not inside:
            raise IsADirectoryError(path)

        file_kwargs = dict(kwargs)
        if block_size is not None:
            file_kwargs["block_size"] = block_size
        if cache_options is not None:
            file_kwargs["cache_options"] = cache_options
        file_kwargs["autocommit"] = autocommit
        if "r" in mode and "size" not in file_kwargs:
            file_kwargs["size"] = 0

        return AsyncLFSFile(
            fs=self,
            path=inside,
            token=self.client.token,
            repo_id=repo["id"],
            ref=ref,
            mode=mode,
            feature_branch=feature_branch,
            **file_kwargs,
        )

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        feature_branch: str | None = None,
        **kwargs,
    ):
        if self.asynchronous:
            raise RuntimeError("Use open_async() with asynchronous=True filesystems")

        feature_branch = self._normalize_feature_branch(
            feature_branch,
            kwargs.pop("feature_branch_prefix", None),
            default=self.feature_branch,
        )

        return asyncio.run(
            self._open_async_lfs_file(
                path,
                mode=mode,
                block_size=block_size,
                autocommit=autocommit,
                cache_options=cache_options,
                feature_branch=feature_branch,
                **kwargs,
            )
        )

    async def open_async(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        feature_branch: str | None = None,
        **kwargs,
    ):
        compression = kwargs.pop("compression", None)
        if "b" not in mode or compression is not None:
            raise ValueError

        feature_branch = self._normalize_feature_branch(
            feature_branch,
            kwargs.pop("feature_branch_prefix", None),
            default=self.feature_branch,
        )

        return await self._open_async_lfs_file(
            path,
            mode=mode,
            block_size=block_size,
            autocommit=autocommit,
            cache_options=cache_options,
            feature_branch=feature_branch,
            **kwargs,
        )

    async def _ls(self, path: str, detail: bool = True, **kwargs):
        """
        List a directory.

        Args:
            path: Root, raw GitLab path, or marker path to list.
            detail: If True, return fsspec-style entry dicts; if False, return
                entry names only.
            **kwargs: Optional listing controls. ``refresh=True`` invalidates
                cached listings, and ``ref`` selects a branch, tag, or commit
                SHA for repository tree listings.

        Returns:
            A list of entry dicts when ``detail`` is True, otherwise a list of
            entry name strings.

        refresh=True forces cache invalidation and index rebuild.
        """
        refresh = bool(kwargs.get("refresh", False))
        path = (path or "").strip().strip("/")

        if path == "":
            cache_key = "__root__"

            out = None if refresh else self.dircache.get(cache_key)
            if out is None:
                await self._ensure_project_index(refresh=refresh)
                out = [
                    {
                        "name": f"{repo['original_path']}{self.root_marker}",
                        "type": "directory",
                    }
                    for repo in self.repos.values()
                ]
                # Return what was just built rather than reading it back. The cache
                # may be configured to keep nothing (use_listings_cache=False) or to
                # expire entries, in which case reading it back would raise.
                self.dircache[cache_key] = out

            return out if detail else [e["name"] for e in out]

        repo, inside = await self._resolve(
            path,
            refresh=refresh,
        )
        key = f"{repo['original_path']}{self.root_marker}"
        cache_key = f"{key}/{inside}" if inside else key

        out = None if refresh else self.dircache.get(cache_key)
        if out is None:
            ref = kwargs.get("ref")
            if ref is None:
                ref = await self.client.get_default_branch(repo["id"])

            items = await self.client.retrieve_project_level(
                repo["id"],
                inside,
                ref=ref,
            )
            out = [
                {
                    "name": f"{key}{i['path']}",
                    "type": "directory" if i.get("type") == "tree" else "file",
                }
                for i in items
            ]
            self.dircache[cache_key] = out

        return out if detail else [e["name"] for e in out]

    async def _list_page(
            self,
            path: str,
            detail: bool = True,
            *,
            offset: int = 0,
            limit: int = 50,
            **kwargs,
    ) -> tuple[list, int]:
        """
        Return one paginated listing page as (entries, total_count).

        Args:
            path: Root, raw GitLab path, or marker path to list.
            detail: If True, return fsspec-style entry dicts; if False, return
                entry names only.
            offset: Zero-based item offset into the listing.
            limit: Maximum number of entries to return.
            **kwargs: Optional listing controls passed through to paging or full
                listing calls. Supported values include ``refresh``, ``ref``,
                ``membership``, ``archived``, and ``simple``.

        Returns:
            ``(entries, total_count)`` where ``entries`` is the requested page
            and ``total_count`` is the total number of entries available when
            GitLab reports one, otherwise a lower bound that grows as the caller
            pages. Only the backend pages covering the requested window are
            fetched; errors are raised rather than retried as a full listing.
        """
        refresh = bool(kwargs.pop("refresh", False))
        path = (path or "").strip().strip("/")

        if limit <= 0:
            return [], 0
        if offset < 0:
            raise ValueError("offset must be >= 0")

        # Backend paging is page/per_page based, so serve the requested window by fetching the pages
        # that cover it and slicing. Asking for per_page=limit only worked when the offset happened
        # to be a multiple of the limit. The projects endpoint behind the root listing also caps
        # per_page at 100 and silently returns a shorter page, so those windows may need more than
        # one request. The repository tree endpoint honours larger values on the tested GitLab
        # version, but GitLab documents 100 as the general maximum, so cap both endpoints.
        per_page = min(limit, GITLAB_MAX_PER_PAGE)
        first_page = (offset // per_page) + 1
        last_page = ((offset + limit - 1) // per_page) + 1
        start_in_first_page = offset - (first_page - 1) * per_page

        if path == "":

            async def fetch_page(page: int) -> tuple[list, int]:
                items, total_count = await self.client.retrieve_root_level_page(
                    page=page,
                    per_page=per_page,
                    membership=bool(kwargs.get("membership", False)),
                    archived=bool(kwargs.get("archived", False)),
                    simple=bool(kwargs.get("simple", True)),
                )

                return [
                    {
                        "name": f"{repo['original_path']}{self.root_marker}",
                        "type": "directory",
                    }
                    for repo in items
                ], total_count

        else:
            repo, inside = await self._resolve(path, refresh=refresh, **kwargs)
            key = f"{repo['original_path']}{self.root_marker}"

            async def fetch_page(page: int) -> tuple[list, int]:
                items, total_count = await self.client.retrieve_project_level_page(
                    repo_id=repo["id"],
                    subdir=inside,
                    ref=kwargs.get("ref"),
                    page=page,
                    per_page=per_page,
                )

                return [
                    {
                        "name": f"{key}{item['path']}",
                        "type": "directory" if item.get("type") == "tree" else "file",
                    }
                    for item in items
                ], total_count

        collected: list = []
        total_count = 0
        for page in range(first_page, last_page + 1):
            entries, page_total = await fetch_page(page)
            collected.extend(entries)
            # Keep the tightest count seen. Without an exact total each page
            # reports only a lower bound, and a later, emptier page reports a
            # looser one than a full page already did.
            total_count = max(total_count, page_total)
            if len(entries) < per_page:
                break

        out = collected[start_in_first_page:start_in_first_page + limit]
        return out if detail else [e["name"] for e in out], total_count

    def list_page(
        self,
        path: str,
        detail: bool = True,
        *,
        offset: int = 0,
        limit: int = 50,
        **kwargs,
    ) -> tuple[list, int]:
        """Return one paginated listing page from synchronous code.

        This is the public synchronous wrapper around ``_list_page``.
        """
        if self.asynchronous:
            raise RuntimeError("Use _list_page() with asynchronous=True filesystems.")

        return sync(
            self.loop,
            self._list_page,
            path,
            detail,
            offset=offset,
            limit=limit,
            **kwargs,
        )

    async def _get_file(self, rpath, lpath, **kwargs):
        """
        Download one remote file to a local path in streaming chunks.

        Args:
            rpath: Remote ARCfs path to read.
            lpath: Local filesystem path to write.
            **kwargs: Optional controls. ``refresh=True`` allows cache refresh
                during resolution, ``ref`` selects the Git ref to read, and
                ``chunk_size`` controls streaming chunk size.

        Returns:
            None.
        """
        refresh = bool(kwargs.get("refresh", False))
        chunk_size = int(kwargs.get("chunk_size", 1024 * 1024))

        repo, inside = await self._resolve(
            rpath,
            refresh=refresh
        )

        if not inside:
            raise IsADirectoryError(rpath)

        ref = kwargs.get("ref")
        if ref is None:
            ref = await self.client.get_default_branch(repo["id"])

        async with aiofiles.open(lpath, "wb") as f:
            async for chunk in self.client.stream_file(
                    repo_id=repo["id"],
                    path=inside,
                    ref=ref,
                    chunk_size=chunk_size,
            ):
                await f.write(chunk)

    async def _put_file(
        self,
        lpath,
        rpath,
        mode="overwrite",
        feature_branch: str | None = None,
        **kwargs,
    ):
        """
        Upload one local file to GitLab as an LFS-backed file.

        Args:
            lpath: Local/native source path.
            rpath: Remote ARCfs target path.
            **kwargs: Optional controls. ``refresh=True`` allows cache refresh
                during resolution, ``ref`` selects the base branch, ``create_mr``
                controls merge request creation, ``feature_branch`` selects the
                complete upload branch name, and ``feature_branch_prefix`` is a
                compatibility alias for ``feature_branch``.
            mode: ``"create"`` refuses an existing target; ``"overwrite"``
                applies the ARCfs safe replacement policy.

        Returns:
            None.
        """
        refresh = bool(kwargs.pop("refresh", False))
        feature_branch = self._normalize_feature_branch(
            feature_branch,
            kwargs.pop("feature_branch_prefix", None),
            default=self.feature_branch,
        )

        repo, inside = await self._resolve(
            rpath,
            refresh=refresh,
            **kwargs,
        )

        if not inside:
            raise IsADirectoryError(rpath)

        await self.client.upload_file_lfs(
            token=str(self.client.token or ""),
            repo=repo,
            final_path=inside,
            local_path=lpath,
            base_branch=kwargs.get("ref"),
            feature_branch=feature_branch,
            create_mr=bool(kwargs.get("create_mr", True)),
            mode=mode,
        )

        if hasattr(self, "_invalidate_after_write"):
            self._invalidate_after_write(repo=repo, inside_path=inside)
        else:
            self.dircache.clear()


    # ------------------------------------------------------------------
    # Explicitly disabled destructive operations
    # ------------------------------------------------------------------
    async def _rm_file(self, path, **kwargs):
        """
        Reject deletion of a single remote file.

        Args:
            path: Remote path requested for deletion.
            **kwargs: Extra fsspec keyword arguments accepted for compatibility.

        Raises:
            PermissionError: Always, because delete operations are disabled.
        """
        raise PermissionError("Delete operations are disabled.")

    async def _rm(self, path, recursive=False, batch_size=None, **kwargs):
        """
        Reject deletion of a remote path or directory tree.

        Args:
            path: Remote path requested for deletion.
            recursive: Whether the caller requested recursive deletion.
            batch_size: Optional fsspec batch size argument.
            **kwargs: Extra fsspec keyword arguments accepted for compatibility.

        Raises:
            PermissionError: Always, because delete operations are disabled.
        """
        raise PermissionError("Delete operations are disabled.")

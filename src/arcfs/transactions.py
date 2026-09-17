from __future__ import annotations

import base64
import binascii
from pathlib import PurePosixPath
from typing import Any, Optional

from .errors import NonLFSFileError, RefNotFound
from .utils import gitattributes_block, lfs_pointer_text, parse_lfs_pointer


async def commit_pointer(*, client, repo_id: int, branch: str, filepath: str, sha: str, size: int) -> None:
    """
    Commit a Git LFS pointer file to a repository branch.

    Args:
        client: GitLab client exposing ``create_commit``.
        repo_id: Numeric GitLab project id.
        branch: Branch that receives the pointer commit.
        filepath: Repository-internal pointer file path.
        sha: SHA256 object id for the LFS object.
        size: Size of the LFS object in bytes.

    Returns:
        None.
    """
    pointer = lfs_pointer_text(sha, size)
    actions = [{"action": "create", "file_path": filepath, "content": pointer, "encoding": "text"}]
    await client.create_commit(repo_id, branch, f"Add LFS pointer {filepath}", actions)


async def move_pointer(*, client, repo_id: int, branch: str, src: str, dst: str) -> None:
    """
    Move an existing pointer file to its final repository path.

    Args:
        client: GitLab client exposing ``create_commit``.
        repo_id: Numeric GitLab project id.
        branch: Branch that receives the move commit.
        src: Current repository-internal pointer path.
        dst: Final repository-internal pointer path.

    Returns:
        None.
    """
    actions = [{"action": "move", "file_path": dst, "previous_path": src}]
    await client.create_commit(repo_id, branch, f"Move pointer {src} -> {dst}", actions)


async def update_gitattributes(*, client, repo_id: int, branch: str, path_str: str) -> bool:
    """
    Ensure ``.gitattributes`` contains the exact Git LFS rule for a path.

    Creates ``.gitattributes`` when it does not exist, or appends the rule to
    the existing file when needed.

    Args:
        client: GitLab client exposing ``get_file`` and ``create_commit``.
        repo_id: Numeric GitLab project id.
        branch: Branch that receives the ``.gitattributes`` commit.
        path_str: Repository-internal path that should be tracked by Git LFS.

    Returns:
        ``True`` if an LFS rule was added, or ``False`` if the exact rule
        already existed.
    """
    ga_path = ".gitattributes"
    block = gitattributes_block(path_str)

    try:
        file_json = await client.get_file(repo_id, ga_path, branch)
    except RefNotFound:
        # A ref that will not resolve is not a missing .gitattributes; committing anyway only
        # turns it into a 400 further along.
        raise
    except FileNotFoundError:
        action = {
            "action": "create",
            "file_path": ga_path,
            "content": block,
            "encoding": "text",
        }
        await client.create_commit(repo_id, branch, f"Add {ga_path}", [action])
        return True

    existing = base64.b64decode(file_json["content"]).decode(
        "utf-8",
        errors="replace",
    )
    rule = block.splitlines()[-1]
    if rule in existing.splitlines():
        return False
    separator = "" if not existing or existing.endswith("\n") else "\n"
    updated = existing + separator + block
    action = {
        "action": "update",
        "file_path": ga_path,
        "content": updated,
        "encoding": "text",
    }
    await client.create_commit(repo_id, branch, f"Update {ga_path}", [action])
    return True


async def _classify_target(
    *,
    client,
    repo_id: int,
    feature_branch: str,
    final_path: str,
) -> dict[str, Any] | None:
    """Return target metadata while rejecting file ancestors and directories."""
    parts = PurePosixPath(final_path).parts
    for index in range(1, len(parts)):
        ancestor = "/".join(parts[:index])
        try:
            await client.get_file(repo_id, ancestor, feature_branch)
        except FileNotFoundError:
            continue
        raise NotADirectoryError(ancestor)

    try:
        return await client.get_file(repo_id, final_path, feature_branch)
    except FileNotFoundError:
        pass

    try:
        entries = await client.retrieve_project_level(
            repo_id,
            final_path,
            ref=feature_branch,
        )
    except FileNotFoundError:
        entries = []
    if entries:
        raise IsADirectoryError(final_path)
    return None


async def _negotiate_lfs_upload(
    *,
    client,
    namespace: str,
    token: str,
    base_branch: str,
    sha: str,
    size: int,
    data_stream,
) -> None:
    """Negotiate an LFS object upload and send bytes only when requested."""
    payload = {
        "operation": "upload",
        "objects": [{"oid": sha, "size": size}],
        "transfers": ["basic"],
        "ref": {"name": f"refs/heads/{base_branch}"},
    }
    batch_resp = await client.lfs_batch(namespace, token, payload)
    obj0 = batch_resp["objects"][0]
    upload = (obj0.get("actions") or {}).get("upload")
    if upload:
        await client.lfs_upload(
            token,
            upload["href"],
            upload.get("header") or {},
            data_stream,
        )


async def commit_lfs_transaction(
    *,
    client,
    token: str,
    repo: dict[str, Any],
    base_branch: Optional[str],
    final_path: str,
    sha: str,
    size: int,
    data_stream,
    feature_branch: str,
    tmp_pointer_name: bool = True,
    create_mr: bool = True,
    mode: str = "overwrite",
) -> str:
    """
    Perform the Git LFS upload and pointer-commit workflow.

    Creates or reuses the requested feature branch, validates the destination,
    negotiates and performs the LFS object upload when required, ensures the
    destination has an LFS rule, and commits the pointer. For a new target, the
    pointer may first be committed under its object id and then moved to the
    final path. For an existing valid LFS pointer, overwrite mode updates it
    safely using its last commit id when GitLab supplies one. A merge request
    is ensured after any required commits when requested.

    Args:
        client: GitLab client exposing branch, file, commit, merge request, and
            LFS helpers.
        token: Token used for Git LFS upload authentication.
        repo: Project dict containing at least ``id`` and ``original_path``.
        base_branch: Branch on which to base the feature branch. If ``None``,
            the project default branch is used.
        final_path: Repository-internal destination path for the LFS pointer.
        sha: SHA256 object id for the LFS object.
        size: Size of the LFS object in bytes.
        data_stream: Async readable binary stream containing the LFS object.
        feature_branch: Complete branch name used for the pointer commits.
        tmp_pointer_name: If ``True`` for a new target, commit the pointer under
            the SHA first and move it to the final path after updating
            ``.gitattributes``.
        create_mr: If ``True``, ensure a merge request after making any required
            commits. An already-complete transaction returns without creating
            one.
        mode: Write mode. ``"create"`` refuses an existing target, while
            ``"overwrite"`` safely updates an existing valid LFS pointer.

    Returns:
        The feature-branch name supplied in ``feature_branch``.

    Raises:
        ValueError: If ``mode`` is neither ``"create"`` nor ``"overwrite"``.
        FileExistsError: If ``mode="create"`` and the target already exists.
        IsADirectoryError: If the target path identifies a directory.
        NotADirectoryError: If an ancestor of the target path is a file.
        NonLFSFileError: If an existing target is not a valid Git LFS pointer.
    """
    if mode not in {"create", "overwrite"}:
        raise ValueError(
            f"Unsupported transaction mode {mode!r}; expected 'create' or "
            "'overwrite'"
        )

    repo_id = repo["id"]
    namespace = repo["original_path"]
    base = base_branch or await client.get_default_branch(repo_id)

    await client.create_branch(repo_id, feature_branch, base)

    existing_file = await _classify_target(
        client=client,
        repo_id=repo_id,
        feature_branch=feature_branch,
        final_path=final_path,
    )
    if existing_file is not None and mode == "create":
        raise FileExistsError(final_path)

    existing_pointer = None
    if existing_file is not None:
        try:
            content = base64.b64decode(existing_file["content"], validate=True)
        except (KeyError, TypeError, ValueError, binascii.Error):
            pass
        else:
            existing_pointer = parse_lfs_pointer(content)
        if existing_pointer is None:
            raise NonLFSFileError(final_path)

    if existing_pointer == (sha, size):
        changed = await update_gitattributes(
            client=client,
            repo_id=repo_id,
            branch=feature_branch,
            path_str=final_path,
        )
        if not changed:
            return feature_branch
    else:
        await _negotiate_lfs_upload(
            client=client,
            namespace=namespace,
            token=token,
            base_branch=base,
            sha=sha,
            size=size,
            data_stream=data_stream,
        )

        if existing_file is not None:
            await update_gitattributes(
                client=client,
                repo_id=repo_id,
                branch=feature_branch,
                path_str=final_path,
            )
            action = {
                "action": "update",
                "file_path": final_path,
                "content": lfs_pointer_text(sha, size),
                "encoding": "text",
            }
            if last_commit_id := existing_file.get("last_commit_id"):
                action["last_commit_id"] = last_commit_id
            await client.create_commit(
                repo_id,
                feature_branch,
                f"Update LFS pointer {final_path}",
                [action],
            )
        else:
            p = PurePosixPath(final_path)
            if tmp_pointer_name:
                tmp_path = str(p.parent / sha) if p.parent != PurePosixPath(".") else sha
                await commit_pointer(
                    client=client, repo_id=repo_id, branch=feature_branch,
                    filepath=tmp_path, sha=sha, size=size,
                )
                await update_gitattributes(
                    client=client, repo_id=repo_id, branch=feature_branch,
                    path_str=final_path,
                )
                await move_pointer(
                    client=client, repo_id=repo_id, branch=feature_branch,
                    src=tmp_path, dst=final_path,
                )
            else:
                await commit_pointer(
                    client=client, repo_id=repo_id, branch=feature_branch,
                    filepath=final_path, sha=sha, size=size,
                )
                await update_gitattributes(
                    client=client, repo_id=repo_id, branch=feature_branch,
                    path_str=final_path,
                )

    if create_mr:
        await client.ensure_merge_request(
            repo_id=repo_id,
            source_branch=feature_branch,
            target_branch=base,
            title=f"ARCfs export {feature_branch}",
        )

    return feature_branch

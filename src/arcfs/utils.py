from __future__ import annotations

from pathlib import Path, PurePosixPath
import hashlib
import re
import aiofiles


_LFS_POINTER_RE = re.compile(
    r"version https://git-lfs.github.com/spec/v1\n"
    r"oid sha256:([0-9a-f]{64})\n"
    r"size (0|[1-9][0-9]*)\n"
)


def split_first(path: Path | str):
    """
    Split a path into (head, tail).

    Args:
        path: Path-like value to split.

    Returns:
        ``(head, tail)`` as ``Path`` objects. ``head`` is the first path
        component and ``tail`` is the remaining path.

    Example:
        split_first("a/b/c") -> ("a", "b/c")
    """
    p = Path(path)
    parts = p.parts
    if not parts:
        return Path(""), Path("")
    return Path(parts[0]), Path(*parts[1:])


def norm_inside(inside: str | None) -> str:
    """
    Normalize a repository-internal path.

    Args:
        inside: Repository-internal path, or ``None``.

    Returns:
        POSIX-style path with no leading or trailing slash. Empty, ``None``, and
        ``"."`` normalize to ``""``.

    - POSIX-style
    - no leading slash
    - no trailing slash
    - empty or "." becomes ""

    Examples:
        norm_inside("/a/b/") -> "a/b"
        norm_inside("")      -> ""
        norm_inside(None)    -> ""
    """
    if not inside:
        return ""
    norm = str(PurePosixPath(str(inside))).strip("/")
    return "" if norm in ("", ".") else norm


async def calculate_sha256(file_path: str) -> str:
    """
    Asynchronously calculate the SHA256 checksum of a local file.

    Args:
        file_path: Local filesystem path to read.

    Returns:
        Hex-encoded SHA256 digest as ``str``.

    Used for LFS pointer creation.
    """
    sha = hashlib.sha256()
    async with aiofiles.open(file_path, "rb") as f:
        while True:
            chunk = await f.read(8192)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def lfs_pointer_text(sha: str, size: int) -> str:
    """
    Generate the exact Git LFS pointer file content.

    Args:
        sha: SHA256 object id for the LFS object.
        size: Size of the LFS object in bytes.

    Returns:
        Git LFS pointer file content as ``str``.
    """
    return (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{sha}\n"
        f"size {size}\n"
    )


def parse_lfs_pointer(content: str | bytes) -> tuple[str, int] | None:
    """Return the SHA-256 OID and size from a canonical Git LFS pointer."""
    if isinstance(content, bytes):
        try:
            content = content.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(content, str):
        return None

    match = _LFS_POINTER_RE.fullmatch(content)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def gitattributes_block(path_str: str) -> str:
    """
    Generate a .gitattributes block compatible with the legacy ARC/LFS layout.

    Args:
        path_str: Repository-internal path that should be tracked by LFS.

    Returns:
        ``.gitattributes`` text block as ``str``.

    IMPORTANT:
    This mirrors your old pyfilesystem-based implementation byte-for-byte.
    """
    return (
        "# Leave following line: auto generated\n"
        f"{path_str} filter=lfs diff=lfs merge=lfs -text \n"
    )

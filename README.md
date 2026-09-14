# arcfs-fsspec

Small `fsspec` backend for browsing GitLab repositories like a filesystem.

## What it does

`arcfs-fsspec` lets you access GitLab repositories through an `fsspec` filesystem interface.

Current features:

- list accessible repositories
- list directories inside repositories
- download files
- upload files through a Git LFS-based workflow

This project is mainly intended as a GitLab-backed filesystem layer and as a backend for Galaxy file sources.

## Installation

    pip install arcfs-fsspec

## Basic idea

Repositories are exposed as top-level directories.

Examples:

- repo root: `group/subgroup/my-repo:-:`
- file inside repo: `group/subgroup/my-repo:-:path/to/file.txt`

The `:-:` marker separates the GitLab repository path from the path inside the repository.

## Basic usage

### Sync-style usage

    from arcfs.fs import GitLabARCFileSystem

    fs = GitLabARCFileSystem(
        base_url="https://gitlab.example.com",
        token="YOUR_GITLAB_TOKEN",
        asynchronous=False,
    )

    # List accessible repositories
    print(fs.ls("", detail=False))

    # List files inside a repository
    print(fs.ls("group/subgroup/my-repo", detail=False))

    # Download a file
    fs.get_file(
        "group/subgroup/my-repo/path/to/file.txt",
        "/tmp/file.txt",
    )

    # Upload a file
    fs.put_file(
        "/tmp/upload.txt",
        "group/subgroup/my-repo/path/to/upload.txt",
    )

### Async usage

    import asyncio
    from arcfs.fs import GitLabARCFileSystem

    async def main():
        fs = GitLabARCFileSystem(
            base_url="https://gitlab.example.com",
            token="YOUR_GITLAB_TOKEN",
            asynchronous=True,
        )

        try:
            # List accessible repositories
            repos = await fs._ls("", detail=False)
            print(repos)

            # List files inside a repository
            items = await fs._ls("group/subgroup/my-repo", detail=False)
            print(items)

            # Download a file
            await fs._get_file(
                "group/subgroup/my-repo/path/to/file.txt",
                "/tmp/file.txt",
            )

            # Upload a file
            await fs._put_file(
                "/tmp/upload.txt",
                "group/subgroup/my-repo/path/to/upload.txt",
            )
        finally:
            await fs._close()

    asyncio.run(main())

## Notes

- Listing supports internal paged access and fallback behavior.
- Uploads use a Git LFS pointer workflow.
- One filesystem instance reuses one UUID-based branch for all of its uploads.
- To batch files across separate filesystem instances, pass the same explicit
  `feature_branch` to every instance or upload in the export.
- `feature_branch_prefix` remains accepted as a compatibility alias for
  `feature_branch`.
- The current focus is the filesystem behavior needed for Galaxy integration first.

## Status

Early version, but listing, download, and upload are already implemented.

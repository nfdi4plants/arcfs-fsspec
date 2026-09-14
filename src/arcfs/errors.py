class NonLFSFileError(FileExistsError):
    """The target exists but is not a valid Git LFS pointer."""

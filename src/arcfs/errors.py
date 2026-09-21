class NonLFSFileError(FileExistsError):
    """The target exists but is not a valid Git LFS pointer."""


class RefNotFound(FileNotFoundError):
    """GitLab could not resolve the ref a request named.

    Subclasses ``FileNotFoundError`` on purpose, so a caller written against an earlier version
    keeps behaving as it did while one that cares catches this first. A repository with no
    commits answers this for its own default branch, where treating it as "nothing to replace"
    is the right reading.
    """

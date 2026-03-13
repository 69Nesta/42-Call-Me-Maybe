from .CallMeMaybeError import CallMeMaybeError


class FileError(CallMeMaybeError):
    """Base class for file-related errors in the project."""
    pass


class FileNotFoundError(FileError):
    """Raised when a file is not found.

    Args:
        file_name (str): Path or name of the missing file.
    """
    def __init__(self, file_name: str) -> None:
        super().__init__(
            f'File \'{file_name}\' not found'
        )


class PermissionError(FileError):
    """Raised when permission is denied accessing a file.

    Args:
        file_name (str): Path or name of the file with denied permissions.
    """
    def __init__(self, file_name: str) -> None:
        super().__init__(
            f'Permission denied for file \'{file_name}\''
        )


class NotAFileError(FileError):
    """Raised when a path is not a regular file.

    Args:
        file_name (str): The path that was expected to be a file.
    """
    def __init__(self, file_name: str) -> None:
        super().__init__(
            f'Path \'{file_name}\' is not a file'
        )


class InvalidJsonFileError(FileError):
    """Raised when a JSON file cannot be decoded due to invalid format.

    Args:
        file_name (str): The path to the JSON file with invalid format.
    """
    def __init__(self, file_name: str) -> None:
        super().__init__(
            f'Invalid JSON format in file \'{file_name}\''
        )

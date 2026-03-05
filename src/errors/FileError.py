from .CallMeMaybeError import CallMeMaybeError


class FileError(CallMeMaybeError):
    pass


class FileNotFoundError(FileError):
    def __init__(self, file_name: str) -> None:
        super().__init__(
            f'File \'{file_name}\' not found'
        )


class PermissionError(FileError):
    def __init__(self, file_name: str) -> None:
        super().__init__(
            f'Permission denied for file \'{file_name}\''
        )


class NotAFileError(FileError):
    def __init__(self, file_name: str) -> None:
        super().__init__(
            f'Path \'{file_name}\' is not a file'
        )

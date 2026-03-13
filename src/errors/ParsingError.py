from .CallMeMaybeError import CallMeMaybeError


class ParsingError(CallMeMaybeError):
    """Raised when a file cannot be parsed correctly.

    Args:
        file_name (str): Path to the file being parsed.
        error (Exception): Original exception that caused the parsing error.
    """
    def __init__(self, file_name: str, error: Exception) -> None:
        super().__init__(
            f'Error parsing file \'{file_name}\' : {error}'
        )

from .CallMeMaybeError import CallMeMaybeError


class ParsingError(CallMeMaybeError):
    def __init__(self, file_name: str, error: Exception) -> None:
        super().__init__(
            f'Error parsing file \'{file_name}\' : {error}'
        )

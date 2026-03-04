from .CallMeMaybeError import CallMeMaybeError


class OpeningError(CallMeMaybeError):
    def __init__(self, file_name: str, error: Exception) -> None:
        super().__init__(
            f'Error opening file \'{file_name}\' : {error}'
        )

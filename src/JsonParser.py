from .errors import (
    InvalidJsonFileError,
    FileNotFoundError as _FileNotFoundError,
    PermissionError as _PermissionError,
    NotAFileError
)
from typing import Any
import json


class JsonParser:
    """Load JSON content from a file and expose it via get_data().

    Args:
        file_path (str): Path to the JSON file to load.
        name (str | None): Optional display name used in error messages.

    Attributes:
        file_path (str): The path provided.
        name (str): Display name for the resource.
        data (Any): Parsed JSON content.
    """
    def __init__(self, file_path: str, name: str | None = None) -> None:
        self.file_path: str = file_path
        self.name: str = name if name else file_path
        self.data: Any = self.load_json_file(file_path)

    def load_json_file(self, file_path: str) -> Any:
        """Read and parse a JSON file, translating IO errors.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            Any: The decoded JSON content.

        Raises:
            FileNotFoundError, PermissionError, NotAFileError,
            InvalidJsonFileError: Translated project-specific errors.
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise _FileNotFoundError(self.name)
        except PermissionError:
            raise _PermissionError(self.name)
        except IsADirectoryError:
            raise NotAFileError(self.name)
        except json.JSONDecodeError:
            raise InvalidJsonFileError(self.file_path)

    def get_data(self) -> Any:
        """Return the parsed JSON content.

        Returns:
            Any: Parsed JSON data previously loaded.
        """
        return self.data

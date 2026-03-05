from .errors import (
    ParsingError,
    FileNotFoundError as _FileNotFoundError,
    PermissionError as _PermissionError,
    NotAFileError
)
from typing import Any
import json


class JsonParser:
    def __init__(self, file_path: str, name: str | None = None) -> None:
        self.file_path: str = file_path
        self.name: str = name if name else file_path
        self.data: Any = self.load_json_file(file_path)

    def load_json_file(self, file_path: str) -> Any:
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise _FileNotFoundError(self.name)
        except PermissionError:
            raise _PermissionError(self.name)
        except IsADirectoryError:
            raise NotAFileError(self.name)
        except json.JSONDecodeError as e:
            raise ParsingError(self.name, e)

    def get_data(self) -> Any:
        return self.data

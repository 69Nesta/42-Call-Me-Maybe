from .errors import (
    NotAFileError,
    PermissionError as _PermissionError,
    FileNotFoundError as _FileNotFoundError
)
from pydantic import BaseModel, ValidationError, Field, PrivateAttr
from typing import Any, Callable
from .utils import Logger, Color
import json


class FunctionCallingTest(BaseModel):
    prompt: str = Field(...)


class CallingTests(BaseModel):
    file_path: str = Field(...)
    prompt_function: Callable[[str], None] = Field(...)

    _logger: Logger = PrivateAttr()
    _content: list[FunctionCallingTest] = PrivateAttr([])

    def model_post_init(self, context: Any) -> None:
        self._logger = Logger(
            ACTIVE=True,
            name='CallingTests',
            color=Color.BRIGHT_BLUE
        )

        try:
            self.parse()
        except FileNotFoundError:
            raise _FileNotFoundError(self.file_path)
        except PermissionError:
            raise _PermissionError(self.file_path)
        except IsADirectoryError:
            raise NotAFileError(self.file_path)

        super().model_post_init(context)

    def parse(self) -> None:
        with open(self.file_path, 'r') as f:
            content = f.read().strip()

            if not content:
                self._content = []
                return

            items: list[dict[str, str]] = json.loads(content)

            try:
                self._content = [
                    FunctionCallingTest(**item) for item in items
                ]
            except ValidationError as e:
                self._logger.error(
                    f"Error in file {self.file_path}:"
                )
                for error in e.errors():
                    location = " -> ".join(str(loc) for loc in error["loc"])
                    self._logger.error(f"Field: {location}")
                    self._logger.error(f"Error: {error['msg']}")
                    self._logger.error(f"Type: {error['type']}")
                    self._logger.error("-" * 40)

                raise ValueError("Invalid function definition format")

    def run_tests(self) -> None:
        for test in self._content:
            self._logger.log(f"Running test: {test.prompt}")
            try:
                self.prompt_function(test.prompt)
            except Exception as e:
                self._logger.error(f"Error running test: {e}")

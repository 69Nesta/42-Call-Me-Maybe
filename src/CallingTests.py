"""Load and run function-calling tests from a JSON file.

The CallingTests module reads a list of prompts from a JSON file and uses a
provided prompt handler (the AI) to run each test while reporting progress
and basic timing statistics.
"""

from .errors import (
    NotAFileError,
    PermissionError as _PermissionError,
    FileNotFoundError as _FileNotFoundError,
    InvalidJsonFileError
)
from pydantic import BaseModel, ValidationError, Field, PrivateAttr
from typing import Any, Callable
from .utils import Logger, Color, ProgressBar, StepName
from .OutputFile import OutputPrompt
import time
import json


class FunctionCallingTest(BaseModel):
    """Simple model representing a single function-calling test.

    Attributes:
        prompt (str): The user prompt to feed to the prompt function.
    """
    prompt: str = Field(...)


class CallingTests(BaseModel):
    """Runner that executes a collection of function-calling tests.

    The runner loads tests from a JSON file, runs them using the provided
    prompt function, updates a progress bar and logs timing statistics.

    Attributes:
        file_path (str): Path to the JSON test file to load.
        prompt_function (Callable[[str], OutputPrompt]): Callable used to
            process a prompt and return an OutputPrompt.
        progress_bar (ProgressBar): ProgressBar instance used during runs.
        verbose (bool): Enable verbose logging.
    """
    file_path: str = Field(...)
    prompt_function: Callable[[str], OutputPrompt] = Field(...)
    progress_bar: ProgressBar = Field(...)
    verbose: bool = Field(default=False)

    _logger: Logger = PrivateAttr()
    _content: list[FunctionCallingTest] = PrivateAttr([])

    def model_post_init(self, context: Any) -> None:
        """Initialize logger and parse the test file.

        Args:
            context (Any): Pydantic post-init context (ignored).

        Raises:
            FileNotFoundError, PermissionError, NotAFileError,
            InvalidJsonFileError: When the test file cannot be accessed or is
            malformed.
        """
        self._logger = Logger(
            ACTIVE=self.verbose,
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
        except json.JSONDecodeError:
            raise InvalidJsonFileError(self.file_path)

        super().model_post_init(context)

    def parse(self) -> None:
        """Read and validate the JSON list of test prompts.

        Raises:
            ValueError: If individual test entries are invalid.
        """
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

    def get_tests(self) -> list[FunctionCallingTest]:
        """Return the loaded tests.

        Returns:
            list[FunctionCallingTest]: List of test models loaded from disk.
        """
        return self._content

    def print_stats(self) -> None:
        """Log simple statistics about the loaded tests.

        No return value; info messages are printed via the logger.
        """
        total_tests: int = len(self._content)

        self._logger.info(
            f"Loaded {total_tests} tests from '{self.file_path}'"
        )
        self._logger.info("Statistics:")
        self._logger.info(f" - Total tests: {total_tests}")

    def run_tests(self) -> None:
        """Execute all loaded tests using the configured prompt function.

        The method updates the progress bar and logs elapsed time. Individual
        test failures are caught and logged so the runner can continue.
        """
        self.print_stats()

        self._logger.info('Running function calling tests...')
        print()
        self.progress_bar.set_total(len(self._content))

        start: float = time.time()
        for test in self._content:
            self._logger.log(f"Running test: {test.prompt}")
            try:
                self.prompt_function(test.prompt)
                self.progress_bar.update(1, StepName.NEXT_PROMPT)
            except Exception as e:
                self._logger.error(f"Error running test: {e}")

        self.progress_bar.update(0, StepName.FINISHED)
        self.progress_bar.end()

        end: float = time.time()
        print()
        self._logger.info(
            f"Finished running tests in {end - start:.2f} seconds"
        )

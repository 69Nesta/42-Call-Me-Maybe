from pydantic import BaseModel, ValidationError, Field, PrivateAttr
from .utils import Logger, Color
from .FunctionDefinitions import FunctionDefinition
from .errors import NotAFileError, PermissionError as _PermissionError
from typing import Any
import json
import os


t_parameters = dict[str, str | float | int | bool]
t_prompt_output_content = dict[str, str | t_parameters]
t_ouput_content = list[t_prompt_output_content]


class OutputPrompt(BaseModel):
    """Model describing a single persisted prompt output.

    Attributes:
        prompt (str): Original prompt used to generate the function call.
        name (str): Name of the function selected.
        parameters (dict): Parameters that were extracted for the call.
    """
    prompt: str = Field(
        ...,
        description='The prompt used to generate the output'
    )
    name: str = Field(
        ...,
        description='The name of the function used to generate the output'
    )
    parameters: t_parameters = Field(
        ...,
        description='The parameters used to generate the output'
    )


class OutputFile(BaseModel):
    """Manage loading and saving of a collection of OutputPrompt items.

    Attributes:
        file_path (str): Path to the output JSON file.
        verbose (bool): Enable verbose logging.
    """
    file_path: str = Field(..., description='The path of the output file')
    verbose: bool = Field(default=False)

    _logger: Logger = PrivateAttr()
    _content: list[OutputPrompt] = PrivateAttr([])

    def model_post_init(self, _: Any) -> None:
        """Initialize logger and attempt to parse existing output file.

        Args:
            _ (Any): Pydantic post-init context (ignored).

        Raises:
            PermissionError: If the file exists but cannot be accessed.
            NotAFileError: If the path points to a directory.
        """
        self._logger = Logger(
            ACTIVE=self.verbose,
            name='OutputFile',
            color=Color.BLUE
        )

        try:
            self.parse()
        except FileNotFoundError:
            self._logger.log(
                f'Output file not found at {self.file_path}, '
                'creating a new one.'
            )
            self.save()
        except PermissionError:
            raise _PermissionError(self.file_path)
        except IsADirectoryError:
            raise NotAFileError(self.file_path)

    def parse(self) -> None:
        """Load and validate existing output items from disk.

        If the file is empty or does not contain a list of items the
        internal content will be left empty.
        """
        with open(self.file_path, "r") as f:
            content: str = f.read().strip()
            self._content = []

            if not content:
                return

            items: list[dict[str, Any]] = json.loads(content)

            if not isinstance(items, list):
                return

            self._logger.log(
                f'Parsing {len(items)} items...'
            )
            for item in items:
                try:
                    self._content.append(OutputPrompt(**item))
                except ValidationError:
                    self._logger.log('Skipping invalid item')

    def add_prompt(
                self,
                prompt: str,
                function: FunctionDefinition
            ) -> OutputPrompt:
        """Append a new OutputPrompt built from a FunctionDefinition.

        Args:
            prompt (str): Original user prompt.
            function (FunctionDefinition): FunctionDefinition populated with
                extracted parameter values.

        Returns:
            OutputPrompt: The appended OutputPrompt instance.
        """
        self._content.append(
            OutputPrompt(
                prompt=prompt,
                name=function.name,
                parameters={
                    parameter.name: parameter.value
                    for parameter in function.parameters.values()
                    if parameter.value is not None
                }
            )
        )

        return self._content[-1]

    def save(self) -> None:
        """Persist the current collection of OutputPrompt items to disk.

        The output directory will be created if it does not exist.
        """
        self._logger.log('Saving...')

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        with open(self.file_path, 'w') as f:
            json.dump(
                [
                    user.model_dump()
                    for user in self._content
                ],
                f,
                indent=4
            )

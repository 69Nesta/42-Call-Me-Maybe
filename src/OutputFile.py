from pydantic import BaseModel, ValidationError, Field, PrivateAttr
from .utils import Logger, Color
from .FunctionDefinitions import FunctionDefinition
from .errors import NotAFileError, PermissionError as _PermissionError
from typing import Any
import json


t_parameters = dict[str, str | float | int]
t_prompt_output_content = dict[str, str | t_parameters]
t_ouput_content = list[t_prompt_output_content]


class OutputPrompt(BaseModel):
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
    file_path: str = Field(..., description='The path of the output file')

    _logger: Logger = PrivateAttr()
    _content: list[OutputPrompt] = PrivateAttr([])

    def model_post_init(self, _: Any) -> None:
        self._logger = Logger(name='OutputFile', color=Color.BLUE)

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
        self._logger.log('Saving...')
        with open(self.file_path, 'w') as f:
            json.dump(
                [
                    user.model_dump()
                    for user in self._content
                ],
                f,
                indent=4
            )

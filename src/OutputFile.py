from pydantic import BaseModel, Field, PrivateAttr
from .utils import Logger, Color
from .FunctionDefinitions import FunctionDefinition
from .errors import NotAFileError, PermissionError as _PermissionError
from typing import Any
import json


t_parameters = dict[str, str | float]
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
            # self._logger.error(
            #     f'Permission denied for file {self.file_path}.'
            # )
            raise _PermissionError(self.file_path)
        except IsADirectoryError:
            # self._logger.error(
            #     f'The path {self.file_path} is not a file.'
            # )
            raise NotAFileError(self.file_path)

    def parse(self) -> None:
        with open(self.file_path, "r") as f:
            content = f.read().strip()

            if not content:
                self._content = []
                return

            items: list[dict[str, Any]] = json.loads(content)

            if isinstance(items, list):
                try:
                    self._content = [OutputPrompt(**item) for item in items]
                except Exception as e:
                    self._logger.error(f"Error parsing output file: {e}")

    def add_prompt(
                self,
                prompt: str,
                function: FunctionDefinition
            ) -> None:
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

    def save(self) -> None:
        with open(self.file_path, 'w') as f:
            json.dump(
                [
                    user.model_dump()
                    for user in self._content
                ],
                f,
                indent=4
            )

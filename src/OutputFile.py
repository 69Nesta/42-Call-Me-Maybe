from pydantic import BaseModel, Field
from .utils import Logger, Color
from .FunctionDefinitions import FunctionDefinition
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
    logger: Logger = Field(...)

    content: list[OutputPrompt] = []

    def model_post_init(self, __context: Any) -> None:
        try:
            self.parse()
        except FileNotFoundError:
            self.logger.log(
                f'Output file not found at {self.file_path}, '
                'creating a new one.'
            )
            self.save()

    def parse(self) -> None:
        with open(self.file_path, 'r') as f:
            items: list[dict] = json.load(f)
            if isinstance(items, list):
                try:
                    self.content = [OutputPrompt(**item) for item in items]
                except Exception as e:
                    self.logger.error(f'Error parsing output file: {e}')

    def add_prompt(
                self,
                prompt: str,
                function: FunctionDefinition
            ) -> None:
        self.content.append(
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
                    for user in self.content
                ],
                f,
                indent=4
            )

from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from llm_sdk import Small_LLM_Model  # type: ignore
from .JsonParser import JsonParder
from .utils import Color, Logger
import numpy as np


AllowedType = Literal['string', 'number']


class Parameter(BaseModel):
    type: AllowedType = Field(
        ...,
        description="Type of the parameter (e.g., number, string)"
    )
    name: str = Field(
        '',
        description="Name of the parameter"
    )
    value: str | float | None = Field(
        None,
        description='The value of the parameter, can be set later'
    )


class ReturnSchema(BaseModel):
    type: AllowedType = Field(
        ...,
        description="Return type"
    )


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: ReturnSchema


class FunctionDefinitions:
    def __init__(self, model: Small_LLM_Model, file_path: str) -> None:
        self.logger = Logger(name='FuncDefs', color=Color.YELLOW)
        self.model: Small_LLM_Model = model
        self.functions_definition_path: str
        self.functions_definition_parser: JsonParder
        self.functions_definition: list[FunctionDefinition]

        self.load(file_path)

        self.functions_inputs: dict[str, list[int]] = {}

    def load(self, file_path: str) -> None:
        self.logger.log('Loading functions definition...')
        self.functions_definition_path = file_path
        self.functions_definition_parser = JsonParder(file_path)
        functions_definition = self.functions_definition_parser.get_data()

        try:
            self.functions_definition = self.parser(functions_definition)
            self.logger.log('Functions definition loaded.')
        except ValueError as e:
            raise e

    def parser(self, raw_functions: list[dict]) -> list[FunctionDefinition]:
        self.logger.log('Parsing functions definition...')

        try:
            functions: list[FunctionDefinition] = [
                FunctionDefinition.model_validate(item)
                for item in raw_functions
            ]

            for fn in functions:
                for name, param in fn.parameters.items():
                    param.name = name

            return functions

        except ValidationError as e:
            self.logger.error(
                f"Error in file {self.functions_definition_path}:"
            )
            for error in e.errors():
                location = " -> ".join(str(loc) for loc in error["loc"])
                self.logger.error(f"Field: {location}")
                self.logger.error(f"Error: {error['msg']}")
                self.logger.error(f"Type: {error['type']}")
                self.logger.error("-" * 40)

            raise ValueError("Invalid function definition format")

    def get_names(self) -> list[str]:
        return [function.name for function in self.functions_definition]

    def get_names_inputs(self) -> dict[str, list[int]]:
        if not self.functions_inputs.keys():
            self.logger.log('Encoding functions name definition...')
            for function in self.functions_definition:
                function_input_ids: list[int] = self.model.encode(
                    function.name
                ).tolist()
                self.functions_inputs.update({
                    function.name: np.concatenate(
                        function_input_ids
                    ).ravel().tolist()
                })

        return self.functions_inputs

    def get_names_inputs_with(self, start: list[int]) -> dict[str, list[int]]:
        self.logger.log(
            f'Getting functions name definition with start: {start}...'
        )
        names_inputs: dict[str, list[int]] = self.get_names_inputs()
        names_inputs_with: dict[str, list[int]] = {}
        for name, input_ids in names_inputs.items():
            if input_ids[:len(start)] == start:
                names_inputs_with.update({name: input_ids})
        return names_inputs_with

    def get_by_name(self, name: str) -> FunctionDefinition:
        function = next(x for x in self.functions_definition if x.name == name)
        if function:
            return function
        raise ValueError(f'Function definition with name \'{name}\' not found')

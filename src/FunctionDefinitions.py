from pydantic import BaseModel, Field, ValidationError, ConfigDict, PrivateAttr
from typing import Literal, Any
from llm_sdk import Small_LLM_Model  # type: ignore
from .JsonParser import JsonParser
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


class FunctionDefinitions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model = Field(
        ...,
        description='The language model used to encode the function names',
    )
    file_path: str = Field(
        ...,
        description='The path to the functions definition file'
    )

    _logger: Logger = PrivateAttr()
    _parser: JsonParser = PrivateAttr()
    _functions_definition: list[FunctionDefinition] = PrivateAttr()
    _functions_inputs: dict[str, list[int]] = PrivateAttr()

    def model_post_init(self, _: Any) -> None:
        self._logger = Logger(name='FuncDefs', color=Color.YELLOW)

        self.load(self.file_path)

        self._functions_inputs: dict[str, list[int]] = {}

    def load(self, file_path: str) -> None:
        self._logger.log('Loading functions definition...')
        self._parser = JsonParser(file_path)

        try:
            self._functions_definition = self.parser(self._parser.get_data())
            self._logger.log('Functions definition loaded.')
        except ValueError as e:
            raise e

        if len(self._functions_definition) < 2:
            raise ValueError(
                'Not enough functions defined, at least '
                '2 functions are required.'
            )

    def parser(
                self,
                raw_functions: list[dict[str, str | dict[str, dict[str, str]]]]
            ) -> list[FunctionDefinition]:
        self._logger.log('Parsing functions definition...')

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

    def get_functions_definition(self) -> list[FunctionDefinition]:
        return self._functions_definition

    def get_names(self) -> list[str]:
        return [function.name for function in self._functions_definition]

    def get_names_inputs(self) -> dict[str, list[int]]:
        if not self._functions_inputs.keys():
            self._logger.log('Encoding functions name definition...')
            for function in self._functions_definition:
                function_input_ids: list[int] = self.model.encode(
                    function.name
                ).tolist()
                self._functions_inputs.update({
                    function.name: np.concatenate(
                        function_input_ids
                    ).ravel().tolist()
                })

        return self._functions_inputs

    def get_names_inputs_with(self, start: list[int]) -> dict[str, list[int]]:
        self._logger.log(
            f'Getting functions name definition with start: {start}...'
        )
        names_inputs: dict[str, list[int]] = self.get_names_inputs()
        names_inputs_with: dict[str, list[int]] = {}
        for name, input_ids in names_inputs.items():
            if input_ids[:len(start)] == start:
                names_inputs_with.update({name: input_ids})
        return names_inputs_with

    def get_by_name(self, name: str) -> FunctionDefinition:
        function = next(
            x
            for x in self._functions_definition
            if x.name == name
        )
        if function:
            return function
        raise ValueError(f'Function definition with name \'{name}\' not found')

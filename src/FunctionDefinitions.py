from pydantic import BaseModel, Field, ValidationError
from llm_sdk import Small_LLM_Model
from .JsonParser import JsonParder
from .utils import Color, Logger
import numpy as np


class FunctionParameter(BaseModel):
    name: str = Field(..., description='The name of the parameter')
    type: str = Field(..., description='The type of the parameter')


class FunctionDefinition(BaseModel):
    name: str = Field(
        ...,
        description='The name of the function'
    )
    description: str = Field(
        ...,
        description='A brief description of the function'
    )
    parameters: dict[str, FunctionParameter] = Field(
        ...,
        description='A dictionary of parameter names and their types'
    )


t_functions_definiton = list[FunctionDefinition]


class FunctionDefinitions:
    def __init__(self, model: Small_LLM_Model, file_path: str) -> None:
        self.logger = Logger('FuncDefs', Color.YELLOW)
        self.model: Small_LLM_Model = model
        self.functions_definition_path: str
        self.functions_definition_parser: JsonParder
        self.functions_definition: t_functions_definiton

        self.load(file_path)

        self.functions_inputs: dict[str, list[int]] = {}

    def load(self, file_path: str) -> None:
        self.logger.log('Loading functions definition...')
        self.functions_definition_path = file_path
        self.functions_definition_parser = JsonParder(file_path)
        functions_definition = self.functions_definition_parser.get_data()

        try:
            self.parser(functions_definition)
        except ValidationError as e:
            message: str = 'Error parsing functions definition:'
            self.logger.error(message)

            for error in list(e.errors()):
                if error['loc']:
                    current_message = f" - {error['loc'][0]}: {error['msg']}"
                    message += f"\n{current_message}"
                    self.logger.error(current_message)

            raise ValueError(message)
        self.logger.log('Functions definition loaded.')

    def parser(self, functions_definition: list[dict]) -> None:
        self.logger.log('Parsing functions definition...')
        self.functions_definition = []
        for function in functions_definition:
            name: str = function.get('name')
            description: str = function.get('description')
            parameters: dict[str, str] = function.get('parameters')

            # self.logger.log(f'Function: {name}')
            # self.logger.log(f'Description: {description}')
            # self.logger.log('Parameters:')
            parameters_parsed: dict[str, FunctionParameter] = {}
            for param_name, param_type in parameters.items():
                parameters_parsed.update({
                    param_name: FunctionParameter(
                        name=param_name,
                        type=param_type.get('type')
                    )
                })
                # self.logger.log(f' - {param_name}: {param_type.get("type")}')
            self.functions_definition.append(
                FunctionDefinition(
                    name=name,
                    description=description,
                    parameters=parameters_parsed
                )
            )

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

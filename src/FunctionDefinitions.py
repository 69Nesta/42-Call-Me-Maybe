from pydantic import BaseModel, Field, ValidationError, ConfigDict, PrivateAttr
from typing import Literal, Any, Callable
from .JsonParser import JsonParser
from .utils import Color, Logger


AllowedType = Literal['string', 'number', 'integer', 'boolean', 'float']


class Parameter(BaseModel):
    """Model describing a single function parameter.

    Attributes:
        type (AllowedType): The type of the parameter (string, number, etc.).
        name (str): The name of the parameter.
        value (str | float | bool | None): The value for the parameter. This
            may be set later during extraction.
    """
    type: AllowedType = Field(
        ...,
        description="Type of the parameter (e.g., number, string)"
    )
    name: str = Field(
        '',
        description="Name of the parameter"
    )
    value: str | float | bool | None = Field(
        None,
        description='The value of the parameter, can be set later'
    )


class ReturnSchema(BaseModel):
    """Model describing the return schema of a function.

    Attributes:
        type (AllowedType): The return type of the function.
    """
    type: AllowedType = Field(
        ...,
        description="Return type"
    )


class FunctionDefinition(BaseModel):
    """Representation of a function definition.

    Attributes:
        name (str): Function name.
        description (str): Function description.
        parameters (dict[str, Parameter]): Mapping from parameter name to
            Parameter model.
        returns (ReturnSchema): The return schema for the function.
    """
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: ReturnSchema


class FunctionDefinitions(BaseModel):
    """Loader and container for all function definitions.

    This class loads function definitions from a JSON file, validates them
    and exposes helpers to retrieve names and encoded inputs.

    Attributes:
        encode_function (Callable[[str], list[int]]): Function used to
            encode function names to token ids.
        file_path (str): Path to the functions definition JSON file.
        verbose (bool): Whether to enable verbose logging.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    encode_function: Callable[[str], list[int]] = Field(
        ...,
        description='The function used to encode the function names'
    )

    file_path: str = Field(
        ...,
        description='The path to the functions definition file'
    )

    verbose: bool = Field(default=False)

    _logger: Logger = PrivateAttr()
    _parser: JsonParser = PrivateAttr()
    _functions_definition: list[FunctionDefinition] = PrivateAttr()
    _functions_inputs: dict[str, list[int]] = PrivateAttr()

    def model_post_init(self, _: Any) -> None:
        """Initialize internal logger and load the definitions.

        Args:
            _ (Any): Pydantic post-init context (ignored).
        """
        self._logger = Logger(
            ACTIVE=self.verbose,
            name='FuncDefs',
            color=Color.YELLOW
        )

        self.load(self.file_path)

        self._functions_inputs: dict[str, list[int]] = {}

    def load(self, file_path: str) -> None:
        """Load and validate functions definition from a JSON file.

        Args:
            file_path (str): Path to the JSON file containing function defs.

        Raises:
            ValueError: If parsing or validation fails or if not enough
                functions are provided.
        """
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
        """Parse raw JSON data into FunctionDefinition models.

        Args:
            raw_functions (list[dict]): Raw list of function definition
                dictionaries as read from the JSON file.

        Returns:
            list[FunctionDefinition]: Validated list of function models.

        Raises:
            ValueError: If the input format is invalid or validation fails.
        """
        self._logger.log('Parsing functions definition...')

        try:
            functions: list[FunctionDefinition] = [
                FunctionDefinition.model_validate(item)
                for item in raw_functions
            ]

            functions_names = [function.name for function in functions]
            if len(functions_names) != len(set(functions_names)):
                raise ValueError('Function names must be unique.')

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
        """Return the loaded function definitions.

        Returns:
            list[FunctionDefinition]: The list of loaded function models.
        """
        return self._functions_definition

    def get_names(self) -> list[str]:
        """Return the list of function names.

        Returns:
            list[str]: Function names in the loaded definitions.
        """
        return [function.name for function in self._functions_definition]

    def get_names_inputs(self) -> dict[str, list[int]]:
        """Return a mapping of function names to their encoded token ids.

        The encoding is performed once and cached for subsequent calls.

        Returns:
            dict[str, list[int]]: Mapping of function name to encoded ids.
        """
        if not self._functions_inputs.keys():
            self._logger.log('Encoding functions name definition...')
            for function in self._functions_definition:
                function_input_ids: list[int] = self.encode_function(
                    function.name
                )
                self._functions_inputs.update({
                    function.name: function_input_ids
                })

        return self._functions_inputs

    def get_names_inputs_with(self, start: list[int]) -> dict[str, list[int]]:
        """Return functions whose encoded ids start with a prefix.

        Args:
            start (list[int]): Token id prefix to filter function names.

        Returns:
            dict[str, list[int]]: Mapping of matching function names to ids.
        """
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
        """Return a FunctionDefinition by name.

        Args:
            name (str): Function name to retrieve.

        Returns:
            FunctionDefinition: The matching function model.

        Raises:
            ValueError: If no function with the given name exists.
        """
        function = next(
            x
            for x in self._functions_definition
            if x.name == name
        )
        if function:
            return function
        raise ValueError(f'Function definition with name \'{name}\' not found')

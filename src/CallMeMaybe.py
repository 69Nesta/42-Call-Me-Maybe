from llm_sdk import Small_LLM_Model  # type: ignore
from .FunctionDefinitions import (
    FunctionDefinitions,
    FunctionDefinition,
    Parameter
)
from .utils import Logger, Color, ProgressBar, StepName
from .OutputFile import OutputFile, OutputPrompt
from .Vocabulary import Vocabulary
from pydantic import BaseModel, Field, PrivateAttr
from typing import Any, ClassVar
import numpy as np
import re


class CallMeMaybe(BaseModel):
    """Main class that handles function extraction from text prompts.

    This class wraps an LLM model and supporting utilities to:
    - Build preprompts describing available functions
    - Encode/decode tokens
    - Identify the best-matching function name from model logits
    - Extract typed parameters (number, string, boolean)
    - Save the extracted function call to an output file

    Attributes:
        functions_definition_path (str): Path to the JSON file
            defining functions.
        output_file_path (str): Path to the output file to save extracted
            prompts.
        progress_bar (ProgressBar): Progress bar used while running tests.
        model_name (str | None): Optional model name to use for the LLM.
        cache_dir (str | None): Optional cache directory for the LLM model.
        verbose (bool): Whether to enable verbose logging.
    """

    functions_definition_path: str = Field(
        ...,
        description='The path to the _functions definition file'
    )
    output_file_path: str = Field(
        ...,
        description='The path to the output file where the prompts and their '
                    'extracted _functions will be saved'
    )
    progress_bar: ProgressBar = Field(...)

    model_name: str | None = Field(
        default=None,
        description='The name of the model to use, if None the default'
    )

    cache_dir: str | None = Field(
        default=None,
        description='The directory where the model will be cached, if None the'
                    ' default cache directory will be used'
    )

    verbose: bool = Field(default=False)

    NUMBER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?:\d+(?:\.\d*)?|\.\d+|\.)$"
    )

    MIN_CONFIDENCE_THRESHOLD: ClassVar[float] = 24.0
    CONFIDENCE_CHECK_ITERATION: ClassVar[int] = 1
    MAX_LOGITS_TO_CHECK: ClassVar[int] = 10
    STRING_TERMINATORS: ClassVar[set[str]] = {
        '"', "\\'", "'", '\"', '\n', '"\n', "'\n"
    }
    MAX_LOGITS_TO_CHECK_FIRST: ClassVar[int] = 3
    MAX_LOGITS_TO_CHECK_OTHERS: ClassVar[int] = 1

    _logger: Logger = PrivateAttr()
    _model: Small_LLM_Model = PrivateAttr()
    _output_file: OutputFile = PrivateAttr()
    _functions: FunctionDefinitions = PrivateAttr()
    _vocab: Vocabulary = PrivateAttr()
    _termonators: list[int] = PrivateAttr()

    def model_post_init(self, _: Any) -> None:
        """Initialize runtime resources after model creation.

        This initializes the logger, LLM model, output file, function
        definitions, vocabulary and terminators.

        Args:
            _ (Any): Pydantic post-init context (ignored).

        Raises:
            ValueError: If the LLM model cannot be initialized.
        """

        self._logger: Logger = Logger(
            ACTIVE=self.verbose,
            name='Core',
            color=Color.CYAN
        )

        try:
            self._logger.log('Initializing LLM...')
            self._model: Small_LLM_Model = Small_LLM_Model(
                model_name=self.model_name,
                cache_dir=self.cache_dir
            )
            self._logger.log('Model initialized.')
        except Exception as e:
            raise ValueError(f'Error initializing model: {e}')

        self._output_file: OutputFile = OutputFile(
            file_path=self.output_file_path,
            verbose=self.verbose
        )

        self._functions = FunctionDefinitions(
            encode_function=self.encode,
            file_path=self.functions_definition_path,
            verbose=self.verbose
        )

        self._vocab = Vocabulary(
            file_path=self._model.get_path_to_vocab_file(),
            verbose=self.verbose
        )

        for key, value in self._functions.get_names_inputs().items():
            self._logger.log(
                f'Function \'{key}\' encoded with input ids: {value}'
            )

        self.create_index_terminators()

    def create_index_terminators(self) -> list[int]:
        """Create and return a list of token ids that terminate strings.

        The terminators are derived from a set of string terminator tokens
        and encoded with the model. The first token id for each terminator
        is collected.

        Returns:
            list[int]: List of token ids that represent string terminators.
        """

        self._logger.log('Creating terminators ids...')
        self._termonators = []

        for terminator in self.STRING_TERMINATORS:
            terminator_id: list[int] = self.encode(terminator)
            if terminator_id is not None and len(terminator_id) > 0 and \
                    terminator_id[0] not in self._termonators:
                self._termonators.append(terminator_id[0])
                self._logger.log(
                    f'Terminator \'{terminator}\' encoded with id: '
                    f'{terminator_id[0]}'
                )

        return self._termonators

    def create_preprompt(self, user_prompt: str) -> str:
        """Build a human-readable preprompt describing available functions.

        Args:
            user_prompt (str): The original user prompt.

        Returns:
            str: A preprompt string that lists available functions and the
            user query used to prime the model for function extraction.
        """

        preprompt: str = 'Extract the function for the user query\n'
        functuions_list: str = ', '.join(self._functions.get_names())

        preprompt += f'functions available: {functuions_list}\n'
        preprompt += f'query: {user_prompt}\n'
        preprompt += 'function to use: '

        return preprompt

    def get_preprompt(self, user_prompt: str) -> list[int]:
        """Encode the preprompt for a given user prompt.

        Args:
            user_prompt (str): The original user prompt.

        Returns:
            list[int]: The encoded token ids for the preprompt.
        """

        prompt: str = self.create_preprompt(user_prompt)

        self._logger.log(f'Preprompt:\n{prompt}')

        return self.encode(prompt)

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids using the underlying LLM.

        Args:
            text (str): Text to encode.

        Returns:
            list[int]: Flattened list of token ids representing the text.
        """

        input_ids_2d: list[list[int]] = self._model.encode(text).tolist()
        return np.concatenate(input_ids_2d).ravel().tolist()

    def decode(self, input_ids: list[int]) -> str:
        """Decode token ids back into text using the LLM.

        Args:
            input_ids (list[int]): List of token ids to decode.

        Returns:
            str: Decoded text string.
        """

        return str(self._model.decode(input_ids))

    def _get_available_function_logits(
        self,
        prompt_ids_2d: list[int],
        function_name_ids: list[int]
    ) -> list[tuple[int, float]]:
        """Obtain logits for function-name continuation tokens.

        Args:
            prompt_ids_2d (list[int]): Current encoded prompt ids.
            function_name_ids (list[int]): Encoded token ids for function
                name prefix already built so far.

        Returns:
            list[tuple[int, float]]: A list of tuples (token_id, logit)
            sorted by descending logit for candidate function name tokens.
        """

        logits: list[float] = self._model.get_logits_from_input_ids(
            prompt_ids_2d
        )

        available_functions_inputs: list[int] = [
            inputs[len(function_name_ids)]
            for _, inputs in self._functions.get_names_inputs_with(
                function_name_ids
            ).items()
            if len(inputs) > len(function_name_ids)
        ]

        availables_functions_logits: list[tuple[int, float]] = [
            (index, logits[index]) for index in available_functions_inputs
        ]

        return sorted(
            availables_functions_logits,
            key=lambda x: x[1],
            reverse=True
        )

    def _validate_function_confidence(
                self,
                iteration: int,
                probability: float,
                prompt: str
            ) -> None:
        """Validate that the model is confident enough about a function.

        Args:
            iteration (int): Current iteration index while decoding name.
            probability (float): Confidence/probability score for the token.
            prompt (str): Original user prompt used for context.

        Raises:
            ValueError: If the model is not confident enough to select a
                function after a critical iteration.
        """

        if (iteration == self.CONFIDENCE_CHECK_ITERATION
           and probability < self.MIN_CONFIDENCE_THRESHOLD):
            raise ValueError(
                f'Function not found in {self._functions.get_names()} '
                f'with enough confidence to answer the prompt: \'{prompt}\'.'
            )

    def _extract_function(
                self,
                prompt: str,
                user_prompt_ids: list[int],
                prompt_ids_2d: list[int]
            ) -> FunctionDefinition:
        """Extract the best matching function definition for a prompt.

        This method iteratively selects token ids that maximize the logits
        for function names until the longest function name length is
        reached or no candidates remain.

        Args:
            prompt (str): The original user prompt.
            user_prompt_ids (list[int]): Encoded token ids of the user prompt.
            prompt_ids_2d (list[int]): Encoded token ids for the preprompt.

        Returns:
            FunctionDefinition: The matched FunctionDefinition object.
        """

        self.progress_bar.update(0, StepName.EXTRACTING_FUNCTION)
        function_name_ids: list[int] = []
        self._logger.log(
            f'{Color.GREEN}User prompt ids: {user_prompt_ids}{Color.RESET}'
        )

        longest_function_name: list[int] = max(
            self._functions.get_names_inputs().values(),
            key=lambda x: len(x)
        )

        self._logger.log(
            f'Finding function name with {len(longest_function_name)} '
            'iterations...'
        )

        for i in range(len(longest_function_name)):
            sorted_functions_logits = self._get_available_function_logits(
                prompt_ids_2d, function_name_ids
            )

            self._logger.log(
                f'Available functions logits: {sorted_functions_logits}'
            )

            if not sorted_functions_logits:
                self._logger.log('No available functions logits.')
                break

            best_logits, best_logits_proba = sorted_functions_logits[0]
            self._validate_function_confidence(i, best_logits_proba, prompt)

            function_name_ids.append(best_logits)
            prompt_ids_2d.append(best_logits)

        function_name: str = self.decode(function_name_ids)
        self._logger.log(
            f'{Color.BRIGHT_BLUE}{Color.BOLD}Function name found: '
            f'\'{function_name}\'{Color.RESET}'
        )

        return self._functions.get_by_name(function_name)

    def _extract_number_parameter(
                self,
                sorted_logits_index: list[int],
                parameter_ids: list[int]
            ) -> tuple[float | int | None, int]:
        """Extract a numeric parameter value from model logits.

        Args:
            sorted_logits_index (list[int]): Candidate token ids sorted by
                descending logits.
            parameter_ids (list[int]): Token ids collected so far for this
                parameter value.

        Returns:
            tuple[float | int | None, int]: A tuple of (number_value or None,
            token_id used). If no number could be extracted None is returned
            as the first element.
        """

        best_logits: int = sorted_logits_index.pop(0)

        number_try: int = self.MAX_LOGITS_TO_CHECK_FIRST
        if len(parameter_ids) > 0:
            number_try = self.MAX_LOGITS_TO_CHECK_OTHERS

        for _ in range(number_try):
            if best_logits in self._termonators:
                break
            if best_logits not in self._vocab.get_numebrs_index():
                best_logits = sorted_logits_index.pop(0)
            else:
                return None, best_logits

        number: str = self.decode(parameter_ids)
        if not number or not number.strip():
            self._logger.log('Best logits is empty, skipping...')
            return 0, best_logits

        return float(number), best_logits

    def _extract_string_parameter(
                self,
                sorted_logits_index: list[int],
                parameter_ids: list[int],
                prompt: str
            ) -> tuple[str | None, int]:
        """Extract a string parameter from candidate token predictions.

        The routine attempts a few top logits and validates the resulting
        decoded candidate against string terminators and whether it appears
        in the original prompt.

        Args:
            sorted_logits_index (list[int]): Candidate token ids sorted by
                descending logits.
            parameter_ids (list[int]): Token ids collected so far for this
                parameter value.
            prompt (str): Original user prompt for context-based checks.

        Returns:
            tuple[str | None, int]: Extracted string (or None if not found)
            and the token id which signaled termination or -1.
        """

        for i in range(self.MAX_LOGITS_TO_CHECK):
            predicted_token_id: int = sorted_logits_index[i]
            predicted_sentence: str = self.decode(
                parameter_ids + [predicted_token_id]
            )

            if (i < 1
               and self.decode([predicted_token_id]).strip()
               in self.STRING_TERMINATORS):
                break

            if not predicted_token_id or not predicted_sentence.strip():
                self._logger.log('Best logits is empty, skipping...')
                break

            if predicted_sentence in prompt:
                return None, predicted_token_id

        return str(
            self.decode(parameter_ids)
        ).strip().strip('"').strip("'").strip(), -1

    def _extract_boolean_parameter(
                self,
                logits: list[float],
            ) -> tuple[bool, int]:
        """Extract a boolean parameter by comparing logits.

        Chooses between the tokens for 'true' and 'false' based on which
        token has the higher logit score.

        Args:
            logits (list[float]): Logits returned by the model for the
                current input.

        Returns:
            tuple[bool, int]: Selected boolean value and the token id used.
        """

        true_ids: list[int] = self.encode('true')
        false_ids: list[int] = self.encode('false')

        if logits[true_ids[0]] > logits[false_ids[0]]:
            return True, true_ids[0]
        else:
            return False, false_ids[0]

    def _extract_single_parameter(
                self,
                parameter_name: str,
                parameter: Parameter,
                prompt: str,
                parameter_prompt_ids: list[int]
            ) -> list[int]:
        """Extract a single parameter value for a function parameter.

        This method loops until a valid value for the parameter is found,
        supporting types: number, string and boolean.

        Args:
            parameter_name (str): Name of the parameter being extracted.
            parameter (Parameter): Parameter model instance to populate.
            prompt (str): Original user prompt used for context.
            parameter_prompt_ids (list[int]): Encoded ids of the prompt used
                to query the model for this parameter value.

        Returns:
            list[int]: Updated prompt ids including the tokens predicted for
            the extracted parameter value.
        """

        parameter_ids: list[int] = []

        while True:
            logits = self._model.get_logits_from_input_ids(
                parameter_prompt_ids
            )
            sorted_logits_index: list[int] = np.argsort(logits).tolist()[::-1]

            best_logits: int
            match parameter.type:
                case 'number' | 'integer' | 'float':
                    num_value: float | None
                    num_value, best_logits = self._extract_number_parameter(
                        sorted_logits_index, parameter_ids
                    )
                    if num_value is not None:
                        parameter.value = num_value
                        self._logger.log(
                            f'{Color.BRIGHT_YELLOW}{Color.BOLD}Parameter'
                            f' {parameter_name}: '
                            f'\'{parameter.value}\'{Color.RESET}'
                        )
                        break

                case 'string':
                    str_value: str | None
                    str_value, best_logits = self._extract_string_parameter(
                        sorted_logits_index,
                        parameter_ids,
                        prompt
                    )
                    if str_value is not None:
                        parameter.value = str_value
                        self._logger.log(
                            f'{Color.BRIGHT_YELLOW}{Color.BOLD}Parameter'
                            f' {parameter_name}: '
                            f'\'{parameter.value}\'{Color.RESET}'
                        )
                        break

                case 'boolean':
                    bool_value: bool
                    bool_value, best_logits = self._extract_boolean_parameter(
                        logits
                    )
                    parameter.value = bool_value
                    self._logger.log(
                        f'{Color.BRIGHT_YELLOW}{Color.BOLD}Parameter'
                        f' {parameter_name}: '
                        f'\'{parameter.value}\'{Color.RESET}'
                    )
                    break

                case _:
                    self._logger.error(
                        f'Parameter type \'{parameter.type}\' not supported,'
                        ' skipping...'
                    )
                    return parameter_prompt_ids

            parameter_prompt_ids.append(best_logits)
            parameter_ids.append(best_logits)

        return parameter_prompt_ids

    def _extract_parameters(
                self,
                prompt: str,
                prompt_ids_2d: list[int],
                function: FunctionDefinition
            ) -> FunctionDefinition:
        """Extract all parameters for the selected function.

        Args:
            prompt (str): Original user prompt.
            prompt_ids_2d (list[int]): Encoded preprompt ids.
            function (FunctionDefinition): FunctionDefinition to populate.

        Returns:
            FunctionDefinition: The function object with parameter values set.
        """

        self.progress_bar.update(0, StepName.EXTRACTING_PARAMETERS)

        for parameter_name, parameter in function.parameters.items():
            self._logger.log(f'Extracting parameter: {parameter_name}')

            parameter_prompt: list[int] = self.encode(
                f'\nparameter {parameter_name} ({parameter.type}): \''
            )

            parameter_prompt_ids: list[int] = np.concatenate(
                (prompt_ids_2d, parameter_prompt)
            ).ravel().tolist()

            prompt_ids_2d = self._extract_single_parameter(
                parameter_name, parameter, prompt, parameter_prompt_ids
            )

        return function

    def _log_extracted_function(self, function: FunctionDefinition) -> None:
        """Log the extracted function name and parameters.

        Args:
            function (FunctionDefinition): The function whose extraction will
                be logged.
        """

        self._logger.log(
            f'{Color.GREEN}Function \'{function.name}\' extracted with '
            f'parameters:{Color.RESET}'
        )
        for parameter in function.parameters.values():
            self._logger.log(
                f' - {parameter.name} ({parameter.type}): {parameter.value}'
            )

    def prompt(self, prompt: str) -> OutputPrompt:
        """Process a user prompt and return the extracted OutputPrompt.

        This is the main entry-point for extracting a function call from a
        prompt. It encodes the prompt, finds the appropriate function,
        extracts parameters, logs results and saves them to the output file.

        Args:
            prompt (str): User prompt to process.

        Returns:
            OutputPrompt: The persisted prompt information including the
            selected function name and extracted parameters.
        """

        user_prompt_ids: list[int] = self.encode(prompt)
        prompt_ids_2d: list[int] = self.get_preprompt(prompt)

        function: FunctionDefinition = self._extract_function(
            prompt=prompt,
            user_prompt_ids=user_prompt_ids,
            prompt_ids_2d=prompt_ids_2d
        )

        self._extract_parameters(
            prompt=prompt,
            prompt_ids_2d=prompt_ids_2d,
            function=function
        )
        self._log_extracted_function(function)

        output_prompt = self._output_file.add_prompt(
            prompt=prompt,
            function=function
        )
        self._output_file.save()

        return output_prompt

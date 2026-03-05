from llm_sdk import Small_LLM_Model  # type: ignore
from .FunctionDefinitions import FunctionDefinitions, FunctionDefinition
from .utils import Logger, Color
from .OutputFile import OutputFile
from pydantic import BaseModel, Field, PrivateAttr
from typing import Any, ClassVar
import numpy as np
import re

t_prompt_dict = dict[str, str | dict[str, str | float]]


class CallMeMaybe(BaseModel):
    functions_definition_path: str = Field(
        ...,
        description='The path to the _functions definition file'
    )
    output_file_path: str = Field(
        ...,
        description='The path to the output file where the prompts and their '
                    'extracted _functions will be saved'
    )

    NUMBER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?:\d+(?:\.\d*)?|\.\d+|\.)$"
    )

    _logger: Logger = PrivateAttr()
    _model: Small_LLM_Model = PrivateAttr()
    _output_file: OutputFile = PrivateAttr()
    _functions: FunctionDefinitions = PrivateAttr()

    def model_post_init(self, _: Any) -> None:
        self._logger: Logger = Logger(name='Core', color=Color.CYAN)

        self._logger.log('Initializing LLM...')
        self._model: Small_LLM_Model = Small_LLM_Model()
        self._logger.log('Model initialized.')

        self._output_file: OutputFile = OutputFile(
            file_path=self.output_file_path
        )

        self._functions = FunctionDefinitions(
            model=self._model,
            file_path=self.functions_definition_path
        )

        for key, value in self._functions.get_names_inputs().items():
            self._logger.log(
                f'Function \'{key}\' encoded with input ids: {value}'
            )

    def create_preprompt(self, user_prompt: str) -> str:
        preprompt: str = 'Extract the function for the user query\n'
        functuions_list: str = ', '.join(self._functions.get_names())

        preprompt += f'_functions available: {functuions_list}\n'
        preprompt += f'query: {user_prompt}\n'
        preprompt += 'function to use: '

        return preprompt

    def get_preprompt(self, user_prompt: str) -> list[int]:
        prompt: str = self.create_preprompt(user_prompt)

        self._logger.log(f'Preprompt:\n{prompt}')

        return self.encode(prompt)

    def encode(self, text: str) -> list[int]:
        input_ids_2d: list[list[int]] = self._model.encode(text).tolist()
        return np.concatenate(input_ids_2d).ravel().tolist()

    def decode(self, input_ids: list[int]) -> str:
        return str(self._model.decode(input_ids))

    def prompt(self, prompt: str) -> None:
        user_prompt_ids: list[int] = self.encode(prompt)
        prompt_ids_2d: list[int] = self.get_preprompt(prompt)
        function_name_ids: list[int] = []
        self._logger.log(
            f'{Color.GREEN}User prompt ids: {user_prompt_ids}'
            f'{Color.RESET}'
        )
        longes_function_name: list[int] = max(
            self._functions.get_names_inputs().values(),
            key=lambda x: len(x)
        )

        self._logger.log(
            'Finding function name with '
            f'{len(longes_function_name)} iterations...'
        )

        for _ in range(len(longes_function_name)):
            logits: list[float] = self._model.get_logits_from_input_ids(
                prompt_ids_2d
            )

            available_functions_inputs: list[int] = [
                inputs[len(function_name_ids)]
                for _, inputs
                in self._functions.get_names_inputs_with(
                    function_name_ids
                ).items()
                if len(inputs) > len(function_name_ids)
            ]
            self._logger.log(
                f'Available _functions inputs: {available_functions_inputs}'
            )

            availables_functions_logits: list[tuple[int, float]] = [
                (index, logits[index])
                for index in available_functions_inputs
            ]

            sorted_functions_logits: list[tuple[int, float]] = sorted(
                availables_functions_logits,
                key=lambda x: x[1],
                reverse=True
            )
            self._logger.log(
                f'Available _functions logits: {sorted_functions_logits}'
            )

            if not len(sorted_functions_logits):
                self._logger.log('No available _functions logits.')
                break

            best_logits: int
            best_logits, _ = sorted_functions_logits.pop(0)
            self._logger.log(
                f'Best logits: {best_logits} - value'
                f': \'{self.decode([best_logits])}\''
            )
            function_name_ids.append(best_logits)
            prompt_ids_2d.append(best_logits)

        function_name: str = self.decode(function_name_ids)
        self._logger.log(
            f'{Color.BRIGHT_BLUE}{Color.BOLD}Function name found'
            f': \'{function_name}\'{Color.RESET}'
        )
        self._logger.log(f'Full prompt: \'{self.decode(prompt_ids_2d)}\'')

        function: FunctionDefinition = self._functions.get_by_name(
            function_name
        )

        for index in user_prompt_ids:
            self._logger.log(
                f'User prompt ids - index: {index} - value: '
                f'\'{self.decode([index])}\''
            )

        for parameter_name, parameter in function.parameters.items():
            self._logger.log(f'Extracting parameter: {parameter_name}')

            parameter_prompt: list[int] = self.encode(
                f'\nparameter {parameter_name} ({parameter.type}): \''
            )

            parameter_prompt_ids: list[int] = np.concatenate(
                (prompt_ids_2d, parameter_prompt)
            ).ravel().tolist()
            self._logger.log(
                f'Prompt: \''
                f'{self.decode(parameter_prompt_ids)}\''
            )
            parameter_ids: list[int] = []

            parameter_complete: bool = False
            while not parameter_complete:
                logits = self._model.get_logits_from_input_ids(
                    parameter_prompt_ids
                )

                sorted_logits_index: list[int] = np.argsort(
                    logits
                ).tolist()[::-1]  # descending order

                self._logger.log('---------------\n')
                self._logger.log('The 10 best logits are:')
                for i in range(10):
                    self._logger.log(
                        f' - index: {sorted_logits_index[i]} - value: '
                        f'\'{self.decode([sorted_logits_index[i]])}\''
                        f' - logit: {logits[sorted_logits_index[i]]:.2f}'
                    )

                match parameter.type:
                    case 'number':
                        best_logits = sorted_logits_index.pop(0)
                        if (not re.search(
                                    self.NUMBER_PATTERN,
                                    self.decode([best_logits])
                                )):
                            self._logger.log(
                                f'Best logits \''
                                f'{self.decode([best_logits])}\''
                                ' is not a number, skipping...'
                            )

                            parameter.value = float(
                                self.decode(parameter_ids)
                            )
                            self._logger.log(
                                f'{Color.BRIGHT_YELLOW}{Color.BOLD}Parameter'
                                f' {parameter_name}: \''
                                f'{parameter.value}\'{Color.RESET}'
                            )
                            break
                    case 'string':
                        best_logits = -1
                        for i in range(10):
                            predicted_token_id: int = sorted_logits_index[i]
                            predicted_sentence: str = self.decode(
                                parameter_ids + [predicted_token_id]
                            )
                            if (i < 1
                               and self.decode([predicted_token_id]).strip()
                               in [
                                        '"', "\\'", "'", '\"',
                                        '\n', '"\n', "'\n"
                                    ]):
                                self._logger.log(
                                    f'{Color.RED}Best logits '
                                    f'{self.decode([predicted_token_id])} '
                                    f'not valid, skipping...{Color.RESET}'
                                )
                                break

                            self._logger.log(
                                f'Checking best logits: {predicted_token_id}'
                                ' - value: \''
                                f'{self.decode([predicted_token_id])}\' '
                                f'- logit: {logits[predicted_token_id]:.2f} - '
                                f'predicted sentence: \'{predicted_sentence}\''
                            )
                            if (not predicted_token_id or
                               not predicted_sentence.strip()):
                                self._logger.log(
                                    'Best logits is empty, skipping...'
                                )
                                break
                            if predicted_sentence in prompt:
                                best_logits = predicted_token_id
                                break

                        if (best_logits == -1):
                            parameter.value = str(
                                self.decode(parameter_ids)
                            ).strip().strip('"').strip("'").strip()
                            self._logger.log(
                                f'{Color.BRIGHT_YELLOW}{Color.BOLD}Parameter'
                                f' {parameter_name}: \''
                                f'{parameter.value}\'{Color.RESET}'
                            )
                            break
                    case _:
                        self._logger.log(
                            f'Parameter type \'{parameter.type}\''
                            ' not supported, skipping...'
                        )
                        break

                self._logger.log(
                    f'Best logits: {best_logits} - value'
                    f': \'{self.decode([best_logits])}\''
                )
                parameter_prompt_ids.append(best_logits)
                parameter_ids.append(best_logits)

            prompt_ids_2d = parameter_prompt_ids

        self._logger.log(
            f'{Color.GREEN}Function \'{function_name}\' extracted'
            f' with parameters:{Color.RESET}'
        )
        for parameter in function.parameters.values():
            self._logger.log(
                f' - {parameter.name} ({parameter.type}): {parameter.value}'
            )

        self._output_file.add_prompt(
            prompt=prompt,
            function=function
        )
        self._output_file.save()

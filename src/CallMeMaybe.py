from llm_sdk import Small_LLM_Model  # type: ignore
from .FunctionDefinitions import FunctionDefinitions, FunctionDefinition
from .utils import Logger, Color
from .OutputFile import OutputFile
import numpy as np
import re

t_prompt_dict = dict[str, str | dict[str, str | float]]


class CallMeMaybe():
    NUMBER_PATTERN = r"^(?:\d+(?:\.\d*)?|\.\d+|\.)$"

    def __init__(
                self,
                functions_definition_path: str,
                output_file_path: str
            ) -> None:
        self.logger: Logger = Logger(name='CallMeMaybe', color=Color.CYAN)
        self.logger.log('Initializing LLM...')
        self.model: Small_LLM_Model = Small_LLM_Model()
        self.logger.log('Model initialized.')
        self.output_file: OutputFile = OutputFile(
            logger=Logger(name='OutputFile', color=Color.BRIGHT_WHITE),
            file_path=output_file_path
        )

        self.functions = FunctionDefinitions(
            model=self.model,
            file_path=functions_definition_path
        )

        for key, value in self.functions.get_names_inputs().items():
            self.logger.log(
                f'Function \'{key}\' encoded with input ids: {value}'
            )

    def create_preprompt(self, user_prompt: str) -> str:
        preprompt: str = 'Extract the function for the user query\n'
        functuions_list: str = ', '.join(
            [function.name for function in self.functions.functions_definition]
        )
        preprompt += f'functions available: {functuions_list}\n'
        preprompt += f'query: {user_prompt}\n'
        preprompt += 'function to use: '

        return preprompt

    def get_preprompt(self, user_prompt: str) -> list[int]:
        prompt: str = self.create_preprompt(user_prompt)

        self.logger.log(f'Preprompt:\n{prompt}')

        return self.encode(prompt)

    def encode(self, text: str) -> list[int]:
        input_ids_2d: list[list[int]] = self.model.encode(text).tolist()
        return np.concatenate(input_ids_2d).ravel().tolist()

    def decode(self, input_ids: list[int]) -> str:
        return self.model.decode(input_ids)

    """
    Todo:
        - while parameters are not complete:
            - Create 'mid  prompt' to extract function parameters
            - Generate parameters with restraints e.g type: number use number
              regex to restrain the generation of the parameter value
    """
    def prompt(self, prompt: str) -> None:
        # prompt: str = prompt.replace('\'', '\\\'').replace('"', '\\"')
        user_prompt_ids: list[int] = self.encode(prompt)
        prompt_ids_2d: list[int] = self.get_preprompt(prompt)
        function_name_ids: list[int] = []
        self.logger.log(
            f'{Color.GREEN}User prompt ids: {user_prompt_ids}'
            f'{Color.RESET}'
        )
        longes_function_name: list[int] = max(
            self.functions.get_names_inputs().values(),
            key=lambda x: len(x)
        )

        self.logger.log(
            'Finding function name with '
            f'{len(longes_function_name)} iterations...'
        )

        for _ in range(len(longes_function_name)):
            logits: list[float] = self.model.get_logits_from_input_ids(
                prompt_ids_2d
            )

            available_functions_inputs: list[int] = [
                inputs[len(function_name_ids)]
                for _, inputs
                in self.functions.get_names_inputs_with(
                    function_name_ids
                ).items()
                if len(inputs) > len(function_name_ids)
            ]
            self.logger.log(
                f'Available functions inputs: {available_functions_inputs}'
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
            self.logger.log(
                f'Available functions logits: {sorted_functions_logits}'
            )

            if not len(sorted_functions_logits):
                self.logger.log('No available functions logits.')
                break

            best_logits: int
            best_logits, _ = sorted_functions_logits.pop(0)
            self.logger.log(
                f'Best logits: {best_logits} - value'
                f': \'{self.model.decode([best_logits])}\''
            )
            function_name_ids.append(best_logits)
            prompt_ids_2d.append(best_logits)

        function_name: str = self.model.decode(function_name_ids)
        self.logger.log(f'{Color.BRIGHT_BLUE}{Color.BOLD}Function name found'
                        f': \'{function_name}\'{Color.RESET}')
        self.logger.log(f'Full prompt: \'{self.model.decode(prompt_ids_2d)}\'')

        function: FunctionDefinition = self.functions.get_by_name(
            function_name
        )

        # print user_prompt_ids seq and index
        for index in user_prompt_ids:
            self.logger.log(
                f'User prompt ids - index: {index} - value: '
                f'\'{self.model.decode([index])}\''
            )

        for parameter_name, parameter in function.parameters.items():
            self.logger.log(f'Extracting parameter: {parameter_name}')

            parameter_prompt: list[int] = self.encode(
                f'\nparameter {parameter_name} ({parameter.type}): \''
            )

            parameter_prompt_ids: list[int] = np.concatenate(
                (prompt_ids_2d, parameter_prompt)
            ).ravel().tolist()
            self.logger.log(f'Prompt: \''
                            f'{self.model.decode(parameter_prompt_ids)}\'')
            parameter_ids: list[int] = []

            parameter_complete: bool = False
            while not parameter_complete:
                logits = self.model.get_logits_from_input_ids(
                    parameter_prompt_ids
                )

                sorted_logits_index: list[int] = np.argsort(
                    logits
                ).tolist()[::-1]  # descending order

                self.logger.log('---------------\n')
                self.logger.log('The 10 best logits are:')
                for i in range(10):
                    self.logger.log(
                        f' - index: {sorted_logits_index[i]} - value: '
                        f'\'{self.model.decode([sorted_logits_index[i]])}\''
                        f' - logit: {logits[sorted_logits_index[i]]:.2f}'
                    )

                match parameter.type:
                    case 'number':
                        best_logits = sorted_logits_index.pop(0)
                        if (not re.search(
                                    self.NUMBER_PATTERN,
                                    self.decode([best_logits])
                                )):
                            self.logger.log(
                                f'Best logits \''
                                f'{self.model.decode([best_logits])}\''
                                ' is not a number, skipping...'
                            )

                            parameter.value = float(
                                self.model.decode(parameter_ids)
                            )
                            self.logger.log(
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
                                self.logger.log(
                                    f'{Color.RED}Best logits '
                                    f'{self.decode([predicted_token_id])} '
                                    f'not valid, skipping...{Color.RESET}'
                                )
                                break

                            self.logger.log(
                                f'Checking best logits: {predicted_token_id}'
                                ' - value: \''
                                f'{self.model.decode([predicted_token_id])}\' '
                                f'- logit: {logits[predicted_token_id]:.2f} - '
                                f'predicted sentence: \'{predicted_sentence}\''
                            )
                            if (not predicted_token_id or
                               not predicted_sentence.strip()):
                                self.logger.log(
                                    'Best logits is empty, skipping...'
                                )
                                break
                            if predicted_sentence in prompt:
                                best_logits = predicted_token_id
                                break

                        if (best_logits == -1):
                            # self.logger.log(
                            #     'Best logits \''
                            #     f'{self.model.decode([best_logits])}'
                            #     '\' already used in parameter, skipping...'
                            # )

                            parameter.value = str(
                                self.model.decode(parameter_ids)
                            ).strip().strip('"').strip("'").strip()
                            self.logger.log(
                                f'{Color.BRIGHT_YELLOW}{Color.BOLD}Parameter'
                                f' {parameter_name}: \''
                                f'{parameter.value}\'{Color.RESET}'
                            )
                            break
                    case _:
                        self.logger.log(
                            f'Parameter type \'{parameter.type}\''
                            ' not supported, skipping...'
                        )
                        break

                self.logger.log(
                    f'Best logits: {best_logits} - value'
                    f': \'{self.model.decode([best_logits])}\''
                )
                parameter_prompt_ids.append(best_logits)
                parameter_ids.append(best_logits)

            prompt_ids_2d = parameter_prompt_ids

        self.logger.log(f'{Color.GREEN}Function \'{function_name}\' extracted'
                        f' with parameters:{Color.RESET}')
        for parameter in function.parameters.values():
            self.logger.log(
                f' - {parameter.name} ({parameter.type}): {parameter.value}'
            )

        self.output_file.add_prompt(
            prompt=prompt,
            function=function
        )
        self.output_file.save()

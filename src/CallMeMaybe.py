from llm_sdk import Small_LLM_Model
from .FunctionDefinitions import FunctionDefinitions, FunctionDefinition
from .Vocabulary import Vocabulary

from .utils import Logger, Color, is_subsequence
import numpy as np



class CallMeMaybe():
    def __init__(self, functions_definition_path: str) -> None:
        self.logger: Logger = Logger('CallMeMaybe', Color.CYAN)
        self.logger.log('Initializing LLM...')
        self.model: Small_LLM_Model = Small_LLM_Model()
        self.logger.log('Model initialized.')

        self.vocab = Vocabulary(
            file_path=self.model.get_path_to_vocab_file()
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

    """
    Todo:
        - while parameters are not complete:
            - Create 'mid  prompt' to extract function parameters
            - Generate parameters with restraints e.g type: number use number
              regex to restrain the generation of the parameter value
    """
    def prompt(self, prompt: str) -> None:
        user_prompt_ids: list[int] = self.encode(prompt)
        prompt_ids_2d: list[int] = self.get_preprompt(prompt)
        function_name_ids: list[int] = []

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

            availables_functions_logits: list[tuple[int, float]] = {
                (index, logits[index])
                for index in available_functions_inputs
            }

            sorted_functions_logits: list[int] = sorted(
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

            best_logits, _ = sorted_functions_logits.pop(0)
            self.logger.log(
                f'Best logits: {best_logits} - value'
                f': \'{self.model.decode([best_logits])}\''
            )
            function_name_ids.append(best_logits)
            prompt_ids_2d.append(best_logits)

        function_name: str = self.model.decode(function_name_ids)
        self.logger.log(f'Function name found: \'{function_name}\'')
        self.logger.log(f'Full prompt: \'{self.model.decode(prompt_ids_2d)}\'')

        function_definition: FunctionDefinition = self.functions.get_by_name(
            function_name
        )

        for parameter in function_definition.parameters.values():
            self.logger.log(f'Extracting parameter: {parameter.name}')

            parameter_prompt: list[int] = self.encode(
                f'\nparameter {parameter.name} ({parameter.type}): '
            )

            parameter_prompt_ids: list[int] = np.concatenate(
                (prompt_ids_2d, parameter_prompt)
            ).ravel().tolist()
            self.logger.log(f'Prompt: \''
                            f'{self.model.decode(parameter_prompt_ids)}\'')
            parameter_ids: list[int] = []

            parameter_complete: bool = False
            while not parameter_complete:
                logits: list[float] = self.model.get_logits_from_input_ids(
                    parameter_prompt_ids
                )

                sorted_logits_index: list[int] = np.argsort(
                    logits
                ).tolist()[::-1]  # descending order

                # self.logger.log('The 10 best logits are:')
                # for i in range(10):
                #     self.logger.log(
                #         f' - index: {sorted_logits_index[i]} - value: '
                #         f'\'{self.model.decode([sorted_logits_index[i]])}\''
                #         f' - logit: {logits[sorted_logits_index[i]]:.2f}'
                #     )

                best_logits: int
                match parameter.type:
                    case 'number':
                        best_logits = sorted_logits_index.pop(0)
                        if (best_logits
                           not in self.vocab.get_numbers_ids().keys()):
                            self.logger.log(
                                f'Best logits \''
                                f'{self.model.decode([best_logits])}\''
                                ' is not a number, skipping...'
                            )
                            
                            # self.logger.log(
                            #     f'Parameter {parameter.name}: \''
                            #     f'{}\''
                            break
                    case 'string':
                        availables_logits_from_prompt: list[tuple[int, float]]
                        availables_logits_from_prompt = {
                            (index, logits[index])
                            for index in user_prompt_ids
                        }
                        if not len(availables_logits_from_prompt):
                            self.logger.log(
                                'No available logits from prompt.'
                            )
                            break
                        best_logits, _ = max(
                            availables_logits_from_prompt,
                            key=lambda x: x[1]
                        )

                        if not is_subsequence(
                                    user_prompt_ids,
                                    parameter_ids + [best_logits]
                                ):
                            self.logger.log(
                                'Best logits \''
                                f'{self.model.decode([best_logits])}'
                                '\' already used in parameter, skipping...'
                            )
                            break
                    case _:
                        self.logger.log(
                            f'Parameter type \'{parameter.type}\''
                            ' not supported, skipping...'
                        )
                        break

                # availables_numbers_logits: list[tuple[int, float]] = {
                #     (index, logits[index])
                #     for index in self.vocab.get_numbers_ids().keys()
                # }

                # self.vocab.get_numbers_ids().keys()

                # self.logger.log(availables_numbers_logits)

                # sorted_numbers_logits: list[int] = sorted(
                #     availables_numbers_logits,
                #     key=lambda x: x[1],
                #     reverse=True
                # )

                # max not in avaliable numbers return

                # best_logits, _ = sorted_numbers_logits.pop(0)
                self.logger.log(
                    f'Best logits: {best_logits} - value'
                    f': \'{self.model.decode([best_logits])}\''
                )
                parameter_prompt_ids.append(best_logits)
                parameter_ids.append(best_logits)

            prompt_ids_2d = parameter_prompt_ids

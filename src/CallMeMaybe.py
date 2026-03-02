from llm_sdk import Small_LLM_Model
from .FunctionDefinitions import FunctionDefinitions
from .Vocabulary import Vocabulary

from .utils import Logger, Color
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
            - Create 'mid prompt' to extract function parameters
            - Generate parameters with restraints e.g type: number use number
              regex to restrain the generation of the parameter value
    """
    def prompt(self, prompt: str) -> None:
        prompt_ids_2d: list[int] = self.get_preprompt(prompt)
        answer_ids_2d: list[int] = []

        longes_function_name: list[int] = max(
            self.functions.get_names_inputs().values(),
            key=lambda x: len(x)
        )

        self.logger.log(
            'Finding function with on '
            f'{len(longes_function_name)} iterations...'
        )
        for _ in range(len(longes_function_name)):
            logits: list[float] = self.model.get_logits_from_input_ids(
                prompt_ids_2d
            )

            available_functions_inputs: list[int] = [
                inputs[len(answer_ids_2d)]
                for _, inputs
                in self.functions.get_names_inputs_with(answer_ids_2d).items()
                if len(inputs) > len(answer_ids_2d)
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
            answer_ids_2d.append(best_logits)
            prompt_ids_2d.append(best_logits)

        function_name: str = self.model.decode(answer_ids_2d)
        self.logger.log(f'Function name found: \'{function_name}\'')

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

    def create_preprompt(self) -> str:
        # 'You are a helpful assistant that can call ' + \
        # 'functions to answer questions.\n' + \
        preprompt: str = 'You have at your dispositoions this functions:\n'

        for function in self.functions_definition:
            preprompt += '----\n'
            preprompt += f'Name: \'{function.get("name")}\'\n'
            preprompt += f'Description: \'{function.get("description")}\'\n'
            preprompt += 'Parameters:\n'
            for name, content in function.get('parameters').items():
                preprompt += f'    - \'{name}\': {content.get("type")}\n'

        # += 'Use only fonctions above to answer the question below.\n'
        # preprompt += ''
        return preprompt

    def get_preprompt(self) -> list[int]:
        if not hasattr(self, 'preprompt_input_ids_2d'):
            prompt: str = self.create_preprompt()
            self.preprompt_input_ids_2d: str = self.encode(prompt)
        return self.preprompt_input_ids_2d

    def encode(self, text: str) -> list[int]:
        input_ids_2d: list[list[int]] = self.model.encode(text).tolist()
        return np.concatenate(input_ids_2d).ravel().tolist()

    def prompt(self, prompt: str) -> None:
        """
        - add preprompt
        - filter les keys of vocabs
        """
        user_prompt_ids: list[int] = np.concatenate(
            self.model.encode(prompt).tolist()
        ).ravel().tolist()

        prompt_ids_2d: list[int] = np.concatenate(
            [self.get_preprompt(), user_prompt_ids]
        ).ravel().tolist()

        answer_ids_2d: list[int] = []

        print('Generating:')
        for _ in range(40):
            logits: list[float] = self.model.get_logits_from_input_ids(
                prompt_ids_2d
            )
            sorted_logits_index: list[int] = np.argsort(logits).tolist()

            best_logits = sorted_logits_index.pop()

            answer_ids_2d.append(best_logits)
            prompt_ids_2d.append(best_logits)
            print(self.model.decode([best_logits]), end='')

        response = self.model.decode(answer_ids_2d)
        print(f'Final Resonse: \'{response}\'')

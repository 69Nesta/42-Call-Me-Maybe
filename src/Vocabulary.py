from .JsonParser import JsonParder
from .utils import Color, Logger
import re


t_vocab = dict[str, int]


class Vocabulary:
    def __init__(self, file_path: str) -> None:
        self.logger = Logger('Vocabulary', Color.BLUE)
        self.logger.log('Loading vocab...')
        self.vocab_path: str = file_path
        self.vocab_parser: JsonParder = JsonParder(file_path)
        self.vocab: t_vocab = self.vocab_parser.get_data()
        self.logger.log('Vocab loaded.')

    def get_numbers_ids(self) -> dict[int, str]:
        if not hasattr(self, 'numebrs_ids'):
            pattern = r"^(?:\d+(?:\.\d*)?|\.\d+|\.)$"
            self.numbers_ids: dict[int, str] = {
                index: key
                for key, index in self.vocab.items()
                if re.search(pattern, key)
            }

        return self.numbers_ids

    def get(self) -> t_vocab:
        return self.vocab

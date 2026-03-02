from .JsonParser import JsonParder
from .utils import Color, Logger


t_vocab = dict[str, int]


class Vocabulary:
    def __init__(self, file_path: str) -> None:
        self.logger = Logger('Vocabulary', Color.BLUE)
        self.logger.log('Loading vocab...')
        self.vocab_path: str = file_path
        self.vocab_parser: JsonParder = JsonParder(file_path)
        self.vocab: t_vocab = self.vocab_parser.get_data()
        self.logger.log('Vocab loaded.')

    def get(self) -> t_vocab:
        return self.vocab

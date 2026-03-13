from pydantic import BaseModel, Field, PrivateAttr
from typing import ClassVar, Any
from .JsonParser import JsonParser
from .utils import Logger, Color
import re


class Vocabulary(BaseModel):
    """Represents a simple token vocabulary focused on numeric tokens.

    Attributes:
        file_path (str): Path to the vocab JSON file.
        verbose (bool): Enable verbose logging.
    """
    file_path: str = Field(..., description='The path to the vocab file')
    verbose: bool = Field(default=False)

    NUMBER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?:\d+(?:\.\d*)?|\.\d+|\.)$"
    )

    _numbers: dict[int, str] = PrivateAttr({})
    _logger: Logger = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        self._logger = Logger(
            ACTIVE=self.verbose,
            name='Vocab',
            color=Color.BLUE
        )

        self.parse()

        return super().model_post_init(context)

    def parse(self) -> None:
        """Load the vocabulary file and build a numeric token map.

        The vocab file is expected to be a mapping from token string to token
        index (or vice versa) and numeric-looking keys will be stored in
        an internal map for quick lookup.
        """
        self._logger.log(f'Initializing vocab from file: {self.file_path}')
        content: Any = JsonParser(file_path=self.file_path).get_data()

        if not isinstance(content, dict):
            raise ValueError(
                f'Expected a dictionary in the vocab file, got {type(content)}'
            )

        for key, value in content.items():
            if self.NUMBER_PATTERN.match(key):
                self._numbers.update({
                    int(value): key
                })

        self._logger.log('Vocab loaded !')
        self._logger.log(f'Numbers found in vocab: {self.get_numbers()}')

    def get_numbers(self) -> dict[int, str]:
        """Return the mapping of numeric token ids to their string form.

        Returns:
            dict[int, str]: Mapping of token id to string representation.
        """
        return self._numbers

    def get_numebrs_index(self) -> list[int]:
        """Return the list of token ids that represent numbers.

        Returns:
            list[int]: Token ids recognized as numeric tokens.
        """
        return list(self._numbers.keys())

    def is_number(self, token_id: int) -> bool:
        """Return whether a token id represents a numeric token.

        Args:
            token_id (int): Token id to check.

        Returns:
            bool: True if token_id corresponds to a numeric token.
        """
        return token_id in self._numbers

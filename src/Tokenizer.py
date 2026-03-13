from pydantic import BaseModel, Field, PrivateAttr
from .Vocabulary import Vocabulary
from .utils import Logger, Color
from typing import Any
# import unicodedata


class Tokenizer(BaseModel):
    vocabulary: Vocabulary = Field(
        ...,
        description="The vocabulary to use for tokenization."
    )
    verbose: bool = Field(
        default=False,
        description="Whether to print verbose logs during tokenization."
    )

    _logger: Logger = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        self._logger = Logger(
            name="Tokenizer",
            color=Color.CYAN,
            ACTIVE=self.verbose
        )

        return super().model_post_init(context)

    def encode(self, text: str) -> list[int]:
        encoded: list[int] = []

        return encoded

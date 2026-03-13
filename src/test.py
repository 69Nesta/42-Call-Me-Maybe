from .Tokenizer import Tokenizer
from .Vocabulary import Vocabulary


def test_tokenizer() -> None:
    vocab: Vocabulary = Vocabulary(
        file_path='~/goinfre/.llm_cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/vocab.json',
        verbose=True
    )

    tokenizer: Tokenizer = Tokenizer(
        vocabulary=vocab,
        verbose=True
    )

    tokenizer.encode("Hello, world!")

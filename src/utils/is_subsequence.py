from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def is_subsequence(main: Iterable[T], sub: Iterable[T]) -> bool:
    it: Iterator[T] = iter(main)
    return all(x in it for x in sub)

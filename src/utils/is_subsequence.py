from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def is_subsequence(main: Iterable[T], sub: Iterable[T]) -> bool:
    """Return True if ``sub`` is a subsequence of ``main``.

    Both ``main`` and ``sub`` are any iterable of comparable items. The
    local variable ``it`` is explicitly typed as ``Iterator[T]`` so static
    type checkers know the iterator element type.
    """
    it: Iterator[T] = iter(main)
    return all(x in it for x in sub)

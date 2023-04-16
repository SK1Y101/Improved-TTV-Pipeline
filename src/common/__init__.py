""" Common utility between all submodules. """
from __future__ import annotations

from logging import Logger
from typing import Any, Iterator, List, Type


def log_children(logger: Logger, root: Type, child_attr: str) -> None:
    def _logtree_(root: Type, prefix: List[str] = [], at_end: bool = True) -> None:
        prefix_str = " ".join(prefix + ["└" if at_end else "├", ""])[2:]
        logger.info(f"{prefix_str}{root}")
        children = getattr(root, child_attr, [])
        for idx, child in enumerate(children):
            _logtree_(
                child, prefix + [" " if at_end else "│"], idx >= len(children) - 1
            )

    _logtree_(root)


def flatten(nested: list[Any]) -> Iterator[Any]:
    for elem in nested:
        if isinstance(elem, list):
            for ele in flatten(elem):
                yield ele
        else:
            yield elem


def expand(cls: Any, attr: str) -> List[Any]:
    if hasattr(cls, attr):
        return [cls] + [expand(obj, attr) for obj in getattr(cls, attr)]
    return [cls]

from __future__ import annotations

from typing import Callable


class Spec:
    def __init__(self, fn: Callable) -> None:
        self._fn = fn

    def __call__(self, state) -> bool:
        return bool(self._fn(state))

    def __and__(self, other: Spec) -> Spec:
        return Spec(lambda s: self(s) and other(s))

    def __or__(self, other: Spec) -> Spec:
        return Spec(lambda s: self(s) or other(s))

    def __invert__(self) -> Spec:
        return Spec(lambda s: not self(s))

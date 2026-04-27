from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    name: str | None = None


@dataclass
class ActionNode(Node):
    func: Callable[..., Any] | None = None
    rollback: Callable[..., Any] | None = None


@dataclass
class SequentialNode(Node):
    children: list[Node] = field(default_factory=list)


@dataclass
class ConcurrentNode(Node):
    children: list[Node] = field(default_factory=list)


@dataclass
class OptionalNode(Node):
    guard: Callable[..., Any] | None = None
    then_: Node | None = None
    else_: Node | None = None


@dataclass
class LoopNode(Node):
    body: Node | None = None
    times: int | None = None
    until: Callable[..., Any] | None = None


@dataclass
class SwitchNode(Node):
    key: Callable[..., Any] | None = None
    cases: dict[str, Node] = field(default_factory=dict)

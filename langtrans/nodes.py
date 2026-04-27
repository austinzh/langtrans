from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Node:
    name: Optional[str] = None


@dataclass
class ActionNode(Node):
    func: Callable = None
    rollback: Optional[Callable] = None


@dataclass
class SequentialNode(Node):
    children: list[Node] = None


@dataclass
class ConcurrentNode(Node):
    children: list[Node] = None


@dataclass
class OptionalNode(Node):
    guard: Callable = None
    then_: Node = None
    else_: Optional[Node] = None


@dataclass
class LoopNode(Node):
    body: Node = None
    times: Optional[int] = None
    until: Optional[Callable] = None


@dataclass
class SwitchNode(Node):
    key: Callable = None
    cases: dict[str, Node] = None

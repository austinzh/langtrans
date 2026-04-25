from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union


@dataclass
class ActionNode:
    func: Callable
    rollback: Optional[Callable] = None
    name: Optional[str] = None


@dataclass
class SequentialNode:
    children: list[Node]


@dataclass
class ConcurrentNode:
    children: list[Node]


@dataclass
class OptionalNode:
    guard: Callable
    then_: Node
    else_: Optional[Node] = None


@dataclass
class LoopNode:
    body: Node
    times: Optional[int] = None
    until: Optional[Callable] = None


@dataclass
class RetryNode:
    body: Node
    max_attempts: int = 3
    delay: float = 0.0


@dataclass
class ProcedureNode:
    name: str
    body: Node


Node = Union[
    ActionNode, SequentialNode, ConcurrentNode, OptionalNode,
    LoopNode, RetryNode, ProcedureNode,
]

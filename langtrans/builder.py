from __future__ import annotations

from typing import Callable, Optional, Union

from langtrans.nodes import (
    ActionNode, ConcurrentNode, LoopNode, Node, OptionalNode,
    ProcedureNode, RetryNode, SequentialNode,
)
from langtrans.spec import Spec


def action(fn: Optional[Callable] = None, *, rollback: Optional[Callable] = None):
    def decorator(f: Callable) -> Callable:
        f._langtrans_rollback = rollback
        return f
    if fn is not None:
        return decorator(fn)
    return decorator


def _to_node(arg) -> Node:
    if isinstance(arg, Trans):
        return arg.build()
    if isinstance(arg, (ActionNode, SequentialNode, ConcurrentNode,
                        OptionalNode, LoopNode, RetryNode, ProcedureNode)):
        return arg
    if callable(arg):
        rollback = getattr(arg, "_langtrans_rollback", None)
        name = getattr(arg, "__name__", None)
        return ActionNode(func=arg, rollback=rollback, name=name)
    raise TypeError(f"Cannot convert {type(arg)} to a Node")


class Trans:
    def __init__(self, *, state_schema=None):
        self._state_schema = state_schema
        self._steps: list[Node] = []

    def sequential(self, *args: Union[Callable, Trans, Node]) -> Trans:
        children = [_to_node(a) for a in args]
        self._steps.append(SequentialNode(children=children))
        return self

    def concurrent(self, *args: Union[Callable, Trans, Node]) -> Trans:
        children = [_to_node(a) for a in args]
        self._steps.append(ConcurrentNode(children=children))
        return self

    def optional(self, guard: Union[Callable, Spec], *, then_: Union[Callable, Trans, Node],
                 else_: Optional[Union[Callable, Trans, Node]] = None) -> Trans:
        then_node = _to_node(then_)
        else_node = _to_node(else_) if else_ is not None else None
        self._steps.append(OptionalNode(guard=guard, then_=then_node, else_=else_node))
        return self

    def loop(self, *, body: Union[Callable, Trans, Node], times: Optional[int] = None,
             until: Optional[Callable] = None) -> Trans:
        self._steps.append(LoopNode(body=_to_node(body), times=times, until=until))
        return self

    def retry(self, target: Union[Callable, Trans, Node], *, max_attempts: int = 3,
              delay: float = 0.0) -> Trans:
        self._steps.append(RetryNode(body=_to_node(target), max_attempts=max_attempts, delay=delay))
        return self

    def procedure(self, name: str, body: Union[Trans, Node]) -> Trans:
        self._steps.append(ProcedureNode(name=name, body=_to_node(body)))
        return self

    def build(self) -> Node:
        if len(self._steps) == 1:
            return self._steps[0]
        return SequentialNode(children=list(self._steps))

    def compile(self, **kwargs):
        from langtrans.compiler import compile_graph
        tree = self.build()
        return compile_graph(tree, state_schema=self._state_schema, **kwargs)

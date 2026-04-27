from __future__ import annotations

from typing import Any, Callable, Protocol, Self, TypeAlias, runtime_checkable

from langtrans.nodes import (
    ActionNode,
    ConcurrentNode,
    LoopNode,
    Node,
    OptionalNode,
    SequentialNode,
    SwitchNode,
)
from langtrans.spec import Spec

__all__ = ["Trans", "Proc", "action", "StepArg", "GuardArg", "RunnableLike"]


@runtime_checkable
class RunnableLike(Protocol):
    """A graph step with ``invoke`` (e.g. LangGraph ``Runnable``) that is not used as a plain ``callable``."""

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        ...


class _BuilderBase:
    def __init__(self) -> None:
        self._steps: list[Node] = []

    def sequential(self, *args: StepArg, name: str | None = None) -> Self:
        children = [_to_node(a) for a in args]
        self._steps.append(SequentialNode(children=children, name=name))
        return self

    def concurrent(self, *args: StepArg, name: str | None = None) -> Self:
        children = [_to_node(a) for a in args]
        self._steps.append(ConcurrentNode(children=children, name=name))
        return self

    def optional(
        self,
        guard: GuardArg,
        *,
        then_: StepArg,
        else_: StepArg | None = None,
        name: str | None = None,
    ) -> Self:
        then_node = _to_node(then_)
        else_node = _to_node(else_) if else_ is not None else None
        self._steps.append(
            OptionalNode(guard=guard, then_=then_node, else_=else_node, name=name)
        )
        return self

    def loop(
        self,
        *,
        body: StepArg,
        times: int | None = None,
        until: Callable[..., Any] | None = None,
        name: str | None = None,
    ) -> Self:
        self._steps.append(
            LoopNode(body=_to_node(body), times=times, until=until, name=name)
        )
        return self

    def switch(
        self,
        *,
        key: Callable[..., Any],
        cases: dict[str, StepArg],
        name: str | None = None,
    ) -> Self:
        compiled_cases = {k: _to_node(v) for k, v in cases.items()}
        self._steps.append(SwitchNode(key=key, cases=compiled_cases, name=name))
        return self

    def _build_steps(self) -> Node:
        if len(self._steps) == 1:
            return self._steps[0]
        return SequentialNode(children=list(self._steps))


# Values accepted at builder composition time (``_to_node``) — plain functions, sub-builders,
# compiled ``Node`` trees, or LangGraph-style runnables (``invoke`` when not a ``callable``).
StepArg: TypeAlias = Node | _BuilderBase | Callable[..., Any] | RunnableLike

# ``optional`` guard: a ``Spec`` combinator or any callable the compiler accepts.
GuardArg: TypeAlias = Spec | Callable[..., Any]


def action(
    fn: Callable[..., Any] | None = None,
    *,
    rollback: Callable[..., Any] | None = None,
) -> Any:
    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        f._langtrans_rollback = rollback
        return f

    if fn is not None:
        return decorator(fn)
    return decorator


def _to_node(arg: StepArg) -> Node:
    if isinstance(arg, _BuilderBase):
        return arg.build()
    if isinstance(arg, Node):
        return arg
    if callable(arg):
        rollback = getattr(arg, "_langtrans_rollback", None)
        name = getattr(arg, "__name__", None)
        return ActionNode(func=arg, rollback=rollback, name=name)
    if hasattr(arg, "invoke"):
        name = getattr(arg, "name", None) or type(arg).__name__
        return ActionNode(func=arg, name=name)
    raise TypeError(f"Cannot convert {type(arg)} to a Node")


class Trans(_BuilderBase):
    def __init__(self, *, state_schema: Any = None) -> None:
        super().__init__()
        self._state_schema = state_schema

    def build(self) -> Node:
        return self._build_steps()

    def compile(self, **kwargs: Any) -> Any:
        from langtrans.compiler import compile_graph

        tree = self.build()
        return compile_graph(tree, state_schema=self._state_schema, **kwargs)


class Proc(_BuilderBase):
    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self._name = name

    def build(self) -> Node:
        inner = self._build_steps()
        if self._name is not None:
            inner.name = self._name
        return inner

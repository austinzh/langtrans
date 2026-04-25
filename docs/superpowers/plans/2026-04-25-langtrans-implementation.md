# langtrans Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python library that provides a fluent DSL for composing AI agent workflows, compiling them to LangGraph `CompiledStateGraph` objects.

**Architecture:** Three layers — DSL (decorators + builder), Node Model (dataclass tree), Compiler (tree-walk emitting StateGraph). Each primitive (Sequential, Concurrent, Optional, Loop, Retry, Procedure) maps to specific LangGraph node/edge patterns.

**Tech Stack:** Python 3.11+, langgraph, pytest, pyproject.toml (PEP 621)

---

## File Structure

```
langtrans/
    __init__.py          # public API: Trans, action, Spec, node types
    nodes.py             # 7 dataclass node types + Node union type
    builder.py           # Trans fluent builder + @action decorator
    compiler.py          # tree-walk compiler → StateGraph
    spec.py              # Spec guard with &, |, ~ combinators
tests/
    conftest.py          # shared fixtures (state schema, simple actions)
    test_nodes.py        # node dataclass construction
    test_spec.py         # Spec combinators
    test_builder.py      # Trans builder produces correct node trees
    test_compiler.py     # compiler emits correct StateGraph structure
    test_integration.py  # full compile + invoke flows
examples/
    sequential_agent.py
    parallel_tools.py
    react_agent.py
    rollback_pipeline.py
pyproject.toml           # package metadata + dependencies
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `langtrans/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "langtrans"
version = "0.1.0"
description = "A high-level DSL that compiles to LangGraph graphs"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]
```

- [ ] **Step 2: Create empty langtrans package**

`langtrans/__init__.py`:
```python
"""langtrans — A high-level DSL that compiles to LangGraph graphs."""
```

- [ ] **Step 3: Create test conftest with shared fixtures**

`tests/conftest.py`:
```python
import operator
from typing import Annotated, TypedDict

import pytest


class SimpleState(TypedDict):
    messages: Annotated[list, operator.add]
    metadata: dict


def make_action(name: str, key: str = "result", value: str = "ok"):
    """Create a simple action function that writes to metadata."""
    def fn(state):
        meta = dict(state.get("metadata", {}))
        lt = dict(meta.get("_langtrans", {}))
        meta["_langtrans"] = lt
        meta[key] = value
        calls = meta.get("_calls", [])
        calls = calls + [name]
        meta["_calls"] = calls
        return {"metadata": meta}
    fn.__name__ = name
    fn.__qualname__ = name
    return fn


@pytest.fixture
def simple_state_schema():
    return SimpleState


@pytest.fixture
def action_a():
    return make_action("action_a", "a", "done_a")


@pytest.fixture
def action_b():
    return make_action("action_b", "b", "done_b")


@pytest.fixture
def action_c():
    return make_action("action_c", "c", "done_c")
```

- [ ] **Step 4: Install in dev mode and verify**

Run: `cd /Users/austinzh/codes/transationDSL && pip install -e ".[dev]"`
Expected: Successful install

- [ ] **Step 5: Run pytest to verify empty test suite**

Run: `pytest tests/ -v`
Expected: "no tests ran" or similar — no errors

- [ ] **Step 6: Initialize git and commit**

```bash
git init
git add pyproject.toml langtrans/__init__.py tests/conftest.py
git commit -m "chore: scaffold langtrans project with pyproject.toml and test fixtures"
```

---

### Task 2: Node Model

**Files:**
- Create: `langtrans/nodes.py`
- Create: `tests/test_nodes.py`

- [ ] **Step 1: Write failing tests for node dataclasses**

`tests/test_nodes.py`:
```python
from langtrans.nodes import (
    ActionNode,
    ConcurrentNode,
    LoopNode,
    Node,
    OptionalNode,
    ProcedureNode,
    RetryNode,
    SequentialNode,
)


def dummy(state):
    return state


def dummy_rollback(state):
    return state


def dummy_guard(state) -> bool:
    return True


class TestActionNode:
    def test_minimal(self):
        node = ActionNode(func=dummy)
        assert node.func is dummy
        assert node.rollback is None
        assert node.name is None

    def test_with_rollback_and_name(self):
        node = ActionNode(func=dummy, rollback=dummy_rollback, name="my_action")
        assert node.rollback is dummy_rollback
        assert node.name == "my_action"


class TestSequentialNode:
    def test_children(self):
        a = ActionNode(func=dummy)
        b = ActionNode(func=dummy)
        node = SequentialNode(children=[a, b])
        assert len(node.children) == 2
        assert node.children[0] is a


class TestConcurrentNode:
    def test_children(self):
        a = ActionNode(func=dummy)
        b = ActionNode(func=dummy)
        node = ConcurrentNode(children=[a, b])
        assert len(node.children) == 2


class TestOptionalNode:
    def test_with_both_branches(self):
        a = ActionNode(func=dummy)
        b = ActionNode(func=dummy)
        node = OptionalNode(guard=dummy_guard, then_=a, else_=b)
        assert node.guard is dummy_guard
        assert node.then_ is a
        assert node.else_ is b

    def test_without_else(self):
        a = ActionNode(func=dummy)
        node = OptionalNode(guard=dummy_guard, then_=a)
        assert node.else_ is None


class TestLoopNode:
    def test_fixed_count(self):
        body = ActionNode(func=dummy)
        node = LoopNode(body=body, times=3)
        assert node.times == 3
        assert node.until is None

    def test_condition_based(self):
        body = ActionNode(func=dummy)
        node = LoopNode(body=body, until=dummy_guard)
        assert node.until is dummy_guard
        assert node.times is None


class TestRetryNode:
    def test_defaults(self):
        body = ActionNode(func=dummy)
        node = RetryNode(body=body)
        assert node.max_attempts == 3
        assert node.delay == 0.0

    def test_custom(self):
        body = ActionNode(func=dummy)
        node = RetryNode(body=body, max_attempts=5, delay=1.5)
        assert node.max_attempts == 5
        assert node.delay == 1.5


class TestProcedureNode:
    def test_named_sub_transaction(self):
        body = SequentialNode(children=[ActionNode(func=dummy)])
        node = ProcedureNode(name="sub_flow", body=body)
        assert node.name == "sub_flow"
        assert isinstance(node.body, SequentialNode)


class TestNodeUnionType:
    def test_action_is_node(self):
        node: Node = ActionNode(func=dummy)
        assert isinstance(node, ActionNode)

    def test_sequential_is_node(self):
        node: Node = SequentialNode(children=[])
        assert isinstance(node, SequentialNode)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nodes.py -v`
Expected: ImportError — `langtrans.nodes` does not exist yet

- [ ] **Step 3: Implement node dataclasses**

`langtrans/nodes.py`:
```python
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
    ActionNode,
    SequentialNode,
    ConcurrentNode,
    OptionalNode,
    LoopNode,
    RetryNode,
    ProcedureNode,
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_nodes.py -v`
Expected: All 11 tests pass

- [ ] **Step 5: Commit**

```bash
git add langtrans/nodes.py tests/test_nodes.py
git commit -m "feat: add node model dataclasses for all 7 DSL primitives"
```

---

### Task 3: Spec Guard Combinators

**Files:**
- Create: `langtrans/spec.py`
- Create: `tests/test_spec.py`

- [ ] **Step 1: Write failing tests for Spec**

`tests/test_spec.py`:
```python
from langtrans.spec import Spec


def test_spec_from_function():
    s = Spec(lambda state: state["metadata"].get("x") == 1)
    assert s({"metadata": {"x": 1}}) is True
    assert s({"metadata": {"x": 2}}) is False


def test_spec_and():
    a = Spec(lambda s: s["metadata"].get("x", 0) > 0)
    b = Spec(lambda s: s["metadata"].get("y", 0) > 0)
    combined = a & b
    assert combined({"metadata": {"x": 1, "y": 1}}) is True
    assert combined({"metadata": {"x": 1, "y": 0}}) is False
    assert combined({"metadata": {"x": 0, "y": 1}}) is False


def test_spec_or():
    a = Spec(lambda s: s["metadata"].get("x", 0) > 0)
    b = Spec(lambda s: s["metadata"].get("y", 0) > 0)
    combined = a | b
    assert combined({"metadata": {"x": 1, "y": 0}}) is True
    assert combined({"metadata": {"x": 0, "y": 1}}) is True
    assert combined({"metadata": {"x": 0, "y": 0}}) is False


def test_spec_not():
    a = Spec(lambda s: s["metadata"].get("x", 0) > 0)
    inverted = ~a
    assert inverted({"metadata": {"x": 0}}) is True
    assert inverted({"metadata": {"x": 1}}) is False


def test_spec_complex_composition():
    x_pos = Spec(lambda s: s["metadata"].get("x", 0) > 0)
    y_pos = Spec(lambda s: s["metadata"].get("y", 0) > 0)
    z_neg = Spec(lambda s: s["metadata"].get("z", 0) < 0)
    combined = (x_pos & y_pos) | ~z_neg
    assert combined({"metadata": {"x": 1, "y": 1, "z": 5}}) is True
    assert combined({"metadata": {"x": 0, "y": 0, "z": 5}}) is True
    assert combined({"metadata": {"x": 0, "y": 0, "z": -1}}) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_spec.py -v`
Expected: ImportError

- [ ] **Step 3: Implement Spec**

`langtrans/spec.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_spec.py -v`
Expected: All 5 tests pass

- [ ] **Step 5: Commit**

```bash
git add langtrans/spec.py tests/test_spec.py
git commit -m "feat: add Spec guard with &, |, ~ combinators"
```

---

### Task 4: Builder — @action Decorator and Trans Builder

**Files:**
- Create: `langtrans/builder.py`
- Create: `tests/test_builder.py`

- [ ] **Step 1: Write failing tests for @action decorator**

`tests/test_builder.py`:
```python
from langtrans.builder import Trans, action
from langtrans.nodes import (
    ActionNode,
    ConcurrentNode,
    LoopNode,
    OptionalNode,
    ProcedureNode,
    RetryNode,
    SequentialNode,
)
from langtrans.spec import Spec


def dummy(state):
    return state


def dummy_rollback(state):
    return state


def guard_true(state) -> bool:
    return True


class TestActionDecorator:
    def test_bare_decorator(self):
        @action
        def my_func(state):
            return state

        assert hasattr(my_func, "_langtrans_rollback")
        assert my_func._langtrans_rollback is None
        assert my_func("test") == "test"

    def test_decorator_with_rollback(self):
        def rollback_fn(state):
            return state

        @action(rollback=rollback_fn)
        def my_func(state):
            return state

        assert my_func._langtrans_rollback is rollback_fn
        assert my_func("test") == "test"


class TestTransBuilderSinglePrimitive:
    def test_sequential(self):
        tree = Trans().sequential(dummy, dummy).build()
        assert isinstance(tree, SequentialNode)
        assert len(tree.children) == 2
        assert all(isinstance(c, ActionNode) for c in tree.children)

    def test_concurrent(self):
        tree = Trans().concurrent(dummy, dummy, dummy).build()
        assert isinstance(tree, ConcurrentNode)
        assert len(tree.children) == 3

    def test_optional_both_branches(self):
        tree = Trans().optional(guard_true, then_=dummy, else_=dummy).build()
        assert isinstance(tree, OptionalNode)
        assert isinstance(tree.then_, ActionNode)
        assert isinstance(tree.else_, ActionNode)

    def test_optional_then_only(self):
        tree = Trans().optional(guard_true, then_=dummy).build()
        assert isinstance(tree, OptionalNode)
        assert tree.else_ is None

    def test_loop_fixed_count(self):
        tree = Trans().loop(body=dummy, times=3).build()
        assert isinstance(tree, LoopNode)
        assert tree.times == 3
        assert isinstance(tree.body, ActionNode)

    def test_loop_until_condition(self):
        tree = Trans().loop(body=dummy, until=guard_true).build()
        assert isinstance(tree, LoopNode)
        assert tree.until is guard_true

    def test_retry(self):
        tree = Trans().retry(dummy, max_attempts=5, delay=2.0).build()
        assert isinstance(tree, RetryNode)
        assert tree.max_attempts == 5
        assert tree.delay == 2.0
        assert isinstance(tree.body, ActionNode)

    def test_procedure(self):
        sub = Trans().sequential(dummy, dummy)
        tree = Trans().procedure("sub_flow", sub).build()
        assert isinstance(tree, ProcedureNode)
        assert tree.name == "sub_flow"
        assert isinstance(tree.body, SequentialNode)


class TestTransBuilderChaining:
    def test_two_methods_wrap_in_sequential(self):
        tree = (
            Trans()
            .sequential(dummy, dummy)
            .concurrent(dummy, dummy)
            .build()
        )
        assert isinstance(tree, SequentialNode)
        assert len(tree.children) == 2
        assert isinstance(tree.children[0], SequentialNode)
        assert isinstance(tree.children[1], ConcurrentNode)

    def test_single_method_no_wrapping(self):
        tree = Trans().sequential(dummy, dummy).build()
        assert isinstance(tree, SequentialNode)
        assert len(tree.children) == 2
        assert all(isinstance(c, ActionNode) for c in tree.children)


class TestTransBuilderNesting:
    def test_nested_trans_as_argument(self):
        tree = (
            Trans()
            .concurrent(
                dummy,
                Trans().sequential(dummy, dummy),
            )
            .build()
        )
        assert isinstance(tree, ConcurrentNode)
        assert len(tree.children) == 2
        assert isinstance(tree.children[0], ActionNode)
        assert isinstance(tree.children[1], SequentialNode)

    def test_deeply_nested(self):
        tree = (
            Trans()
            .sequential(
                dummy,
                Trans().concurrent(
                    Trans().sequential(dummy, dummy),
                    dummy,
                ),
                Trans().optional(guard_true, then_=dummy, else_=dummy),
            )
            .build()
        )
        assert isinstance(tree, SequentialNode)
        assert len(tree.children) == 3
        assert isinstance(tree.children[0], ActionNode)
        assert isinstance(tree.children[1], ConcurrentNode)
        assert isinstance(tree.children[2], OptionalNode)


class TestActionDecoratorInBuilder:
    def test_action_with_rollback_preserves_rollback(self):
        @action(rollback=dummy_rollback)
        def my_action(state):
            return state

        tree = Trans().sequential(my_action).build()
        assert isinstance(tree, SequentialNode)
        action_node = tree.children[0]
        assert isinstance(action_node, ActionNode)
        assert action_node.rollback is dummy_rollback

    def test_spec_guard_in_optional(self):
        spec = Spec(guard_true)
        tree = Trans().optional(spec, then_=dummy).build()
        assert isinstance(tree, OptionalNode)
        assert isinstance(tree.guard, Spec)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_builder.py -v`
Expected: ImportError

- [ ] **Step 3: Implement @action decorator and Trans builder**

`langtrans/builder.py`:
```python
from __future__ import annotations

from typing import Callable, Optional, Union

from langtrans.nodes import (
    ActionNode,
    ConcurrentNode,
    LoopNode,
    Node,
    OptionalNode,
    ProcedureNode,
    RetryNode,
    SequentialNode,
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

    def optional(
        self,
        guard: Union[Callable, Spec],
        *,
        then_: Union[Callable, Trans, Node],
        else_: Optional[Union[Callable, Trans, Node]] = None,
    ) -> Trans:
        then_node = _to_node(then_)
        else_node = _to_node(else_) if else_ is not None else None
        self._steps.append(OptionalNode(guard=guard, then_=then_node, else_=else_node))
        return self

    def loop(
        self,
        *,
        body: Union[Callable, Trans, Node],
        times: Optional[int] = None,
        until: Optional[Callable] = None,
    ) -> Trans:
        self._steps.append(LoopNode(body=_to_node(body), times=times, until=until))
        return self

    def retry(
        self,
        target: Union[Callable, Trans, Node],
        *,
        max_attempts: int = 3,
        delay: float = 0.0,
    ) -> Trans:
        self._steps.append(
            RetryNode(body=_to_node(target), max_attempts=max_attempts, delay=delay)
        )
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_builder.py -v`
Expected: All 16 tests pass

- [ ] **Step 5: Commit**

```bash
git add langtrans/builder.py tests/test_builder.py
git commit -m "feat: add Trans fluent builder and @action decorator"
```

---

### Task 5: Compiler — Sequential and Action

**Files:**
- Create: `langtrans/compiler.py`
- Create: `tests/test_compiler.py`

- [ ] **Step 1: Write failing tests for compiling ActionNode and SequentialNode**

`tests/test_compiler.py`:
```python
import operator
from typing import Annotated, TypedDict

from langtrans.builder import Trans
from langtrans.compiler import compile_graph
from langtrans.nodes import ActionNode, SequentialNode


class State(TypedDict):
    messages: Annotated[list, operator.add]
    metadata: dict


def action_a(state):
    meta = dict(state.get("metadata", {}))
    calls = list(meta.get("_calls", []))
    calls.append("a")
    meta["_calls"] = calls
    return {"metadata": meta}


def action_b(state):
    meta = dict(state.get("metadata", {}))
    calls = list(meta.get("_calls", []))
    calls.append("b")
    meta["_calls"] = calls
    return {"metadata": meta}


def action_c(state):
    meta = dict(state.get("metadata", {}))
    calls = list(meta.get("_calls", []))
    calls.append("c")
    meta["_calls"] = calls
    return {"metadata": meta}


class TestCompileAction:
    def test_single_action_compiles_and_runs(self):
        tree = ActionNode(func=action_a, name="action_a")
        graph = compile_graph(tree, state_schema=State)
        result = graph.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["a"]


class TestCompileSequential:
    def test_sequential_runs_in_order(self):
        tree = SequentialNode(children=[
            ActionNode(func=action_a, name="action_a"),
            ActionNode(func=action_b, name="action_b"),
            ActionNode(func=action_c, name="action_c"),
        ])
        graph = compile_graph(tree, state_schema=State)
        result = graph.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["a", "b", "c"]

    def test_sequential_via_builder(self):
        graph = (
            Trans(state_schema=State)
            .sequential(action_a, action_b, action_c)
            .compile()
        )
        result = graph.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["a", "b", "c"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_compiler.py -v`
Expected: ImportError — `langtrans.compiler` does not exist yet

- [ ] **Step 3: Implement compiler for ActionNode and SequentialNode**

`langtrans/compiler.py`:
```python
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from langtrans.nodes import (
    ActionNode,
    ConcurrentNode,
    LoopNode,
    Node,
    OptionalNode,
    ProcedureNode,
    RetryNode,
    SequentialNode,
)


class _Compiler:
    def __init__(self, graph: StateGraph):
        self._graph = graph
        self._counter = 0

    def _unique_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def compile_node(self, node: Node) -> tuple[str, str]:
        if isinstance(node, ActionNode):
            return self._compile_action(node)
        if isinstance(node, SequentialNode):
            return self._compile_sequential(node)
        if isinstance(node, ConcurrentNode):
            return self._compile_concurrent(node)
        if isinstance(node, OptionalNode):
            return self._compile_optional(node)
        if isinstance(node, LoopNode):
            return self._compile_loop(node)
        if isinstance(node, RetryNode):
            return self._compile_retry(node)
        if isinstance(node, ProcedureNode):
            return self._compile_procedure(node)
        raise TypeError(f"Unknown node type: {type(node)}")

    def _compile_action(self, node: ActionNode) -> tuple[str, str]:
        name = node.name or node.func.__name__
        node_id = name
        if node_id in [n for n, _ in self._graph._nodes.items()]:
            node_id = self._unique_id(name)
        self._graph.add_node(node_id, node.func)
        return node_id, node_id

    def _compile_sequential(self, node: SequentialNode) -> tuple[str, str]:
        if not node.children:
            noop_id = self._unique_id("noop")
            self._graph.add_node(noop_id, lambda state: {})
            return noop_id, noop_id

        first_entry = None
        prev_exit = None
        for child in node.children:
            entry, exit_ = self.compile_node(child)
            if first_entry is None:
                first_entry = entry
            if prev_exit is not None:
                self._graph.add_edge(prev_exit, entry)
            prev_exit = exit_
        return first_entry, prev_exit

    def _compile_concurrent(self, node: ConcurrentNode) -> tuple[str, str]:
        raise NotImplementedError("ConcurrentNode: implemented in Task 6")

    def _compile_optional(self, node: OptionalNode) -> tuple[str, str]:
        raise NotImplementedError("OptionalNode: implemented in Task 7")

    def _compile_loop(self, node: LoopNode) -> tuple[str, str]:
        raise NotImplementedError("LoopNode: implemented in Task 8")

    def _compile_retry(self, node: RetryNode) -> tuple[str, str]:
        raise NotImplementedError("RetryNode: implemented in Task 9")

    def _compile_procedure(self, node: ProcedureNode) -> tuple[str, str]:
        raise NotImplementedError("ProcedureNode: implemented in Task 10")


def compile_graph(tree: Node, *, state_schema, **kwargs) -> Any:
    graph = StateGraph(state_schema)
    compiler = _Compiler(graph)
    entry, exit_ = compiler.compile_node(tree)
    graph.add_edge(START, entry)
    graph.add_edge(exit_, END)
    return graph.compile(**kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_compiler.py -v`
Expected: All 3 tests pass

- [ ] **Step 5: Commit**

```bash
git add langtrans/compiler.py tests/test_compiler.py
git commit -m "feat: add compiler with ActionNode and SequentialNode support"
```

---

### Task 6: Compiler — ConcurrentNode

**Files:**
- Modify: `langtrans/compiler.py`
- Modify: `tests/test_compiler.py`

- [ ] **Step 1: Write failing tests for ConcurrentNode compilation**

Append to `tests/test_compiler.py`:
```python
class TestCompileConcurrent:
    def test_concurrent_runs_all_branches(self):
        graph = (
            Trans(state_schema=State)
            .concurrent(action_a, action_b, action_c)
            .compile()
        )
        result = graph.invoke({"messages": [], "metadata": {}})
        calls = result["metadata"]["_calls"]
        assert set(calls) == {"a", "b", "c"}
        assert len(calls) == 3

    def test_concurrent_then_sequential(self):
        graph = (
            Trans(state_schema=State)
            .concurrent(action_a, action_b)
            .sequential(action_c)
            .compile()
        )
        result = graph.invoke({"messages": [], "metadata": {}})
        calls = result["metadata"]["_calls"]
        assert "c" in calls
        assert calls[-1] == "c"
```

- [ ] **Step 2: Run tests to verify the new tests fail**

Run: `pytest tests/test_compiler.py::TestCompileConcurrent -v`
Expected: NotImplementedError

- [ ] **Step 3: Implement _compile_concurrent**

Replace the `_compile_concurrent` method in `langtrans/compiler.py`:
```python
    def _compile_concurrent(self, node: ConcurrentNode) -> tuple[str, str]:
        fork_id = self._unique_id("fork")
        join_id = self._unique_id("join")

        self._graph.add_node(fork_id, lambda state: {})
        self._graph.add_node(join_id, lambda state: {})

        for child in node.children:
            entry, exit_ = self.compile_node(child)
            self._graph.add_edge(fork_id, entry)
            self._graph.add_edge(exit_, join_id)

        return fork_id, join_id
```

- [ ] **Step 4: Run all compiler tests to verify they pass**

Run: `pytest tests/test_compiler.py -v`
Expected: All 5 tests pass

- [ ] **Step 5: Commit**

```bash
git add langtrans/compiler.py tests/test_compiler.py
git commit -m "feat: add ConcurrentNode compilation (fan-out/fan-in)"
```

---

### Task 7: Compiler — OptionalNode

**Files:**
- Modify: `langtrans/compiler.py`
- Modify: `tests/test_compiler.py`

- [ ] **Step 1: Write failing tests for OptionalNode compilation**

Append to `tests/test_compiler.py`:
```python
from langtrans.spec import Spec


def guard_always_true(state) -> bool:
    return True


def guard_always_false(state) -> bool:
    return False


def guard_has_flag(state) -> bool:
    return state.get("metadata", {}).get("flag", False)


class TestCompileOptional:
    def test_optional_takes_then_branch(self):
        graph = (
            Trans(state_schema=State)
            .optional(guard_always_true, then_=action_a, else_=action_b)
            .compile()
        )
        result = graph.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["a"]

    def test_optional_takes_else_branch(self):
        graph = (
            Trans(state_schema=State)
            .optional(guard_always_false, then_=action_a, else_=action_b)
            .compile()
        )
        result = graph.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["b"]

    def test_optional_without_else_skips(self):
        graph = (
            Trans(state_schema=State)
            .optional(guard_always_false, then_=action_a)
            .compile()
        )
        result = graph.invoke({"messages": [], "metadata": {}})
        assert result["metadata"].get("_calls") is None or result["metadata"]["_calls"] == []

    def test_optional_with_spec_guard(self):
        spec = Spec(guard_has_flag)
        graph = (
            Trans(state_schema=State)
            .optional(spec, then_=action_a, else_=action_b)
            .compile()
        )
        result = graph.invoke({"messages": [], "metadata": {"flag": True}})
        assert result["metadata"]["_calls"] == ["a"]

        result = graph.invoke({"messages": [], "metadata": {"flag": False}})
        assert result["metadata"]["_calls"] == ["b"]

    def test_optional_dynamic_guard(self):
        graph = (
            Trans(state_schema=State)
            .sequential(action_a)
            .optional(
                lambda s: "a" in s.get("metadata", {}).get("_calls", []),
                then_=action_b,
                else_=action_c,
            )
            .compile()
        )
        result = graph.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["a", "b"]
```

- [ ] **Step 2: Run tests to verify the new tests fail**

Run: `pytest tests/test_compiler.py::TestCompileOptional -v`
Expected: NotImplementedError

- [ ] **Step 3: Implement _compile_optional**

Replace the `_compile_optional` method in `langtrans/compiler.py`:
```python
    def _compile_optional(self, node: OptionalNode) -> tuple[str, str]:
        decision_id = self._unique_id("decision")
        merge_id = self._unique_id("merge")

        self._graph.add_node(decision_id, lambda state: {})
        self._graph.add_node(merge_id, lambda state: {})

        then_entry, then_exit = self.compile_node(node.then_)
        self._graph.add_edge(then_exit, merge_id)

        guard = node.guard

        if node.else_ is not None:
            else_entry, else_exit = self.compile_node(node.else_)
            self._graph.add_edge(else_exit, merge_id)

            def route_with_else(state, _guard=guard, _then=then_entry, _else=else_entry):
                result = _guard(state) if callable(_guard) else _guard(state)
                return _then if result else _else

            self._graph.add_conditional_edges(
                decision_id,
                route_with_else,
                [then_entry, else_entry],
            )
        else:
            def route_without_else(state, _guard=guard, _then=then_entry, _merge=merge_id):
                result = _guard(state) if callable(_guard) else _guard(state)
                return _then if result else _merge

            self._graph.add_conditional_edges(
                decision_id,
                route_without_else,
                [then_entry, merge_id],
            )

        return decision_id, merge_id
```

- [ ] **Step 4: Run all compiler tests to verify they pass**

Run: `pytest tests/test_compiler.py -v`
Expected: All 10 tests pass

- [ ] **Step 5: Commit**

```bash
git add langtrans/compiler.py tests/test_compiler.py
git commit -m "feat: add OptionalNode compilation with conditional routing"
```

---

### Task 8: Compiler — LoopNode

**Files:**
- Modify: `langtrans/compiler.py`
- Modify: `tests/test_compiler.py`

- [ ] **Step 1: Write failing tests for LoopNode compilation**

Append to `tests/test_compiler.py`:
```python
def counting_action(state):
    meta = dict(state.get("metadata", {}))
    count = meta.get("count", 0) + 1
    meta["count"] = count
    calls = list(meta.get("_calls", []))
    calls.append(f"iter_{count}")
    meta["_calls"] = calls
    return {"metadata": meta}


class TestCompileLoop:
    def test_loop_fixed_count(self):
        graph = (
            Trans(state_schema=State)
            .loop(body=counting_action, times=3)
            .compile()
        )
        result = graph.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["count"] == 3
        assert len(result["metadata"]["_calls"]) == 3

    def test_loop_until_condition(self):
        graph = (
            Trans(state_schema=State)
            .loop(
                body=counting_action,
                until=lambda s: s.get("metadata", {}).get("count", 0) >= 2,
            )
            .compile()
        )
        result = graph.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["count"] == 2

    def test_loop_after_sequential(self):
        graph = (
            Trans(state_schema=State)
            .sequential(action_a)
            .loop(body=counting_action, times=2)
            .compile()
        )
        result = graph.invoke({"messages": [], "metadata": {}})
        assert "a" in result["metadata"]["_calls"]
        assert result["metadata"]["count"] == 2
```

- [ ] **Step 2: Run tests to verify the new tests fail**

Run: `pytest tests/test_compiler.py::TestCompileLoop -v`
Expected: NotImplementedError

- [ ] **Step 3: Implement _compile_loop**

Replace the `_compile_loop` method in `langtrans/compiler.py`:
```python
    def _compile_loop(self, node: LoopNode) -> tuple[str, str]:
        loop_id = self._unique_id("loop")
        loop_gate_id = self._unique_id("loop_gate")
        loop_exit_id = self._unique_id("loop_exit")

        if node.times is not None:
            counter_key = f"_loop_{loop_id}_counter"

            def loop_init(state, _key=counter_key):
                meta = dict(state.get("metadata", {}))
                lt = dict(meta.get("_langtrans", {}))
                lt[_key] = 0
                meta["_langtrans"] = lt
                return {"metadata": meta}

            self._graph.add_node(loop_id, loop_init)

            body_entry, body_exit = self.compile_node(node.body)

            def loop_gate_fn(state, _key=counter_key):
                meta = dict(state.get("metadata", {}))
                lt = dict(meta.get("_langtrans", {}))
                lt[_key] = lt.get(_key, 0) + 1
                meta["_langtrans"] = lt
                return {"metadata": meta}

            self._graph.add_node(loop_gate_id, loop_gate_fn)
            self._graph.add_node(loop_exit_id, lambda state: {})

            self._graph.add_edge(loop_id, body_entry)
            self._graph.add_edge(body_exit, loop_gate_id)

            def check_count(state, _key=counter_key, _times=node.times,
                            _body=body_entry, _exit=loop_exit_id):
                count = state.get("metadata", {}).get("_langtrans", {}).get(_key, 0)
                return _body if count < _times else _exit

            self._graph.add_conditional_edges(
                loop_gate_id, check_count, [body_entry, loop_exit_id]
            )
            return loop_id, loop_exit_id

        elif node.until is not None:
            self._graph.add_node(loop_id, lambda state: {})
            self._graph.add_node(loop_exit_id, lambda state: {})

            body_entry, body_exit = self.compile_node(node.body)

            self._graph.add_edge(loop_id, body_entry)

            until_fn = node.until

            def check_until(state, _until=until_fn, _body=body_entry, _exit=loop_exit_id):
                return _exit if _until(state) else _body

            self._graph.add_conditional_edges(
                body_exit, check_until, [body_entry, loop_exit_id]
            )
            return loop_id, loop_exit_id

        else:
            raise ValueError("LoopNode requires either 'times' or 'until'")
```

- [ ] **Step 4: Run all compiler tests to verify they pass**

Run: `pytest tests/test_compiler.py -v`
Expected: All 13 tests pass

- [ ] **Step 5: Commit**

```bash
git add langtrans/compiler.py tests/test_compiler.py
git commit -m "feat: add LoopNode compilation (fixed count and condition-based)"
```

---

### Task 9: Compiler — RetryNode

**Files:**
- Modify: `langtrans/compiler.py`
- Modify: `tests/test_compiler.py`

- [ ] **Step 1: Write failing tests for RetryNode compilation**

Append to `tests/test_compiler.py`:
```python
class TestCompileRetry:
    def test_retry_succeeds_first_try(self):
        graph = (
            Trans(state_schema=State)
            .retry(action_a, max_attempts=3)
            .compile()
        )
        result = graph.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["a"]

    def test_retry_succeeds_after_failures(self):
        call_count = {"n": 0}

        def flaky_action(state):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ValueError("transient error")
            meta = dict(state.get("metadata", {}))
            calls = list(meta.get("_calls", []))
            calls.append("flaky_ok")
            meta["_calls"] = calls
            return {"metadata": meta}

        flaky_action.__name__ = "flaky_action"

        graph = (
            Trans(state_schema=State)
            .retry(flaky_action, max_attempts=5, delay=0.0)
            .compile()
        )
        result = graph.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["flaky_ok"]
        assert call_count["n"] == 3

    def test_retry_exhausted_raises(self):
        def always_fail(state):
            raise ValueError("permanent error")

        always_fail.__name__ = "always_fail"

        graph = (
            Trans(state_schema=State)
            .retry(always_fail, max_attempts=2, delay=0.0)
            .compile()
        )
        import pytest
        with pytest.raises(ValueError, match="permanent error"):
            graph.invoke({"messages": [], "metadata": {}})
```

- [ ] **Step 2: Run tests to verify the new tests fail**

Run: `pytest tests/test_compiler.py::TestCompileRetry -v`
Expected: NotImplementedError

- [ ] **Step 3: Implement _compile_retry**

Replace the `_compile_retry` method in `langtrans/compiler.py`:
```python
    def _compile_retry(self, node: RetryNode) -> tuple[str, str]:
        import time

        retry_id = self._unique_id("retry")
        retry_exit_id = self._unique_id("retry_exit")
        attempts_key = f"_retry_{retry_id}_attempts"

        body_func = node.body.func if isinstance(node.body, ActionNode) else None
        body_name = (node.body.name or node.body.func.__name__) if isinstance(node.body, ActionNode) else "retry_body"

        max_attempts = node.max_attempts
        delay = node.delay

        if body_func is not None:
            def retry_wrapper(state, _func=body_func, _max=max_attempts,
                              _delay=delay, _key=attempts_key):
                last_err = None
                for attempt in range(1, _max + 1):
                    try:
                        return _func(state)
                    except Exception as e:
                        last_err = e
                        if attempt < _max and _delay > 0:
                            time.sleep(_delay)
                raise last_err

            wrapper_id = self._unique_id(f"retry_{body_name}")
            self._graph.add_node(wrapper_id, retry_wrapper)
            return wrapper_id, wrapper_id
        else:
            raise NotImplementedError(
                "RetryNode with non-ActionNode body requires graph-level retry (not yet supported)"
            )
```

- [ ] **Step 4: Run all compiler tests to verify they pass**

Run: `pytest tests/test_compiler.py -v`
Expected: All 16 tests pass

- [ ] **Step 5: Commit**

```bash
git add langtrans/compiler.py tests/test_compiler.py
git commit -m "feat: add RetryNode compilation with attempt tracking"
```

---

### Task 10: Compiler — ProcedureNode

**Files:**
- Modify: `langtrans/compiler.py`
- Modify: `tests/test_compiler.py`

- [ ] **Step 1: Write failing tests for ProcedureNode compilation**

Append to `tests/test_compiler.py`:
```python
class TestCompileProcedure:
    def test_procedure_compiles_inline(self):
        sub = Trans().sequential(action_a, action_b)
        graph = (
            Trans(state_schema=State)
            .procedure("sub_flow", sub)
            .compile()
        )
        result = graph.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["a", "b"]

    def test_procedure_with_surrounding_steps(self):
        sub = Trans().sequential(action_b)
        graph = (
            Trans(state_schema=State)
            .sequential(action_a)
            .procedure("middle", sub)
            .sequential(action_c)
            .compile()
        )
        result = graph.invoke({"messages": [], "metadata": {}})
        calls = result["metadata"]["_calls"]
        assert calls == ["a", "b", "c"]

    def test_multiple_procedures_no_name_collision(self):
        sub1 = Trans().sequential(action_a)
        sub2 = Trans().sequential(action_b)
        graph = (
            Trans(state_schema=State)
            .procedure("proc1", sub1)
            .procedure("proc2", sub2)
            .compile()
        )
        result = graph.invoke({"messages": [], "metadata": {}})
        calls = result["metadata"]["_calls"]
        assert set(calls) >= {"a", "b"}
```

- [ ] **Step 2: Run tests to verify the new tests fail**

Run: `pytest tests/test_compiler.py::TestCompileProcedure -v`
Expected: NotImplementedError

- [ ] **Step 3: Implement _compile_procedure**

Replace the `_compile_procedure` method in `langtrans/compiler.py`:
```python
    def _compile_procedure(self, node: ProcedureNode) -> tuple[str, str]:
        saved_counter = self._counter
        prefix = node.name

        original_add_node = self._graph.add_node

        def prefixed_add_node(name, fn, **kwargs):
            prefixed_name = f"{prefix}.{name}"
            return original_add_node(prefixed_name, fn, **kwargs)

        self._graph.add_node = prefixed_add_node
        old_unique = self._unique_id

        def prefixed_unique(p):
            return f"{prefix}.{old_unique(p)}"

        self._unique_id = prefixed_unique

        try:
            entry, exit_ = self.compile_node(node.body)
        finally:
            self._graph.add_node = original_add_node
            self._unique_id = old_unique

        return entry, exit_
```

- [ ] **Step 4: Run all compiler tests to verify they pass**

Run: `pytest tests/test_compiler.py -v`
Expected: All 19 tests pass

- [ ] **Step 5: Commit**

```bash
git add langtrans/compiler.py tests/test_compiler.py
git commit -m "feat: add ProcedureNode compilation with name-prefixed subgraphs"
```

---

### Task 11: Compiler — Rollback / Compensation

**Files:**
- Modify: `langtrans/compiler.py`
- Modify: `tests/test_compiler.py`

- [ ] **Step 1: Write failing tests for rollback in Sequential**

Append to `tests/test_compiler.py`:
```python
from langtrans.builder import action


class TestCompileRollback:
    def test_rollback_runs_on_failure(self):
        rollback_log = []

        def rollback_a(state):
            rollback_log.append("rollback_a")
            return {}

        def rollback_b(state):
            rollback_log.append("rollback_b")
            return {}

        @action(rollback=rollback_a)
        def step_a(state):
            meta = dict(state.get("metadata", {}))
            calls = list(meta.get("_calls", []))
            calls.append("a")
            meta["_calls"] = calls
            return {"metadata": meta}

        @action(rollback=rollback_b)
        def step_b(state):
            meta = dict(state.get("metadata", {}))
            calls = list(meta.get("_calls", []))
            calls.append("b")
            meta["_calls"] = calls
            return {"metadata": meta}

        def step_c_fail(state):
            raise ValueError("step_c failed")

        step_c_fail.__name__ = "step_c_fail"

        graph = (
            Trans(state_schema=State)
            .sequential(step_a, step_b, step_c_fail)
            .compile()
        )

        import pytest
        with pytest.raises(ValueError, match="step_c failed"):
            graph.invoke({"messages": [], "metadata": {}})

        assert "rollback_b" in rollback_log
        assert "rollback_a" in rollback_log
        assert rollback_log.index("rollback_b") < rollback_log.index("rollback_a")

    def test_no_rollback_when_all_succeed(self):
        rollback_log = []

        def rollback_a(state):
            rollback_log.append("rollback_a")
            return {}

        @action(rollback=rollback_a)
        def step_a(state):
            meta = dict(state.get("metadata", {}))
            calls = list(meta.get("_calls", []))
            calls.append("a")
            meta["_calls"] = calls
            return {"metadata": meta}

        graph = (
            Trans(state_schema=State)
            .sequential(step_a, action_b)
            .compile()
        )

        result = graph.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["a", "b"]
        assert rollback_log == []
```

- [ ] **Step 2: Run tests to verify the new tests fail**

Run: `pytest tests/test_compiler.py::TestCompileRollback -v`
Expected: Tests fail — rollback not triggered

- [ ] **Step 3: Implement rollback support in _compile_sequential**

Add a helper method and modify `_compile_action` and `_compile_sequential` in `langtrans/compiler.py`:

```python
    def _compile_action(self, node: ActionNode) -> tuple[str, str]:
        name = node.name or node.func.__name__
        node_id = name
        if node_id in [n for n in self._graph._nodes]:
            node_id = self._unique_id(name)

        if node.rollback is not None:
            func = node.func
            rollback = node.rollback
            rollback_key = f"_rollback_stack"

            def action_with_rollback(state, _func=func, _rollback=rollback, _key=rollback_key):
                meta = dict(state.get("metadata", {}))
                lt = dict(meta.get("_langtrans", {}))
                stack = list(lt.get(_key, []))
                try:
                    result = _func(state)
                    result_meta = dict(result.get("metadata", meta))
                    result_lt = dict(result_meta.get("_langtrans", lt))
                    stack.append(id(_rollback))
                    result_lt[_key] = stack
                    result_meta["_langtrans"] = result_lt
                    result["metadata"] = result_meta
                    return result
                except Exception:
                    for rb_id in reversed(stack):
                        for registered_rb in _Compiler._rollback_registry.values():
                            if id(registered_rb) == rb_id:
                                try:
                                    registered_rb(state)
                                except Exception:
                                    pass
                    raise

            _Compiler._rollback_registry[node_id] = rollback
            self._graph.add_node(node_id, action_with_rollback)
        else:
            self._graph.add_node(node_id, node.func)

        return node_id, node_id
```

Actually, the rollback-via-graph-edges approach from the spec is complex. A simpler and more reliable approach: wrap sequential children that have rollbacks so that on failure, the entire sequential runs compensation. Let me redesign this to be practical.

Replace the rollback implementation with a wrapper approach in `_compile_sequential`:

```python
    def _compile_sequential(self, node: SequentialNode) -> tuple[str, str]:
        if not node.children:
            noop_id = self._unique_id("noop")
            self._graph.add_node(noop_id, lambda state: {})
            return noop_id, noop_id

        has_rollbacks = any(
            isinstance(c, ActionNode) and c.rollback is not None
            for c in node.children
        )

        if has_rollbacks:
            return self._compile_sequential_with_rollback(node)

        first_entry = None
        prev_exit = None
        for child in node.children:
            entry, exit_ = self.compile_node(child)
            if first_entry is None:
                first_entry = entry
            if prev_exit is not None:
                self._graph.add_edge(prev_exit, entry)
            prev_exit = exit_
        return first_entry, prev_exit

    def _compile_sequential_with_rollback(self, node: SequentialNode) -> tuple[str, str]:
        steps = []
        for child in node.children:
            if isinstance(child, ActionNode):
                steps.append((child.func, child.rollback, child.name or child.func.__name__))
            else:
                raise NotImplementedError(
                    "Rollback in sequential only supports ActionNode children"
                )

        seq_id = self._unique_id("seq_rollback")

        def sequential_with_compensation(state, _steps=steps):
            completed_rollbacks = []
            current_state = state
            for func, rollback, name in _steps:
                try:
                    result = func(current_state)
                    merged = dict(current_state)
                    merged.update(result)
                    current_state = merged
                    if rollback is not None:
                        completed_rollbacks.append(rollback)
                except Exception:
                    for rb in reversed(completed_rollbacks):
                        try:
                            rb(current_state)
                        except Exception:
                            pass
                    raise
            result_update = {}
            for key in current_state:
                if key in state and current_state[key] != state[key]:
                    result_update[key] = current_state[key]
                elif key not in state:
                    result_update[key] = current_state[key]
            return result_update if result_update else {}

        self._graph.add_node(seq_id, sequential_with_compensation)
        return seq_id, seq_id
```

- [ ] **Step 4: Run all compiler tests to verify they pass**

Run: `pytest tests/test_compiler.py -v`
Expected: All 21 tests pass

- [ ] **Step 5: Commit**

```bash
git add langtrans/compiler.py tests/test_compiler.py
git commit -m "feat: add rollback/compensation support in sequential nodes"
```

---

### Task 12: Public API Exports

**Files:**
- Modify: `langtrans/__init__.py`

- [ ] **Step 1: Write a smoke test for public imports**

Create `tests/test_public_api.py`:
```python
def test_public_imports():
    from langtrans import Trans, action, Spec
    from langtrans import (
        ActionNode,
        SequentialNode,
        ConcurrentNode,
        OptionalNode,
        LoopNode,
        RetryNode,
        ProcedureNode,
        Node,
    )
    assert Trans is not None
    assert action is not None
    assert Spec is not None
    assert ActionNode is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_public_api.py -v`
Expected: ImportError

- [ ] **Step 3: Update __init__.py with exports**

`langtrans/__init__.py`:
```python
"""langtrans — A high-level DSL that compiles to LangGraph graphs."""

from langtrans.builder import Trans, action
from langtrans.nodes import (
    ActionNode,
    ConcurrentNode,
    LoopNode,
    Node,
    OptionalNode,
    ProcedureNode,
    RetryNode,
    SequentialNode,
)
from langtrans.spec import Spec

__all__ = [
    "Trans",
    "action",
    "Spec",
    "ActionNode",
    "ConcurrentNode",
    "LoopNode",
    "Node",
    "OptionalNode",
    "ProcedureNode",
    "RetryNode",
    "SequentialNode",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_public_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add langtrans/__init__.py tests/test_public_api.py
git commit -m "feat: add public API exports"
```

---

### Task 13: Integration Tests

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration tests that exercise full compile + invoke**

`tests/test_integration.py`:
```python
import operator
from typing import Annotated, TypedDict

from langtrans import Trans, Spec, action


class State(TypedDict):
    messages: Annotated[list, operator.add]
    metadata: dict


def append_call(name):
    def fn(state):
        meta = dict(state.get("metadata", {}))
        calls = list(meta.get("_calls", []))
        calls.append(name)
        meta["_calls"] = calls
        return {"metadata": meta}
    fn.__name__ = name
    fn.__qualname__ = name
    return fn


fetch = append_call("fetch")
llm = append_call("llm")
respond = append_call("respond")
search_web = append_call("search_web")
search_db = append_call("search_db")
clarify = append_call("clarify")


class TestSequentialAgent:
    def test_three_step_pipeline(self):
        app = (
            Trans(state_schema=State)
            .sequential(fetch, llm, respond)
            .compile()
        )
        result = app.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["fetch", "llm", "respond"]


class TestParallelWithConditional:
    def test_concurrent_then_conditional(self):
        has_results = lambda s: len(s.get("metadata", {}).get("_calls", [])) >= 2

        app = (
            Trans(state_schema=State)
            .concurrent(search_web, search_db)
            .optional(has_results, then_=respond, else_=clarify)
            .compile()
        )
        result = app.invoke({"messages": [], "metadata": {}})
        calls = result["metadata"]["_calls"]
        assert "search_web" in calls
        assert "search_db" in calls
        assert "respond" in calls


class TestLoopWithNesting:
    def test_loop_with_nested_optional(self):
        counter = {"n": 0}

        def increment(state):
            counter["n"] += 1
            meta = dict(state.get("metadata", {}))
            meta["count"] = counter["n"]
            calls = list(meta.get("_calls", []))
            calls.append(f"inc_{counter['n']}")
            meta["_calls"] = calls
            return {"metadata": meta}

        increment.__name__ = "increment"

        def is_even(state):
            return state.get("metadata", {}).get("count", 0) % 2 == 0

        mark_even = append_call("even")
        mark_odd = append_call("odd")

        app = (
            Trans(state_schema=State)
            .loop(
                body=Trans().sequential(
                    increment,
                    Trans().optional(is_even, then_=mark_even, else_=mark_odd),
                ),
                times=3,
            )
            .compile()
        )
        result = app.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["count"] == 3


class TestProcedureComposition:
    def test_procedure_as_reusable_unit(self):
        fetch_pipeline = Trans().sequential(fetch, llm)

        app = (
            Trans(state_schema=State)
            .procedure("fetch_pipe", fetch_pipeline)
            .sequential(respond)
            .compile()
        )
        result = app.invoke({"messages": [], "metadata": {}})
        calls = result["metadata"]["_calls"]
        assert "fetch" in calls
        assert "llm" in calls
        assert "respond" in calls


class TestSpecGuards:
    def test_composed_spec_in_optional(self):
        has_fetch = Spec(lambda s: "fetch" in s.get("metadata", {}).get("_calls", []))
        has_llm = Spec(lambda s: "llm" in s.get("metadata", {}).get("_calls", []))

        app = (
            Trans(state_schema=State)
            .sequential(fetch, llm)
            .optional(has_fetch & has_llm, then_=respond, else_=clarify)
            .compile()
        )
        result = app.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["fetch", "llm", "respond"]
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_integration.py -v`
Expected: All 5 tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full compile + invoke flows"
```

---

### Task 14: Example Scripts

**Files:**
- Create: `examples/sequential_agent.py`
- Create: `examples/parallel_tools.py`
- Create: `examples/react_agent.py`
- Create: `examples/rollback_pipeline.py`

- [ ] **Step 1: Create examples directory and sequential_agent.py**

`examples/sequential_agent.py`:
```python
"""Example 1: Sequential Agent — fetch data, call LLM, respond."""
import operator
from typing import Annotated, TypedDict

from langtrans import Trans


class State(TypedDict):
    messages: Annotated[list, operator.add]
    metadata: dict


def fetch_data(state):
    meta = dict(state.get("metadata", {}))
    meta["data"] = "fetched content"
    return {"metadata": meta}


def call_llm(state):
    data = state["metadata"].get("data", "")
    meta = dict(state.get("metadata", {}))
    meta["response"] = f"LLM processed: {data}"
    return {"metadata": meta}


def format_response(state):
    meta = dict(state.get("metadata", {}))
    meta["final"] = f"Response: {meta.get('response', '')}"
    return {"metadata": meta}


app = (
    Trans(state_schema=State)
    .sequential(fetch_data, call_llm, format_response)
    .compile()
)

if __name__ == "__main__":
    result = app.invoke({"messages": [], "metadata": {}})
    print(result["metadata"]["final"])
```

- [ ] **Step 2: Create parallel_tools.py**

`examples/parallel_tools.py`:
```python
"""Example 2: Parallel tool calls with conditional routing."""
import operator
from typing import Annotated, TypedDict

from langtrans import Trans


class State(TypedDict):
    messages: Annotated[list, operator.add]
    metadata: dict


def search_web(state):
    meta = dict(state.get("metadata", {}))
    results = list(meta.get("results", []))
    results.append("web result")
    meta["results"] = results
    return {"metadata": meta}


def search_db(state):
    meta = dict(state.get("metadata", {}))
    results = list(meta.get("results", []))
    results.append("db result")
    meta["results"] = results
    return {"metadata": meta}


def search_docs(state):
    meta = dict(state.get("metadata", {}))
    results = list(meta.get("results", []))
    results.append("docs result")
    meta["results"] = results
    return {"metadata": meta}


def has_enough_info(state) -> bool:
    return len(state.get("metadata", {}).get("results", [])) >= 2


def respond(state):
    meta = dict(state.get("metadata", {}))
    meta["answer"] = f"Found {len(meta.get('results', []))} results"
    return {"metadata": meta}


def ask_clarification(state):
    meta = dict(state.get("metadata", {}))
    meta["answer"] = "Need more information"
    return {"metadata": meta}


app = (
    Trans(state_schema=State)
    .concurrent(search_web, search_db, search_docs)
    .optional(has_enough_info, then_=respond, else_=ask_clarification)
    .compile()
)

if __name__ == "__main__":
    result = app.invoke({"messages": [], "metadata": {}})
    print(result["metadata"]["answer"])
```

- [ ] **Step 3: Create react_agent.py**

`examples/react_agent.py`:
```python
"""Example 3: ReAct-style agent loop with retry on tool failures."""
import operator
from typing import Annotated, TypedDict

from langtrans import Trans


class State(TypedDict):
    messages: Annotated[list, operator.add]
    metadata: dict


def call_agent(state):
    meta = dict(state.get("metadata", {}))
    step = meta.get("step", 0) + 1
    meta["step"] = step
    if step >= 3:
        meta["done"] = True
        meta["tool_needed"] = False
    else:
        meta["tool_needed"] = True
    return {"metadata": meta}


def should_use_tool(state) -> bool:
    return state.get("metadata", {}).get("tool_needed", False)


call_count = {"n": 0}


def execute_tool(state):
    call_count["n"] += 1
    meta = dict(state.get("metadata", {}))
    tools = list(meta.get("tool_results", []))
    tools.append(f"tool_result_{meta.get('step', 0)}")
    meta["tool_results"] = tools
    return {"metadata": meta}


app = (
    Trans(state_schema=State)
    .loop(
        until=lambda s: s.get("metadata", {}).get("done", False),
        body=Trans().sequential(
            call_agent,
            Trans().optional(
                should_use_tool,
                then_=Trans().retry(execute_tool, max_attempts=3),
            ),
        ),
    )
    .compile()
)

if __name__ == "__main__":
    result = app.invoke({"messages": [], "metadata": {}})
    print(f"Steps: {result['metadata']['step']}")
    print(f"Tool results: {result['metadata'].get('tool_results', [])}")
    print(f"Done: {result['metadata']['done']}")
```

- [ ] **Step 4: Create rollback_pipeline.py**

`examples/rollback_pipeline.py`:
```python
"""Example 4: Multi-step pipeline with Saga-pattern rollback."""
import operator
from typing import Annotated, TypedDict

from langtrans import Trans, action


class State(TypedDict):
    messages: Annotated[list, operator.add]
    metadata: dict


rollback_log = []


def undo_write_db(state):
    rollback_log.append("undo_write_db")
    print("Rolling back: undo_write_db")
    return {}


def undo_notify(state):
    rollback_log.append("undo_notify")
    print("Rolling back: undo_notify")
    return {}


def validate_input(state):
    meta = dict(state.get("metadata", {}))
    meta["validated"] = True
    return {"metadata": meta}


@action(rollback=undo_write_db)
def write_db(state):
    meta = dict(state.get("metadata", {}))
    meta["db_written"] = True
    return {"metadata": meta}


@action(rollback=undo_notify)
def notify_service(state):
    if state.get("metadata", {}).get("should_fail", False):
        raise ValueError("Notification service unavailable")
    meta = dict(state.get("metadata", {}))
    meta["notified"] = True
    return {"metadata": meta}


def send_confirmation(state):
    meta = dict(state.get("metadata", {}))
    meta["confirmed"] = True
    return {"metadata": meta}


app = (
    Trans(state_schema=State)
    .sequential(validate_input, write_db, notify_service, send_confirmation)
    .compile()
)

if __name__ == "__main__":
    print("=== Success case ===")
    rollback_log.clear()
    result = app.invoke({"messages": [], "metadata": {}})
    print(f"Result: {result['metadata']}")
    print(f"Rollbacks: {rollback_log}")

    print("\n=== Failure case ===")
    rollback_log.clear()
    try:
        result = app.invoke({"messages": [], "metadata": {"should_fail": True}})
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Rollbacks: {rollback_log}")
```

- [ ] **Step 5: Run all examples to verify they work**

```bash
python examples/sequential_agent.py
python examples/parallel_tools.py
python examples/react_agent.py
python examples/rollback_pipeline.py
```

Expected: Each prints output without errors.

- [ ] **Step 6: Run the full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add examples/
git commit -m "docs: add 4 example scripts showing langtrans vs raw LangGraph"
```

---

### Task 15: Final Verification

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite with coverage**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass (nodes, spec, builder, compiler, integration, public API)

- [ ] **Step 2: Run all examples**

```bash
python examples/sequential_agent.py && python examples/parallel_tools.py && python examples/react_agent.py && python examples/rollback_pipeline.py
```
Expected: All print expected output

- [ ] **Step 3: Verify package installs cleanly**

```bash
pip install -e ".[dev]"
python -c "from langtrans import Trans, action, Spec; print('OK')"
```
Expected: "OK"

- [ ] **Step 4: Final commit with any fixes**

Only if fixes were needed. Otherwise skip.

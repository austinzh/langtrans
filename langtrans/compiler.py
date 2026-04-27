from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeAlias

from langgraph.graph import END, START, StateGraph

from langtrans.nodes import (
    ActionNode,
    ConcurrentNode,
    LoopNode,
    Node,
    OptionalNode,
    SequentialNode,
    SwitchNode,
)

_Step: TypeAlias = tuple[Callable[..., Any], Callable[..., Any] | None]


class _Compiler:
    def __init__(self, graph: StateGraph[Any]) -> None:
        self._graph = graph
        self._counter = 0
        self._prefix = ""

    def _unique_id(self, base: str) -> str:
        name = f"{self._prefix}{base}" if self._prefix else base
        if name not in self._graph.nodes:
            return name
        self._counter += 1
        unique = f"{name}_{self._counter}"
        while unique in self._graph.nodes:
            self._counter += 1
            unique = f"{name}_{self._counter}"
        return unique

    def compile_node(self, node: Node) -> tuple[str, str]:
        node_name = getattr(node, "name", None)
        if node_name is not None and not isinstance(node, ActionNode):
            old_prefix = self._prefix
            self._prefix = f"{old_prefix}{node_name}."
            try:
                return self._compile_inner(node)
            finally:
                self._prefix = old_prefix
        return self._compile_inner(node)

    def _compile_inner(self, node: Node) -> tuple[str, str]:
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
        if isinstance(node, SwitchNode):
            return self._compile_switch(node)
        raise TypeError(f"Unknown node type: {type(node)}")

    # ------------------------------------------------------------------
    # ActionNode
    # ------------------------------------------------------------------
    def _compile_action(self, node: ActionNode) -> tuple[str, str]:
        assert node.func is not None, "ActionNode.func must be set"
        base = node.name or node.func.__name__
        node_id = self._unique_id(base)
        self._graph.add_node(node_id, node.func)
        return (node_id, node_id)

    # ------------------------------------------------------------------
    # SequentialNode
    # ------------------------------------------------------------------
    def _compile_sequential(self, node: SequentialNode) -> tuple[str, str]:
        if not node.children:
            noop_id = self._unique_id("seq_empty")
            self._graph.add_node(noop_id, lambda s: {})
            return (noop_id, noop_id)

        if self._has_rollbacks(node):
            return self._compile_sequential_with_rollback(node)

        pairs = [self.compile_node(child) for child in node.children]
        for i in range(len(pairs) - 1):
            self._graph.add_edge(pairs[i][1], pairs[i + 1][0])
        return (pairs[0][0], pairs[-1][1])

    # ------------------------------------------------------------------
    # ConcurrentNode
    # ------------------------------------------------------------------
    def _compile_concurrent(self, node: ConcurrentNode) -> tuple[str, str]:
        fork_id = self._unique_id("fork")
        join_id = self._unique_id("join")

        self._graph.add_node(fork_id, lambda s: {})
        self._graph.add_node(join_id, lambda s: {})

        for child in node.children:
            entry, exit_ = self.compile_node(child)
            self._graph.add_edge(fork_id, entry)
            self._graph.add_edge(exit_, join_id)

        return (fork_id, join_id)

    # ------------------------------------------------------------------
    # OptionalNode
    # ------------------------------------------------------------------
    def _compile_optional(self, node: OptionalNode) -> tuple[str, str]:
        decision_id = self._unique_id("decision")
        merge_id = self._unique_id("merge")

        self._graph.add_node(decision_id, lambda s: {})
        self._graph.add_node(merge_id, lambda s: {})

        then_entry, then_exit = self.compile_node(node.then_)  # type: ignore[arg-type]
        self._graph.add_edge(then_exit, merge_id)

        guard = node.guard

        if node.else_ is not None:
            else_entry, else_exit = self.compile_node(node.else_)
            self._graph.add_edge(else_exit, merge_id)

            def router_with_else(
                state: Any,
                _guard: Any = guard,
                _then: str = then_entry,
                _else: str = else_entry,
            ) -> str:
                result = _guard(state) if callable(_guard) else bool(_guard)
                return _then if result else _else

            self._graph.add_conditional_edges(
                decision_id, router_with_else, [then_entry, else_entry]
            )
        else:

            def router_no_else(
                state: Any,
                _guard: Any = guard,
                _then: str = then_entry,
                _merge: str = merge_id,
            ) -> str:
                result = _guard(state) if callable(_guard) else bool(_guard)
                return _then if result else _merge

            self._graph.add_conditional_edges(
                decision_id, router_no_else, [then_entry, merge_id]
            )

        return (decision_id, merge_id)

    # ------------------------------------------------------------------
    # LoopNode
    # ------------------------------------------------------------------
    def _compile_loop(self, node: LoopNode) -> tuple[str, str]:
        if node.times is not None:
            return self._compile_loop_fixed(node)
        if node.until is not None:
            return self._compile_loop_until(node)
        raise ValueError("LoopNode must have either 'times' or 'until'")

    def _compile_loop_fixed(self, node: LoopNode) -> tuple[str, str]:
        loop_id = self._counter
        self._counter += 1
        counter_key = f"_loop_{loop_id}_counter"
        times = node.times

        init_id = self._unique_id("loop_init")
        gate_id = self._unique_id("loop_gate")
        exit_id = self._unique_id("loop_exit")

        def init_fn(state: Any, _key: str = counter_key) -> dict[str, Any]:
            meta = dict(state.get("metadata", {}))
            lt = dict(meta.get("_langtrans", {}))
            lt[_key] = 0
            meta["_langtrans"] = lt
            return {"metadata": meta}

        def gate_fn(state: Any, _key: str = counter_key) -> dict[str, Any]:
            meta = dict(state.get("metadata", {}))
            lt = dict(meta.get("_langtrans", {}))
            lt[_key] = lt.get(_key, 0) + 1
            meta["_langtrans"] = lt
            return {"metadata": meta}

        self._graph.add_node(init_id, init_fn)
        self._graph.add_node(gate_id, gate_fn)
        self._graph.add_node(exit_id, lambda s: {})

        body_entry, body_exit = self.compile_node(node.body)  # type: ignore[arg-type]

        self._graph.add_edge(init_id, body_entry)
        self._graph.add_edge(body_exit, gate_id)

        def gate_router(
            state: Any,
            _key: str = counter_key,
            _times: int | None = times,
            _body: str = body_entry,
            _exit: str = exit_id,
        ) -> str:
            count = state.get("metadata", {}).get("_langtrans", {}).get(_key, 0)
            return _body if count < (_times or 0) else _exit

        self._graph.add_conditional_edges(gate_id, gate_router, [body_entry, exit_id])

        return (init_id, exit_id)

    def _compile_loop_until(self, node: LoopNode) -> tuple[str, str]:
        entry_id = self._unique_id("loop_entry")
        exit_id = self._unique_id("loop_exit")
        until_fn = node.until

        self._graph.add_node(entry_id, lambda s: {})
        self._graph.add_node(exit_id, lambda s: {})

        body_entry, body_exit = self.compile_node(node.body)  # type: ignore[arg-type]

        self._graph.add_edge(entry_id, body_entry)

        def body_router(
            state: Any,
            _until: Any = until_fn,
            _body: str = body_entry,
            _exit: str = exit_id,
        ) -> str:
            return _exit if _until(state) else _body

        self._graph.add_conditional_edges(body_exit, body_router, [body_entry, exit_id])

        return (entry_id, exit_id)

    # ------------------------------------------------------------------
    # SwitchNode
    # ------------------------------------------------------------------
    def _compile_switch(self, node: SwitchNode) -> tuple[str, str]:
        dispatch_id = self._unique_id("switch")
        merge_id = self._unique_id("switch_merge")

        self._graph.add_node(dispatch_id, lambda s: {})
        self._graph.add_node(merge_id, lambda s: {})

        case_entries: dict[str, str] = {}
        for case_key, case_node in node.cases.items():
            entry, exit_ = self.compile_node(case_node)
            self._graph.add_edge(exit_, merge_id)
            case_entries[case_key] = entry

        key_fn = node.key

        def router(
            state: Any,
            _key_fn: Any = key_fn,
            _cases: dict[str, str] = case_entries,
        ) -> str:
            return _cases[_key_fn(state)]

        self._graph.add_conditional_edges(
            dispatch_id, router, list(case_entries.values())
        )

        return (dispatch_id, merge_id)

    # ------------------------------------------------------------------
    # Sequential with rollback
    # ------------------------------------------------------------------
    def _has_rollbacks(self, node: SequentialNode) -> bool:
        return any(
            isinstance(child, ActionNode) and child.rollback is not None
            for child in node.children
        )

    def _compile_sequential_with_rollback(
        self, node: SequentialNode
    ) -> tuple[str, str]:
        rollback_id = self._unique_id("seq_rollback")

        steps: list[_Step] = []
        for child in node.children:
            if isinstance(child, ActionNode):
                steps.append((child.func, child.rollback))  # type: ignore[arg-type]
            else:
                raise TypeError(
                    "Rollback sequential only supports ActionNode children, "
                    f"got {type(child)}"
                )

        has_async = any(
            inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(rb)
            for fn, rb in steps
            if fn is not None
        )

        if has_async:

            async def async_rollback_runner(
                state: Any,
                _steps: list[_Step] = steps,
            ) -> dict[str, Any]:
                current_state = dict(state)
                completed_rollbacks: list[Callable[..., Any]] = []

                for func, rollback in _steps:
                    try:
                        updates = (
                            await func(current_state)
                            if inspect.iscoroutinefunction(func)
                            else func(current_state)
                        )
                        if updates:
                            for k, v in updates.items():
                                current_state[k] = v
                        if rollback is not None:
                            completed_rollbacks.append(rollback)
                    except Exception:
                        for rb in reversed(completed_rollbacks):
                            try:
                                rb_updates = (
                                    await rb(current_state)
                                    if inspect.iscoroutinefunction(rb)
                                    else rb(current_state)
                                )
                                if rb_updates:
                                    for k, v in rb_updates.items():
                                        current_state[k] = v
                            except Exception:
                                pass
                        raise

                result: dict[str, Any] = {}
                for k, v in current_state.items():
                    if k not in state or state[k] != v:
                        result[k] = v
                return result

            self._graph.add_node(rollback_id, async_rollback_runner)
        else:

            def sync_rollback_runner(
                state: Any,
                _steps: list[_Step] = steps,
            ) -> dict[str, Any]:
                current_state = dict(state)
                completed_rollbacks: list[Callable[..., Any]] = []

                for func, rollback in _steps:
                    try:
                        updates = func(current_state)
                        if updates:
                            for k, v in updates.items():
                                current_state[k] = v
                        if rollback is not None:
                            completed_rollbacks.append(rollback)
                    except Exception:
                        for rb in reversed(completed_rollbacks):
                            try:
                                rb_updates = rb(current_state)
                                if rb_updates:
                                    for k, v in rb_updates.items():
                                        current_state[k] = v
                            except Exception:
                                pass
                        raise

                result: dict[str, Any] = {}
                for k, v in current_state.items():
                    if k not in state or state[k] != v:
                        result[k] = v
                return result

            self._graph.add_node(rollback_id, sync_rollback_runner)
        return (rollback_id, rollback_id)


def compile_graph(tree: Node, *, state_schema: type[Any], **kwargs: Any) -> Any:
    graph: StateGraph[Any] = StateGraph(state_schema)
    compiler = _Compiler(graph)
    entry, exit_ = compiler.compile_node(tree)
    graph.add_edge(START, entry)
    graph.add_edge(exit_, END)
    return graph.compile(**kwargs)

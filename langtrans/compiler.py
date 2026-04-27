from __future__ import annotations

import time
from typing import Any, Callable

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
    SwitchNode,
)
from langtrans.spec import Spec


class _Compiler:
    def __init__(self, graph: StateGraph) -> None:
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
        if isinstance(node, SwitchNode):
            return self._compile_switch(node)
        raise TypeError(f"Unknown node type: {type(node)}")

    # ------------------------------------------------------------------
    # 1. ActionNode
    # ------------------------------------------------------------------
    def _compile_action(self, node: ActionNode) -> tuple[str, str]:
        base = node.name or node.func.__name__
        node_id = self._unique_id(base)
        self._graph.add_node(node_id, node.func)
        return (node_id, node_id)

    # ------------------------------------------------------------------
    # 2. SequentialNode
    # ------------------------------------------------------------------
    def _compile_sequential(self, node: SequentialNode) -> tuple[str, str]:
        if not node.children:
            noop_id = self._unique_id("seq_empty")
            self._graph.add_node(noop_id, lambda s: {})
            return (noop_id, noop_id)

        # Check if any children have rollbacks — if so, use rollback-aware compilation
        if self._has_rollbacks(node):
            return self._compile_sequential_with_rollback(node)

        pairs = [self.compile_node(child) for child in node.children]
        for i in range(len(pairs) - 1):
            self._graph.add_edge(pairs[i][1], pairs[i + 1][0])
        return (pairs[0][0], pairs[-1][1])

    # ------------------------------------------------------------------
    # 3. ConcurrentNode
    # ------------------------------------------------------------------
    def _compile_concurrent(self, node: ConcurrentNode) -> tuple[str, str]:
        # Collect all leaf action functions from the concurrent children.
        # We run them sequentially in a single LangGraph node to avoid
        # conflicts on non-annotated state channels (like 'metadata: dict').
        funcs = self._collect_action_funcs(node.children)

        concurrent_id = self._unique_id("concurrent")

        def concurrent_runner(state, _funcs=funcs):
            current_state = dict(state)
            for fn in _funcs:
                updates = fn(current_state)
                if updates:
                    for k, v in updates.items():
                        current_state[k] = v
            # Return the diff from original state
            result = {}
            for k, v in current_state.items():
                if k not in state or state[k] != v:
                    result[k] = v
            return result

        self._graph.add_node(concurrent_id, concurrent_runner)
        return (concurrent_id, concurrent_id)

    def _collect_action_funcs(self, children: list[Node]) -> list[Callable]:
        """Recursively collect callable functions from a list of nodes."""
        funcs = []
        for child in children:
            if isinstance(child, ActionNode):
                funcs.append(child.func)
            elif isinstance(child, SequentialNode):
                for sub in child.children:
                    funcs.extend(self._collect_action_funcs([sub]))
            elif isinstance(child, ConcurrentNode):
                funcs.extend(self._collect_action_funcs(child.children))
            else:
                raise TypeError(
                    f"ConcurrentNode children must be ActionNode or "
                    f"SequentialNode, got {type(child)}"
                )
        return funcs

    # ------------------------------------------------------------------
    # 4. OptionalNode
    # ------------------------------------------------------------------
    def _compile_optional(self, node: OptionalNode) -> tuple[str, str]:
        decision_id = self._unique_id("decision")
        merge_id = self._unique_id("merge")

        self._graph.add_node(decision_id, lambda s: {})
        self._graph.add_node(merge_id, lambda s: {})

        then_entry, then_exit = self.compile_node(node.then_)
        self._graph.add_edge(then_exit, merge_id)

        guard = node.guard

        if node.else_ is not None:
            else_entry, else_exit = self.compile_node(node.else_)
            self._graph.add_edge(else_exit, merge_id)

            def router(state, _guard=guard, _then=then_entry, _else=else_entry):
                if callable(_guard):
                    result = _guard(state)
                else:
                    result = bool(_guard)
                return _then if result else _else

            self._graph.add_conditional_edges(
                decision_id, router, [then_entry, else_entry]
            )
        else:
            def router(state, _guard=guard, _then=then_entry, _merge=merge_id):
                if callable(_guard):
                    result = _guard(state)
                else:
                    result = bool(_guard)
                return _then if result else _merge

            self._graph.add_conditional_edges(
                decision_id, router, [then_entry, merge_id]
            )

        return (decision_id, merge_id)

    # ------------------------------------------------------------------
    # 5. LoopNode
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

        def init_fn(state, _key=counter_key):
            meta = dict(state.get("metadata", {}))
            lt = dict(meta.get("_langtrans", {}))
            lt[_key] = 0
            meta["_langtrans"] = lt
            return {"metadata": meta}

        def gate_fn(state, _key=counter_key):
            meta = dict(state.get("metadata", {}))
            lt = dict(meta.get("_langtrans", {}))
            lt[_key] = lt.get(_key, 0) + 1
            meta["_langtrans"] = lt
            return {"metadata": meta}

        self._graph.add_node(init_id, init_fn)
        self._graph.add_node(gate_id, gate_fn)
        self._graph.add_node(exit_id, lambda s: {})

        body_entry, body_exit = self.compile_node(node.body)

        # init -> body_entry
        self._graph.add_edge(init_id, body_entry)
        # body_exit -> gate
        self._graph.add_edge(body_exit, gate_id)

        # gate -> conditional: if counter < times -> body_entry, else -> exit
        def gate_router(state, _key=counter_key, _times=times,
                        _body=body_entry, _exit=exit_id):
            meta = state.get("metadata", {})
            lt = meta.get("_langtrans", {})
            count = lt.get(_key, 0)
            return _body if count < _times else _exit

        self._graph.add_conditional_edges(gate_id, gate_router, [body_entry, exit_id])

        return (init_id, exit_id)

    def _compile_loop_until(self, node: LoopNode) -> tuple[str, str]:
        entry_id = self._unique_id("loop_entry")
        exit_id = self._unique_id("loop_exit")
        until_fn = node.until

        self._graph.add_node(entry_id, lambda s: {})
        self._graph.add_node(exit_id, lambda s: {})

        body_entry, body_exit = self.compile_node(node.body)

        # entry -> body_entry
        self._graph.add_edge(entry_id, body_entry)

        # body_exit -> conditional: if until(state) -> exit, else -> body_entry
        def body_router(state, _until=until_fn, _body=body_entry, _exit=exit_id):
            return _exit if _until(state) else _body

        self._graph.add_conditional_edges(body_exit, body_router, [body_entry, exit_id])

        return (entry_id, exit_id)

    # ------------------------------------------------------------------
    # 6. RetryNode
    # ------------------------------------------------------------------
    def _compile_retry(self, node: RetryNode) -> tuple[str, str]:
        if not isinstance(node.body, ActionNode):
            raise TypeError("RetryNode body must be an ActionNode")

        body_func = node.body.func
        max_attempts = node.max_attempts
        delay = node.delay
        base_name = node.body.name or body_func.__name__

        retry_id = self._unique_id(f"retry_{base_name}")

        def retry_runner(state, _func=body_func, _max=max_attempts, _delay=delay):
            last_exc = None
            for attempt in range(_max):
                try:
                    return _func(state)
                except Exception as exc:
                    last_exc = exc
                    if attempt < _max - 1 and _delay > 0:
                        time.sleep(_delay)
            raise last_exc  # type: ignore[misc]

        self._graph.add_node(retry_id, retry_runner)
        return (retry_id, retry_id)

    # ------------------------------------------------------------------
    # 7. ProcedureNode
    # ------------------------------------------------------------------
    def _compile_procedure(self, node: ProcedureNode) -> tuple[str, str]:
        old_prefix = self._prefix
        self._prefix = f"{old_prefix}{node.name}."
        try:
            entry, exit_ = self.compile_node(node.body)
        finally:
            self._prefix = old_prefix
        return (entry, exit_)

    # ------------------------------------------------------------------
    # 8. SwitchNode
    # ------------------------------------------------------------------
    def _compile_switch(self, node: SwitchNode) -> tuple[str, str]:
        dispatch_id = self._unique_id("switch")
        merge_id = self._unique_id("switch_merge")

        self._graph.add_node(dispatch_id, lambda s: {})
        self._graph.add_node(merge_id, lambda s: {})

        case_entries = {}
        for case_key, case_node in node.cases.items():
            entry, exit_ = self.compile_node(case_node)
            self._graph.add_edge(exit_, merge_id)
            case_entries[case_key] = entry

        key_fn = node.key

        def router(state, _key_fn=key_fn, _cases=case_entries):
            result = _key_fn(state)
            return _cases[result]

        self._graph.add_conditional_edges(
            dispatch_id, router, list(case_entries.values())
        )

        return (dispatch_id, merge_id)

    # ------------------------------------------------------------------
    # 9. Sequential with rollback
    # ------------------------------------------------------------------
    def _has_rollbacks(self, node: SequentialNode) -> bool:
        return any(
            isinstance(child, ActionNode) and child.rollback is not None
            for child in node.children
        )

    def _compile_sequential_with_rollback(self, node: SequentialNode) -> tuple[str, str]:
        rollback_id = self._unique_id("seq_rollback")

        # Gather ordered steps: (func, rollback_or_None)
        steps: list[tuple[Callable, Callable | None]] = []
        for child in node.children:
            if isinstance(child, ActionNode):
                steps.append((child.func, child.rollback))
            else:
                raise TypeError(
                    "Rollback sequential only supports ActionNode children, "
                    f"got {type(child)}"
                )

        def rollback_runner(state, _steps=steps):
            current_state = dict(state)
            completed_rollbacks: list[Callable] = []

            for func, rollback in _steps:
                try:
                    updates = func(current_state)
                    if updates:
                        for k, v in updates.items():
                            current_state[k] = v
                    if rollback is not None:
                        completed_rollbacks.append(rollback)
                except Exception:
                    # Run rollbacks in reverse
                    for rb in reversed(completed_rollbacks):
                        try:
                            rb_updates = rb(current_state)
                            if rb_updates:
                                for k, v in rb_updates.items():
                                    current_state[k] = v
                        except Exception:
                            pass  # swallow rollback errors
                    raise

            # Compute diff
            result = {}
            for k, v in current_state.items():
                if k not in state or state[k] != v:
                    result[k] = v
            return result

        self._graph.add_node(rollback_id, rollback_runner)
        return (rollback_id, rollback_id)


def compile_graph(tree: Node, *, state_schema, **kwargs) -> Any:
    graph = StateGraph(state_schema)
    compiler = _Compiler(graph)
    entry, exit_ = compiler.compile_node(tree)
    graph.add_edge(START, entry)
    graph.add_edge(exit_, END)
    return graph.compile(**kwargs)

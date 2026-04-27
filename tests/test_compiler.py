"""Tests for langtrans.compiler — each primitive is tested in order."""

import operator
from typing import Annotated, TypedDict

import pytest

from langtrans.builder import Proc, Trans
from langtrans.spec import Spec

# ── State schema ──────────────────────────────────────────────────────


class State(TypedDict):
    messages: Annotated[list, operator.add]
    metadata: dict


# ── Reusable test actions ─────────────────────────────────────────────


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


INIT_STATE = {"messages": [], "metadata": {}}


# =====================================================================
# 1. ActionNode — single LangGraph node
# =====================================================================


class TestActionNode:
    def test_single_action(self):
        app = Trans(state_schema=State).sequential(action_a).compile()
        result = app.invoke(INIT_STATE)
        assert result["metadata"]["_calls"] == ["a"]

    def test_action_with_explicit_name(self):
        from langtrans.nodes import ActionNode

        node = ActionNode(func=action_a, name="custom_name")
        app = Trans(state_schema=State).sequential(node).compile()
        result = app.invoke(INIT_STATE)
        assert result["metadata"]["_calls"] == ["a"]


# =====================================================================
# 2. SequentialNode — chain of edges
# =====================================================================


class TestSequentialNode:
    def test_sequential_three_actions(self):
        app = (
            Trans(state_schema=State).sequential(action_a, action_b, action_c).compile()
        )
        result = app.invoke(INIT_STATE)
        assert result["metadata"]["_calls"] == ["a", "b", "c"]

    def test_sequential_two_actions(self):
        app = Trans(state_schema=State).sequential(action_a, action_b).compile()
        result = app.invoke(INIT_STATE)
        assert result["metadata"]["_calls"] == ["a", "b"]

    def test_sequential_single_action(self):
        app = Trans(state_schema=State).sequential(action_a).compile()
        result = app.invoke(INIT_STATE)
        assert result["metadata"]["_calls"] == ["a"]


# =====================================================================
# 3. ConcurrentNode — fan-out / fan-in
# =====================================================================


def msg_action(name):
    def fn(state):
        return {"messages": [name]}

    fn.__name__ = name
    fn.__qualname__ = name
    return fn


class TestConcurrentNode:
    def test_concurrent_three_actions(self):
        app = (
            Trans(state_schema=State)
            .concurrent(msg_action("a"), msg_action("b"), msg_action("c"))
            .compile()
        )
        result = app.invoke(INIT_STATE)
        assert set(result["messages"]) == {"a", "b", "c"}

    def test_concurrent_two_actions(self):
        app = (
            Trans(state_schema=State)
            .concurrent(msg_action("a"), msg_action("b"))
            .compile()
        )
        result = app.invoke(INIT_STATE)
        assert set(result["messages"]) == {"a", "b"}


# =====================================================================
# 4. OptionalNode — conditional routing
# =====================================================================


def always_true(state) -> bool:
    return True


def always_false(state) -> bool:
    return False


class TestOptionalNode:
    def test_guard_true_takes_then_branch(self):
        app = (
            Trans(state_schema=State)
            .optional(always_true, then_=action_a, else_=action_b)
            .compile()
        )
        result = app.invoke(INIT_STATE)
        assert result["metadata"]["_calls"] == ["a"]

    def test_guard_false_takes_else_branch(self):
        app = (
            Trans(state_schema=State)
            .optional(always_false, then_=action_a, else_=action_b)
            .compile()
        )
        result = app.invoke(INIT_STATE)
        assert result["metadata"]["_calls"] == ["b"]

    def test_guard_false_no_else_skips(self):
        app = Trans(state_schema=State).optional(always_false, then_=action_a).compile()
        result = app.invoke(INIT_STATE)
        # No branch taken, _calls should not exist or be empty
        assert (
            result["metadata"].get("_calls") is None
            or result["metadata"]["_calls"] == []
        )

    def test_spec_guard(self):
        spec = Spec(lambda s: True)
        app = (
            Trans(state_schema=State)
            .optional(spec, then_=action_a, else_=action_b)
            .compile()
        )
        result = app.invoke(INIT_STATE)
        assert result["metadata"]["_calls"] == ["a"]

    def test_spec_guard_false(self):
        spec = Spec(lambda s: False)
        app = (
            Trans(state_schema=State)
            .optional(spec, then_=action_a, else_=action_b)
            .compile()
        )
        result = app.invoke(INIT_STATE)
        assert result["metadata"]["_calls"] == ["b"]

    def test_dynamic_guard_based_on_state(self):
        """Guard reads state set by a prior step."""

        def check_flag(state):
            return state.get("metadata", {}).get("flag") is True

        def set_flag(state):
            meta = dict(state.get("metadata", {}))
            meta["flag"] = True
            return {"metadata": meta}

        app = (
            Trans(state_schema=State)
            .sequential(set_flag)
            .optional(check_flag, then_=action_a, else_=action_b)
            .compile()
        )
        result = app.invoke(INIT_STATE)
        assert result["metadata"]["_calls"] == ["a"]


# =====================================================================
# 5. LoopNode — cycle in graph
# =====================================================================


_loop_counter = 0


def counting_action(state):
    meta = dict(state.get("metadata", {}))
    count = meta.get("_count", 0) + 1
    meta["_count"] = count
    return {"metadata": meta}


class TestLoopNode:
    def test_loop_fixed_times(self):
        app = Trans(state_schema=State).loop(body=counting_action, times=3).compile()
        result = app.invoke(INIT_STATE)
        assert result["metadata"]["_count"] == 3

    def test_loop_fixed_times_one(self):
        app = Trans(state_schema=State).loop(body=counting_action, times=1).compile()
        result = app.invoke(INIT_STATE)
        assert result["metadata"]["_count"] == 1

    def test_loop_until_condition(self):
        app = (
            Trans(state_schema=State)
            .loop(
                body=counting_action,
                until=lambda s: s.get("metadata", {}).get("_count", 0) >= 2,
            )
            .compile()
        )
        result = app.invoke(INIT_STATE)
        assert result["metadata"]["_count"] == 2

    def test_loop_until_larger(self):
        app = (
            Trans(state_schema=State)
            .loop(
                body=counting_action,
                until=lambda s: s.get("metadata", {}).get("_count", 0) >= 5,
            )
            .compile()
        )
        result = app.invoke(INIT_STATE)
        assert result["metadata"]["_count"] == 5


# =====================================================================
# 6. Named nodes — name-prefixed subgraphs
# =====================================================================


class TestNamedNodes:
    def test_procedure_runs_body(self):
        app = (
            Trans(state_schema=State)
            .sequential(Proc("sub_flow").sequential(action_a, action_b))
            .compile()
        )
        result = app.invoke(INIT_STATE)
        assert result["metadata"]["_calls"] == ["a", "b"]

    def test_procedure_with_surrounding_steps(self):
        app = (
            Trans(state_schema=State)
            .sequential(
                action_a,
                Proc("middle").sequential(action_b),
                action_c,
            )
            .compile()
        )
        result = app.invoke(INIT_STATE)
        assert result["metadata"]["_calls"] == ["a", "b", "c"]

    def test_two_procedures_no_collision(self):
        app = (
            Trans(state_schema=State)
            .sequential(
                Proc("first").sequential(action_a),
                Proc("second").sequential(action_b),
            )
            .compile()
        )
        result = app.invoke(INIT_STATE)
        assert result["metadata"]["_calls"] == ["a", "b"]


# =====================================================================
# 8. Rollback / Compensation in Sequential
# =====================================================================


class TestRollback:
    def test_rollback_on_failure(self):
        """step_a/b with rollback, step_c fails, rollback_b then rollback_a."""
        rollback_order = []

        def step_a(state):
            meta = dict(state.get("metadata", {}))
            calls = list(meta.get("_calls", []))
            calls.append("a")
            meta["_calls"] = calls
            return {"metadata": meta}

        def rollback_a(state):
            rollback_order.append("rollback_a")
            return {}

        def step_b(state):
            meta = dict(state.get("metadata", {}))
            calls = list(meta.get("_calls", []))
            calls.append("b")
            meta["_calls"] = calls
            return {"metadata": meta}

        def rollback_b(state):
            rollback_order.append("rollback_b")
            return {}

        def step_c_fail(state):
            raise RuntimeError("step_c failed")

        from langtrans.nodes import ActionNode

        node_a = ActionNode(func=step_a, rollback=rollback_a, name="step_a")
        node_b = ActionNode(func=step_b, rollback=rollback_b, name="step_b")
        node_c = ActionNode(func=step_c_fail, name="step_c_fail")

        app = Trans(state_schema=State).sequential(node_a, node_b, node_c).compile()

        with pytest.raises(RuntimeError, match="step_c failed"):
            app.invoke(INIT_STATE)

        assert rollback_order == ["rollback_b", "rollback_a"]

    def test_all_succeed_no_rollbacks(self):
        rollback_called = {"called": False}

        def rollback_fn(state):
            rollback_called["called"] = True
            return {}

        from langtrans.nodes import ActionNode

        node_a = ActionNode(func=action_a, rollback=rollback_fn, name="ra")
        node_b = ActionNode(func=action_b, rollback=rollback_fn, name="rb")

        app = Trans(state_schema=State).sequential(node_a, node_b).compile()

        result = app.invoke(INIT_STATE)
        assert result["metadata"]["_calls"] == ["a", "b"]
        assert rollback_called["called"] is False

import operator
from typing import Annotated, TypedDict

from langtrans import Trans, Proc, SwitchNode
from langtrans.builder import _to_node
from langtrans.nodes import ActionNode


class State(TypedDict):
    messages: Annotated[list, operator.add]
    metadata: dict


def set_state(name):
    def fn(state):
        meta = dict(state.get("metadata", {}))
        calls = list(meta.get("_calls", []))
        calls.append(name)
        meta["_calls"] = calls
        meta["_state"] = name
        return {"metadata": meta}
    fn.__name__ = name
    fn.__qualname__ = name
    return fn


action_a = set_state("a")
action_b = set_state("b")
action_c = set_state("c")


class TestSwitchNodeModel:
    def test_switch_node_creation(self):
        node = SwitchNode(
            key=lambda s: "a",
            cases={"a": ActionNode(func=action_a), "b": ActionNode(func=action_b)},
        )
        assert len(node.cases) == 2
        assert "a" in node.cases
        assert "b" in node.cases


class TestSwitchBuilder:
    def test_switch_builds_correct_node(self):
        tree = (
            Proc()
            .switch(
                key=lambda s: "a",
                cases={"a": action_a, "b": action_b},
            )
            .build()
        )
        assert isinstance(tree, SwitchNode)
        assert len(tree.cases) == 2
        assert isinstance(tree.cases["a"], ActionNode)

    def test_switch_with_nested_trans(self):
        tree = (
            Proc()
            .switch(
                key=lambda s: "x",
                cases={
                    "x": Proc().sequential(action_a, action_b),
                    "y": action_c,
                },
            )
            .build()
        )
        assert isinstance(tree, SwitchNode)


class TestSwitchCompiler:
    def test_switch_routes_to_correct_case(self):
        app = (
            Trans(state_schema=State)
            .switch(
                key=lambda s: s.get("metadata", {}).get("_state", "a"),
                cases={
                    "a": action_a,
                    "b": action_b,
                    "c": action_c,
                },
            )
            .compile()
        )
        result = app.invoke({"messages": [], "metadata": {"_state": "b"}})
        assert result["metadata"]["_calls"] == ["b"]

    def test_switch_default_case(self):
        app = (
            Trans(state_schema=State)
            .switch(
                key=lambda s: s.get("metadata", {}).get("_state", "a"),
                cases={
                    "a": action_a,
                    "b": action_b,
                },
            )
            .compile()
        )
        result = app.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["a"]

    def test_switch_with_sequential_case(self):
        app = (
            Trans(state_schema=State)
            .switch(
                key=lambda s: "x",
                cases={
                    "x": Proc().sequential(action_a, action_b),
                    "y": action_c,
                },
            )
            .compile()
        )
        result = app.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["a", "b"]


class TestSwitchStateMachine:
    def test_full_state_machine_with_loop(self):
        """A → B → A → C → done: classic state machine via loop + switch."""

        call_count = {"n": 0}

        def state_a(state):
            call_count["n"] += 1
            meta = dict(state.get("metadata", {}))
            calls = list(meta.get("_calls", []))
            calls.append("a")
            meta["_calls"] = calls
            # First time go to B, second time go to C
            meta["_state"] = "b" if call_count["n"] == 1 else "c"
            return {"metadata": meta}

        state_a.__name__ = "state_a"

        def state_b(state):
            meta = dict(state.get("metadata", {}))
            calls = list(meta.get("_calls", []))
            calls.append("b")
            meta["_calls"] = calls
            meta["_state"] = "a"
            return {"metadata": meta}

        state_b.__name__ = "state_b"

        def state_c(state):
            meta = dict(state.get("metadata", {}))
            calls = list(meta.get("_calls", []))
            calls.append("c")
            meta["_calls"] = calls
            meta["_state"] = "done"
            return {"metadata": meta}

        state_c.__name__ = "state_c"

        app = (
            Trans(state_schema=State)
            .loop(
                until=lambda s: s.get("metadata", {}).get("_state") == "done",
                body=Proc().switch(
                    key=lambda s: s.get("metadata", {}).get("_state", "a"),
                    cases={
                        "a": state_a,
                        "b": state_b,
                        "c": state_c,
                    },
                ),
            )
            .compile()
        )

        result = app.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["a", "b", "a", "c"]
        assert result["metadata"]["_state"] == "done"

    def test_three_state_cycle(self):
        """A → B → C → A → B → C → done (cycle 2 times)."""

        cycle_count = {"n": 0}

        def go_a(state):
            meta = dict(state.get("metadata", {}))
            calls = list(meta.get("_calls", []))
            calls.append("a")
            meta["_calls"] = calls
            meta["_state"] = "b"
            return {"metadata": meta}

        go_a.__name__ = "go_a"

        def go_b(state):
            meta = dict(state.get("metadata", {}))
            calls = list(meta.get("_calls", []))
            calls.append("b")
            meta["_calls"] = calls
            meta["_state"] = "c"
            return {"metadata": meta}

        go_b.__name__ = "go_b"

        def go_c(state):
            cycle_count["n"] += 1
            meta = dict(state.get("metadata", {}))
            calls = list(meta.get("_calls", []))
            calls.append("c")
            meta["_calls"] = calls
            meta["_state"] = "done" if cycle_count["n"] >= 2 else "a"
            return {"metadata": meta}

        go_c.__name__ = "go_c"

        app = (
            Trans(state_schema=State)
            .loop(
                until=lambda s: s.get("metadata", {}).get("_state") == "done",
                body=Proc().switch(
                    key=lambda s: s.get("metadata", {}).get("_state", "a"),
                    cases={"a": go_a, "b": go_b, "c": go_c},
                ),
            )
            .compile()
        )

        result = app.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["a", "b", "c", "a", "b", "c"]
        assert result["metadata"]["_state"] == "done"

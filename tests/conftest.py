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

import operator
from typing import Annotated, TypedDict

import pytest

from langtrans import Trans, action


class State(TypedDict):
    messages: Annotated[list, operator.add]
    metadata: dict


async def async_action_a(state):
    meta = dict(state.get("metadata", {}))
    calls = list(meta.get("_calls", []))
    calls.append("a")
    meta["_calls"] = calls
    return {"metadata": meta}


async def async_action_b(state):
    meta = dict(state.get("metadata", {}))
    calls = list(meta.get("_calls", []))
    calls.append("b")
    meta["_calls"] = calls
    return {"metadata": meta}


async def async_action_c(state):
    meta = dict(state.get("metadata", {}))
    calls = list(meta.get("_calls", []))
    calls.append("c")
    meta["_calls"] = calls
    return {"metadata": meta}


class TestAsyncSequential:
    @pytest.mark.asyncio
    async def test_async_sequential(self):
        app = (
            Trans(state_schema=State)
            .sequential(async_action_a, async_action_b, async_action_c)
            .compile()
        )
        result = await app.ainvoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["a", "b", "c"]


class TestAsyncConcurrent:
    @staticmethod
    async def msg_a(state):
        return {"messages": ["a"]}

    @staticmethod
    async def msg_b(state):
        return {"messages": ["b"]}

    @staticmethod
    async def msg_c(state):
        return {"messages": ["c"]}

    @staticmethod
    async def async_msg(state):
        return {"messages": ["async"]}

    @staticmethod
    def sync_msg(state):
        return {"messages": ["sync"]}

    @pytest.mark.asyncio
    async def test_async_concurrent(self):
        app = (
            Trans(state_schema=State)
            .concurrent(self.msg_a, self.msg_b, self.msg_c)
            .compile()
        )
        result = await app.ainvoke({"messages": [], "metadata": {}})
        assert set(result["messages"]) == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_mixed_sync_async_concurrent(self):
        app = (
            Trans(state_schema=State)
            .concurrent(self.async_msg, self.sync_msg)
            .compile()
        )
        result = await app.ainvoke({"messages": [], "metadata": {}})
        assert "async" in result["messages"]
        assert "sync" in result["messages"]


class TestAsyncRollback:
    @staticmethod
    async def step_fail(state):
        raise ValueError("boom")

    @pytest.mark.asyncio
    async def test_async_rollback_on_failure(self):
        rollback_log = []

        async def async_rollback_a(state):
            rollback_log.append("rollback_a")
            return {}

        @action(rollback=async_rollback_a)
        async def step_a(state):
            meta = dict(state.get("metadata", {}))
            meta["_calls"] = ["a"]
            return {"metadata": meta}

        app = Trans(state_schema=State).sequential(step_a, self.step_fail).compile()

        with pytest.raises(ValueError, match="boom"):
            await app.ainvoke({"messages": [], "metadata": {}})

        assert "rollback_a" in rollback_log


class TestAsyncLoop:
    @staticmethod
    async def counter(state):
        meta = dict(state.get("metadata", {}))
        meta["count"] = meta.get("count", 0) + 1
        return {"metadata": meta}

    @pytest.mark.asyncio
    async def test_async_loop_until(self):
        app = (
            Trans(state_schema=State)
            .loop(
                body=self.counter,
                until=lambda s: s.get("metadata", {}).get("count", 0) >= 3,
            )
            .compile()
        )
        result = await app.ainvoke({"messages": [], "metadata": {}})
        assert result["metadata"]["count"] == 3

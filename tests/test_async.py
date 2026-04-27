import asyncio
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
    @pytest.mark.asyncio
    async def test_async_concurrent(self):
        async def msg_a(state):
            return {"messages": ["a"]}

        async def msg_b(state):
            return {"messages": ["b"]}

        async def msg_c(state):
            return {"messages": ["c"]}

        msg_a.__name__ = "msg_a"
        msg_b.__name__ = "msg_b"
        msg_c.__name__ = "msg_c"

        app = (
            Trans(state_schema=State)
            .concurrent(msg_a, msg_b, msg_c)
            .compile()
        )
        result = await app.ainvoke({"messages": [], "metadata": {}})
        assert set(result["messages"]) == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_mixed_sync_async_concurrent(self):
        async def async_msg(state):
            return {"messages": ["async"]}

        def sync_msg(state):
            return {"messages": ["sync"]}

        async_msg.__name__ = "async_msg"
        sync_msg.__name__ = "sync_msg"

        app = (
            Trans(state_schema=State)
            .concurrent(async_msg, sync_msg)
            .compile()
        )
        result = await app.ainvoke({"messages": [], "metadata": {}})
        assert "async" in result["messages"]
        assert "sync" in result["messages"]


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_async_retry_succeeds(self):
        call_count = {"n": 0}

        async def flaky(state):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ValueError("not yet")
            meta = dict(state.get("metadata", {}))
            meta["_calls"] = ["flaky_ok"]
            return {"metadata": meta}

        flaky.__name__ = "flaky"

        app = (
            Trans(state_schema=State)
            .retry(flaky, max_attempts=5, delay=0.0)
            .compile()
        )
        result = await app.ainvoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["flaky_ok"]
        assert call_count["n"] == 3


class TestAsyncRollback:
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

        async def step_fail(state):
            raise ValueError("boom")

        step_fail.__name__ = "step_fail"

        app = (
            Trans(state_schema=State)
            .sequential(step_a, step_fail)
            .compile()
        )

        with pytest.raises(ValueError, match="boom"):
            await app.ainvoke({"messages": [], "metadata": {}})

        assert "rollback_a" in rollback_log


class TestAsyncLoop:
    @pytest.mark.asyncio
    async def test_async_loop_until(self):
        async def counter(state):
            meta = dict(state.get("metadata", {}))
            meta["count"] = meta.get("count", 0) + 1
            return {"metadata": meta}

        counter.__name__ = "counter"

        app = (
            Trans(state_schema=State)
            .loop(
                body=counter,
                until=lambda s: s.get("metadata", {}).get("count", 0) >= 3,
            )
            .compile()
        )
        result = await app.ainvoke({"messages": [], "metadata": {}})
        assert result["metadata"]["count"] == 3

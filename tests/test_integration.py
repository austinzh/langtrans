import operator
from typing import Annotated, TypedDict

from langtrans.builder import Trans, action
from langtrans.spec import Spec


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
        def msg_search_web(state):
            return {"messages": ["search_web"]}

        def msg_search_db(state):
            return {"messages": ["search_db"]}

        msg_search_web.__name__ = "msg_search_web"
        msg_search_db.__name__ = "msg_search_db"

        has_results = lambda s: len(s.get("messages", [])) >= 2

        app = (
            Trans(state_schema=State)
            .concurrent(msg_search_web, msg_search_db)
            .optional(has_results, then_=respond, else_=clarify)
            .compile()
        )
        result = app.invoke({"messages": [], "metadata": {}})
        assert "search_web" in result["messages"]
        assert "search_db" in result["messages"]


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

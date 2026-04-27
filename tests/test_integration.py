import operator
from typing import Annotated, TypedDict

from langtrans.builder import Proc, Trans
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
        app = Trans(state_schema=State).sequential(fetch, llm, respond).compile()
        result = app.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["_calls"] == ["fetch", "llm", "respond"]


class TestParallelWithConditional:
    @staticmethod
    def msg_search_web(state):
        return {"messages": ["search_web"]}

    @staticmethod
    def msg_search_db(state):
        return {"messages": ["search_db"]}

    def test_concurrent_then_conditional(self):
        def has_results(s):
            return len(s.get("messages", [])) >= 2

        app = (
            Trans(state_schema=State)
            .concurrent(self.msg_search_web, self.msg_search_db)
            .optional(has_results, then_=respond, else_=clarify)
            .compile()
        )
        result = app.invoke({"messages": [], "metadata": {}})
        assert "search_web" in result["messages"]
        assert "search_db" in result["messages"]


class TestLoopWithNesting:
    @staticmethod
    def increment(state):
        meta = dict(state.get("metadata", {}))
        count = meta.get("count", 0) + 1
        meta["count"] = count
        calls = list(meta.get("_calls", []))
        calls.append(f"inc_{count}")
        meta["_calls"] = calls
        return {"metadata": meta}

    @staticmethod
    def is_even(state):
        return state.get("metadata", {}).get("count", 0) % 2 == 0

    def test_loop_with_nested_optional(self):
        mark_even = append_call("even")
        mark_odd = append_call("odd")

        app = (
            Trans(state_schema=State)
            .loop(
                body=Proc().sequential(
                    self.increment,
                    Proc().optional(self.is_even, then_=mark_even, else_=mark_odd),
                ),
                times=3,
            )
            .compile()
        )
        result = app.invoke({"messages": [], "metadata": {}})
        assert result["metadata"]["count"] == 3


class TestProcedureComposition:
    def test_procedure_as_reusable_unit(self):
        app = (
            Trans(state_schema=State)
            .sequential(
                Proc("fetch_pipe").sequential(fetch, llm),
                respond,
            )
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

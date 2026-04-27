"""Example 1: Sequential Agent — fetch data, call LLM, respond."""
import operator
from typing import Annotated, TypedDict

from langtrans.builder import Trans


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
    .sequential(
        fetch_data, 
        call_llm, 
        format_response
    ).compile()
)

if __name__ == "__main__":
    result = app.invoke({"messages": [], "metadata": {}})
    print(result["metadata"]["final"])

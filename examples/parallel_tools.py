"""Example 2: Parallel tool calls with conditional routing.

Parallel branches write to `messages` (which has operator.add reducer)
so LangGraph can merge results from concurrent nodes. The conditional
routing then checks the accumulated messages.
"""
import operator
from typing import Annotated, TypedDict

from langtrans.builder import Trans


class State(TypedDict):
    messages: Annotated[list, operator.add]
    metadata: dict


def search_web(state):
    return {"messages": [{"source": "web", "result": "web result"}]}


def search_db(state):
    return {"messages": [{"source": "db", "result": "db result"}]}


def search_docs(state):
    return {"messages": [{"source": "docs", "result": "docs result"}]}


def has_enough_info(state) -> bool:
    results = [m for m in state.get("messages", []) if isinstance(m, dict) and "source" in m]
    return len(results) >= 2


def respond(state):
    results = [m for m in state["messages"] if isinstance(m, dict) and "source" in m]
    meta = dict(state.get("metadata", {}))
    meta["answer"] = f"Found {len(results)} results"
    return {"metadata": meta}


def ask_clarification(state):
    meta = dict(state.get("metadata", {}))
    meta["answer"] = "Need more information"
    return {"metadata": meta}


app = (
    Trans(state_schema=State)
    .concurrent(search_web, search_db, search_docs)
    .optional(has_enough_info, then_=respond, else_=ask_clarification)
    .compile()
)

if __name__ == "__main__":
    result = app.invoke({"messages": [], "metadata": {}})
    print(result["metadata"]["answer"])

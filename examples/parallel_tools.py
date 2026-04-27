"""Example 2: Parallel tool calls with conditional routing."""
import operator
from typing import Annotated, TypedDict

from langtrans.builder import Trans


class State(TypedDict):
    messages: Annotated[list, operator.add]
    metadata: dict


def search_web(state: State):
    meta = dict(state.get("metadata", {}))
    results = list(meta.get("results", []))
    results.append("web result")
    meta["results"] = results
    return {"metadata": meta}


def search_db(state):
    meta = dict(state.get("metadata", {}))
    results = list(meta.get("results", []))
    results.append("db result")
    meta["results"] = results
    return {"metadata": meta}


def search_docs(state):
    meta = dict(state.get("metadata", {}))
    results = list(meta.get("results", []))
    results.append("docs result")
    meta["results"] = results
    return {"metadata": meta}


def has_enough_info(state) -> bool:
    return len(state.get("metadata", {}).get("results", [])) >= 2


def respond(state):
    meta = dict(state.get("metadata", {}))
    meta["answer"] = f"Found {len(meta.get('results', []))} results"
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

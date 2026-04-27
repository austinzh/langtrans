"""Example 3: ReAct-style agent loop."""

import operator
from typing import Annotated, TypedDict

from langtrans.builder import Proc, Trans


class State(TypedDict):
    messages: Annotated[list, operator.add]
    metadata: dict


def call_agent(state):
    meta = dict(state.get("metadata", {}))
    step = meta.get("step", 0) + 1
    meta["step"] = step
    if step >= 3:
        meta["done"] = True
        meta["tool_needed"] = False
    else:
        meta["tool_needed"] = True
    return {"metadata": meta}


def should_use_tool(state) -> bool:
    return state.get("metadata", {}).get("tool_needed", False)


def execute_tool(state):
    meta = dict(state.get("metadata", {}))
    tools = list(meta.get("tool_results", []))
    tools.append(f"tool_result_{meta.get('step', 0)}")
    meta["tool_results"] = tools
    return {"metadata": meta}


app = (
    Trans(state_schema=State)
    .loop(
        until=lambda s: s.get("metadata", {}).get("done", False),
        body=Proc().sequential(
            call_agent,
            Proc().optional(
                should_use_tool,
                then_=execute_tool,
            ),
        ),
    )
    .compile()
)

if __name__ == "__main__":
    result = app.invoke({"messages": [], "metadata": {}})
    print(f"Steps: {result['metadata']['step']}")
    print(f"Tool results: {result['metadata'].get('tool_results', [])}")
    print(f"Done: {result['metadata']['done']}")

"""
ReAct Agent: langtrans vs raw LangGraph

A real LLM-powered ReAct agent that uses OpenAI to decide whether
to call tools or respond directly. This is the canonical LangGraph
example, shown both ways.

Requires: OPENAI_API_KEY environment variable

Usage:
    python examples/react_agent_openai.py "What is 25 * 47?"
"""

import operator
import sys
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from langtrans import Trans


# ── Shared setup ─────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


@tool
def search(query: str) -> str:
    """Search the web for information."""
    return (
        f"Search results for '{query}': Python is a programming language "
        "created by Guido van Rossum in 1991."
    )


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))


tools = [search, calculate]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


def call_llm(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def has_tool_calls(state: AgentState) -> bool:
    last = state["messages"][-1]
    return bool(getattr(last, "tool_calls", None))


# =====================================================================
# Raw LangGraph — 12 lines of graph wiring
# =====================================================================

def build_raw_langgraph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", call_llm)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        lambda s: "tools" if has_tool_calls(s) else END,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


# =====================================================================
# langtrans — 7 lines
# =====================================================================

def build_langtrans():
    return (
        Trans(state_schema=AgentState)
        .sequential(call_llm)
        .loop(
            until=lambda s: not has_tool_calls(s),
            body=Trans().sequential(tool_node, call_llm),
        )
        .compile()
    )


# =====================================================================
# Run
# =====================================================================

if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "What is 25 * 47?"

    print("=" * 60)
    print("QUESTION:", question)
    print("=" * 60)

    print("\n── Raw LangGraph ──")
    raw_app = build_raw_langgraph()
    raw_result = raw_app.invoke({"messages": [HumanMessage(content=question)]})
    print(raw_result["messages"][-1].content)

    print("\n── langtrans ──")
    lt_app = build_langtrans()
    lt_result = lt_app.invoke({"messages": [HumanMessage(content=question)]})
    print(lt_result["messages"][-1].content)

    print("\n" + "=" * 60)

# langtrans

A high-level Python DSL that compiles to [LangGraph](https://github.com/langchain-ai/langgraph) graphs.

Inspired by [trans-dsl](https://github.com/agiledragon/trans-dsl), langtrans lets you declaratively describe agent workflows using composable primitives — and compiles them into LangGraph's `CompiledStateGraph` with full access to persistence, streaming, human-in-the-loop, and observability.

## Why?

LangGraph gives you powerful runtime capabilities (checkpointing, streaming, time-travel debugging). But building graphs is verbose:

```python
# Raw LangGraph — 12 lines of wiring
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
app = graph.compile()
```

```python
# langtrans — same graph, 7 lines
app = (
    Trans(state_schema=AgentState)
    .sequential(call_llm)
    .loop(
        until=lambda s: not has_tool_calls(s),
        body=Trans().sequential(tool_node, call_llm),
    )
    .compile()
)
```

Same runtime superpowers. Less boilerplate.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+ and `langgraph`.

## Quick Start

```python
import operator
from typing import Annotated, TypedDict
from langtrans import Trans

class State(TypedDict):
    messages: Annotated[list, operator.add]
    metadata: dict

def fetch_data(state):
    meta = dict(state["metadata"])
    meta["data"] = "fetched"
    return {"metadata": meta}

def process(state):
    meta = dict(state["metadata"])
    meta["result"] = f"processed {meta['data']}"
    return {"metadata": meta}

def respond(state):
    meta = dict(state["metadata"])
    meta["response"] = f"Done: {meta['result']}"
    return {"metadata": meta}

app = (
    Trans(state_schema=State)
    .sequential(fetch_data, process, respond)
    .compile()
)

result = app.invoke({"messages": [], "metadata": {}})
print(result["metadata"]["response"])
# Done: processed fetched
```

## Primitives

### Sequential

Run steps in order:

```python
Trans(state_schema=State).sequential(step_a, step_b, step_c).compile()
```

### Concurrent

Run steps in parallel (fan-out / fan-in):

```python
Trans(state_schema=State).concurrent(search_web, search_db, search_docs).compile()
```

### Optional

Conditional branching:

```python
Trans(state_schema=State).optional(
    is_confident, then_=respond, else_=ask_clarification
).compile()
```

Guards can be plain functions or composable `Spec` objects:

```python
from langtrans import Spec

confident = Spec(lambda s: s["metadata"]["confidence"] > 0.8)
has_data = Spec(lambda s: "data" in s["metadata"])

Trans(state_schema=State).optional(
    confident & has_data, then_=respond, else_=gather_more
).compile()
```

### Loop

Repeat a body N times or until a condition:

```python
# Fixed count
Trans(state_schema=State).loop(body=retry_step, times=3).compile()

# Until condition
Trans(state_schema=State).loop(
    body=agent_step,
    until=lambda s: s["metadata"].get("done"),
).compile()
```

### Retry

Retry on failure with configurable attempts and delay:

```python
Trans(state_schema=State).retry(flaky_api_call, max_attempts=3, delay=1.0).compile()
```

### Procedure

Named sub-transactions for reuse and organization:

```python
fetch_pipeline = Trans().sequential(fetch, validate, transform)

Trans(state_schema=State)
    .procedure("ingest", fetch_pipeline)
    .sequential(respond)
    .compile()
```

### Switch

Multi-way branching — the key to arbitrary state machines:

```python
Trans(state_schema=State).loop(
    until=lambda s: s["metadata"].get("done"),
    body=Trans().switch(
        key=classify_situation,
        cases={
            "need_tools":   execute_tools,
            "need_research": do_research,
            "need_approval": wait_for_human,
            "ready":         generate_response,
        },
    ),
).compile()
```

### Nesting

All primitives compose via nested `Trans()` builders:

```python
app = (
    Trans(state_schema=State)
    .sequential(
        fetch_data,
        Trans().concurrent(
            Trans().sequential(call_llm, parse_result),
            search_web,
        ),
        Trans().optional(
            is_confident,
            then_=respond,
            else_=Trans().retry(escalate, max_attempts=3),
        ),
    )
    .compile()
)
```

## Rollback / Saga Pattern

Actions can declare rollback functions for automatic compensation on failure:

```python
from langtrans import Trans, action

@action(rollback=undo_write_db)
def write_db(state): ...

@action(rollback=undo_notify)
def notify_service(state): ...

app = (
    Trans(state_schema=State)
    .sequential(validate, write_db, notify_service, confirm)
    .compile()
)
# If notify_service fails: undo_write_db runs automatically
```

## LangGraph Pass-Through

`compile()` passes all kwargs directly to LangGraph:

```python
from langgraph.checkpoint.memory import MemorySaver

app = flow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["human_review"],
)
```

You get the full `CompiledStateGraph` — use it exactly as you would any LangGraph graph.

## State Management

Define state as a `TypedDict` with LangGraph-style annotations:

```python
class MyState(TypedDict):
    messages: Annotated[list, operator.add]  # accumulates via +
    metadata: dict                           # replaced on update
```

Actions return partial state updates: `{"metadata": new_meta}`. The library doesn't transform state — it's a direct pass-through to LangGraph.

## Examples

| Example | What it shows |
|---------|--------------|
| [`sequential_agent.py`](examples/sequential_agent.py) | Simple 3-step pipeline |
| [`parallel_tools.py`](examples/parallel_tools.py) | Concurrent search + conditional routing |
| [`react_agent.py`](examples/react_agent.py) | ReAct loop (simulated LLM) |
| [`react_agent_openai.py`](examples/react_agent_openai.py) | Real OpenAI ReAct agent vs raw LangGraph |
| [`rollback_pipeline.py`](examples/rollback_pipeline.py) | Saga-pattern rollback on failure |
| [`smart_assistant.py`](examples/smart_assistant.py) | Domain-driven state machine via `loop` + `switch` |

```bash
python examples/sequential_agent.py
python examples/smart_assistant.py
```

## When to Use langtrans vs Plain Python

Use **plain Python** when you don't need failure recovery, human-in-the-loop, streaming, or observability.

Use **langtrans** (compiling to LangGraph) when you need:
- Checkpoint/resume after failures
- Human approval gates
- Time-travel debugging
- LangSmith tracing
- Token streaming from LLM nodes
- Workflow visualization

langtrans doesn't add computation power over Python. It makes LangGraph's runtime superpowers easy to access.

## License

MIT

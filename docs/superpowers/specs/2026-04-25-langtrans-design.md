# langtrans — Design Spec

A high-level Python DSL inspired by [trans-dsl](https://github.com/agiledragon/trans-dsl) that compiles to [LangGraph](https://github.com/langchain-ai/langgraph) graphs.

## Goal

Replace verbose LangGraph graph construction with a fluent, declarative builder API. Users describe workflows using composable primitives (Sequential, Concurrent, Optional, Loop, Retry, Procedure) and `langtrans` compiles the description into a `CompiledStateGraph`.

Primary audience: AI agent workflows. Also supports general transaction orchestration.

## Architecture

Three layers:

1. **DSL Layer** — `@action` decorator and `Trans` fluent builder. User-facing API.
2. **Node Model** — Dataclass tree (`ActionNode`, `SequentialNode`, etc.). Pure data, no logic.
3. **Compiler** — Walks the node tree, emits `StateGraph` nodes/edges, returns `CompiledStateGraph`.

```
User code (decorators + builder)
        ↓
   DSL Node Tree
        ↓
   Compiler (tree walk)
        ↓
   LangGraph StateGraph
        ↓
   .compile(**kwargs)  →  pass-through to LangGraph
        ↓
   CompiledStateGraph
```

Single package. Only dependency: `langgraph`.

## DSL Layer

### Decorators

```python
@action
def fetch_data(state):
    return {"metadata": {**state["metadata"], "data": call_api()}}

@action(rollback=undo_fetch)
def write_db(state):
    return {"metadata": {**state["metadata"], "record_id": save(state)}}

def undo_fetch(state):
    cleanup(state["metadata"]["data"])
    return state
```

Actions are `State -> dict` (partial state updates), matching LangGraph's node signature. Rollback functions have the same signature. Both are optional — `@action` without rollback is valid.

### Builder

```python
flow = (
    Trans(state_schema=MyState)
    .sequential(
        fetch_data,
        write_db,
    )
    .concurrent(call_llm, search_web)
    .optional(is_confident, then_=respond, else_=escalate)
    .loop(times=3, body=retry_step)
    .retry(call_api, max_attempts=3, delay=1.0)
    .procedure("sub_flow", sub_transaction)
    .compile(checkpointer=MemorySaver())
)
```

- When chaining multiple builder methods (e.g., `.sequential(...).concurrent(...)`), they are wrapped in an implicit top-level Sequential.
- A single builder method call produces that primitive as the root directly — no wrapping.
- Nesting is done by passing `Trans()` sub-builders as arguments: `concurrent(a, Trans().sequential(b, c))`.
- `.compile(**kwargs)` builds the node tree and compiles to LangGraph. All kwargs pass through to LangGraph's `graph.compile()`.
- `then_` / `else_` use trailing underscores to avoid Python keyword conflicts.

### Guards

Plain functions or composable `Spec` objects:

```python
def is_confident(state) -> bool:
    return state["metadata"].get("confidence", 0) > 0.8

# composable:
confident = Spec(is_confident)
has_data = Spec(lambda s: "data" in s["metadata"])

flow.optional(confident & has_data, then_=respond, else_=escalate)
```

`Spec` supports `&` (AllOf), `|` (AnyOf), `~` (Not). Plain `Callable[[State], bool]` is accepted everywhere a `Spec` is.

## Node Model

```python
Node = Union[ActionNode, SequentialNode, ConcurrentNode, OptionalNode,
             LoopNode, RetryNode, ProcedureNode]

@dataclass
class ActionNode:
    func: Callable
    rollback: Optional[Callable] = None
    name: Optional[str] = None          # defaults to func.__name__

@dataclass
class SequentialNode:
    children: list[Node]

@dataclass
class ConcurrentNode:
    children: list[Node]

@dataclass
class OptionalNode:
    guard: Callable | Spec
    then_: Node
    else_: Optional[Node] = None

@dataclass
class LoopNode:
    body: Node
    times: Optional[int] = None         # fixed count
    until: Optional[Callable] = None    # condition-based exit

@dataclass
class RetryNode:
    body: Node
    max_attempts: int = 3
    delay: float = 0.0

@dataclass
class ProcedureNode:
    name: str
    body: Node
```

Each node gets a unique ID during compilation for LangGraph node names.

## Compiler — Primitive Mappings

### Single entry / single exit contract

Every `compile_node()` call returns `(entry_id, exit_id)` — one entry point and one exit point. This is the key invariant that makes composition work: a parent node can always wire `prev_exit → entry` and `exit → next_entry` without knowing what's inside.

Primitives with multiple branches (Optional, Switch, Concurrent) use synthetic no-op nodes to maintain this contract:

- **fork / join** — ConcurrentNode fans out from one fork node and converges at one join node
- **decision / merge** — OptionalNode and SwitchNode route from one decision node and converge at one merge node

These no-op nodes (`lambda s: {}`) cost nothing at runtime — LangGraph passes through them. Without them, multi-branch primitives would need to return multiple exit IDs, complicating every parent's edge-wiring logic.

### Primitive mappings

**ActionNode** → Single LangGraph node.

**SequentialNode** → Chain of edges: `add_edge(child_n, child_n+1)`.

**ConcurrentNode** → Fan-out / fan-in. Synthetic "fork" node edges to all children. All branches edge to synthetic "join" node. LangGraph runs independent branches concurrently. Concurrent branches must write to state channels with reducers (e.g., `Annotated[list, operator.add]`).

**OptionalNode** → Synthetic "decision" node with `add_conditional_edges` using the guard. Routes to `then_` or `else_` (or directly to merge if no else). Both branches converge at a "merge" node.

**SwitchNode** → Synthetic "switch" decision node with `add_conditional_edges` using the key function. Routes to one of N case branches. All cases converge at a "switch_merge" node.

**LoopNode** → Cycle in the graph. Body compiles normally. Conditional edge at the end loops back or exits. Loop counter stored in `metadata["_langtrans"]`.

**Named nodes** → Any node with `name` set compiles its children with a prefix (`"name.child_id"`), providing namespace isolation. This replaces the former ProcedureNode.

**Rollback / Compensation** — Sequential nodes containing actions with rollbacks are wrapped in a single LangGraph node that runs steps in order, tracks completed rollbacks, and on failure runs them in reverse.

```
Sequential(A, B, C) where A,B have rollbacks:

  A → B → C → END
       ↘       ↘
    rollback_B → rollback_A → END (with error)
```

## State Management

```python
from typing import TypedDict, Annotated
import operator

class MyState(TypedDict):
    messages: Annotated[list, operator.add]
    metadata: dict
```

- `state_schema` passes directly to `StateGraph(state_schema)`.
- The library does not interpret or transform state.
- Internal bookkeeping lives under `metadata["_langtrans"]`:

```python
metadata["_langtrans"] = {
    "loop_0_counter": 2,
    "retry_0_attempts": 1,
    "_rollback_stack": ["write_db", "fetch_data"],
}
```

## Error Handling

- Exceptions propagate by default (same as LangGraph).
- Actions with rollbacks trigger the compensation chain on failure.
- RetryNode catches exceptions and re-routes. After exhausting attempts, exception propagates.
- No custom error types imposed — users catch exceptions inside their actions if needed.

## Examples — langtrans vs raw LangGraph

### Example 1: Sequential Agent with Tools

A chatbot that fetches data, calls an LLM, then responds.

**Raw LangGraph:**
```python
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    messages: Annotated[list, operator.add]
    metadata: dict

graph = StateGraph(State)
graph.add_node("fetch", fetch_data)
graph.add_node("llm", call_llm)
graph.add_node("respond", format_response)
graph.add_edge(START, "fetch")
graph.add_edge("fetch", "llm")
graph.add_edge("llm", "respond")
graph.add_edge("respond", END)
app = graph.compile()
```

**langtrans:**
```python
app = (
    Trans(state_schema=State)
    .sequential(fetch_data, call_llm, format_response)
    .compile()
)
```

### Example 2: Parallel Tool Calls with Conditional Routing

An agent that searches multiple sources in parallel, then decides whether to respond or ask for clarification.

**Raw LangGraph:**
```python
graph = StateGraph(State)
graph.add_node("search_web", search_web)
graph.add_node("search_db", search_db)
graph.add_node("search_docs", search_docs)
graph.add_node("merge", merge_results)
graph.add_node("respond", respond)
graph.add_node("clarify", ask_clarification)

graph.add_edge(START, "search_web")
graph.add_edge(START, "search_db")
graph.add_edge(START, "search_docs")
graph.add_edge("search_web", "merge")
graph.add_edge("search_db", "merge")
graph.add_edge("search_docs", "merge")
graph.add_conditional_edges("merge", has_enough_info, {True: "respond", False: "clarify"})
graph.add_edge("respond", END)
graph.add_edge("clarify", END)
app = graph.compile()
```

**langtrans:**
```python
app = (
    Trans(state_schema=State)
    .concurrent(search_web, search_db, search_docs)
    .optional(has_enough_info, then_=respond, else_=ask_clarification)
    .compile()
)
```

### Example 3: ReAct Agent with Retry

An agent loop: the LLM decides to use a tool or finish, with retry on tool failures.

**Raw LangGraph:**
```python
graph = StateGraph(State)
graph.add_node("agent", call_agent)
graph.add_node("tools", execute_tool)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_use_tool, {True: "tools", False: END})
graph.add_edge("tools", "agent")
app = graph.compile()
# retry logic requires custom wrapper around execute_tool
```

**langtrans:**
```python
app = (
    Trans(state_schema=State)
    .loop(
        until=lambda s: s["metadata"].get("done", False),
        body=Trans().sequential(
            call_agent,
            Trans().optional(
                should_use_tool,
                then_=Trans().retry(execute_tool, max_attempts=3),
            ),
        ),
    )
    .compile()
)
```

### Example 4: Multi-step Pipeline with Rollback

A data pipeline that writes to external systems, with compensation on failure.

**Raw LangGraph:** Requires manual rollback orchestration — LangGraph has no built-in Saga support. Users must build compensation nodes and failure edges by hand.

**langtrans:**
```python
@action(rollback=undo_write_db)
def write_db(state): ...

@action(rollback=undo_notify)
def notify_service(state): ...

app = (
    Trans(state_schema=State)
    .sequential(
        validate_input,
        write_db,
        notify_service,
        send_confirmation,
    )
    .compile()
)
# If notify_service fails: undo_write_db runs automatically
```

## Package Structure

```
langtrans/
    __init__.py          # public API exports
    nodes.py             # dataclass node model
    builder.py           # Trans builder + @action decorator
    compiler.py          # DSL tree → StateGraph
    spec.py              # Spec guard combinators
tests/
    test_nodes.py
    test_builder.py
    test_compiler.py
    test_spec.py
    test_integration.py
examples/
    sequential_agent.py
    parallel_tools.py
    react_agent.py
    rollback_pipeline.py
```

## Out of Scope for V1

- Custom LangGraph node types (e.g., ToolNode)
- Streaming callbacks from within the DSL
- Graph visualization helpers
- Async action support (add in V2)
- Persistence / state serialization beyond LangGraph's built-in

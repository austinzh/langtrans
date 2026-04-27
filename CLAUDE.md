# CLAUDE.md

## Project

langtrans — A high-level Python DSL inspired by [trans-dsl](https://github.com/agiledragon/trans-dsl) that compiles to [LangGraph](https://github.com/langchain-ai/langgraph) graphs.

## Quick start

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run an example
python examples/smart_assistant.py
```

## Architecture

Three layers:

1. **DSL Layer** (`builder.py`) — `@action` decorator and `Trans` fluent builder. User-facing API.
2. **Node Model** (`nodes.py`) — Dataclass tree: `ActionNode`, `SequentialNode`, `ConcurrentNode`, `OptionalNode`, `LoopNode`, `RetryNode`, `ProcedureNode`, `SwitchNode`. Pure data, no logic.
3. **Compiler** (`compiler.py`) — Walks the node tree, emits `StateGraph` nodes/edges, returns `CompiledStateGraph`.

Supporting: `spec.py` (Spec guard combinators with `&`, `|`, `~`).

## Package layout

```
langtrans/
    __init__.py      # public API exports
    nodes.py         # dataclass node types + Node union type
    builder.py       # Trans builder, @action decorator, _to_node()
    compiler.py      # _Compiler class, compile_graph() entry point
    spec.py          # Spec guard with boolean combinators
tests/
    conftest.py      # shared fixtures (SimpleState, make_action)
    test_nodes.py
    test_spec.py
    test_builder.py
    test_compiler.py # tests for each primitive's compilation
    test_switch.py
    test_integration.py
    test_public_api.py
examples/           # runnable demos
```

## Conventions

- Actions are `State -> dict` (partial state update), matching LangGraph's node signature
- Compiler methods return `(entry_node_id, exit_node_id)` tuples
- Internal bookkeeping goes in `metadata["_langtrans"]` to avoid user key collisions
- Node IDs are auto-generated with `_unique_id()` to avoid LangGraph name collisions
- The builder accepts plain callables, `Trans` sub-builders, `Node` dataclasses, and LangGraph `Runnable` objects

## Testing

```bash
pytest tests/ -v              # full suite (73 tests)
pytest tests/test_switch.py   # just switch tests
pytest tests/test_compiler.py # just compiler tests
```

## Key design decisions

- `Trans.compile(**kwargs)` passes all kwargs through to LangGraph's `graph.compile()` — no wrapping
- Rollback/compensation wraps sequential steps in a single LangGraph node that runs them in order with try/catch
- `ConcurrentNode` runs actions in a single LangGraph node to avoid merge conflicts on `metadata: dict` (not annotated with a reducer)
- `switch` + `loop` composes into arbitrary state machines — no need for explicit goto or arbitrary graph edges

# CLAUDE.md

## Project

langtrans — A high-level Python DSL inspired by [trans-dsl](https://github.com/agiledragon/trans-dsl) that compiles to [LangGraph](https://github.com/langchain-ai/langgraph) graphs.

## Quick start

```bash
# Install in dev mode
uv sync

# Run checks (ruff + mypy)
make check

# Run tests
make test

# Run an example
uv run python examples/smart_assistant.py
```

## Architecture

Three layers:

1. **DSL Layer** (`builder.py`) — `@action` decorator, `Trans` top-level builder, `Proc` named sub-builder, and top-level convenience functions (`sequential`, `concurrent`, `optional`, `loop`, `switch`). User-facing API.
2. **Node Model** (`nodes.py`) — Dataclass tree: `ActionNode`, `SequentialNode`, `ConcurrentNode`, `OptionalNode`, `LoopNode`, `SwitchNode`. Pure data, no logic.
3. **Compiler** (`compiler.py`) — Walks the node tree, emits `StateGraph` nodes/edges, returns `CompiledStateGraph`.

Supporting: `spec.py` (Spec guard combinators with `&`, `|`, `~`).

## Package layout

```
langtrans/
    __init__.py      # public API exports
    nodes.py         # dataclass node types + Node union type
    builder.py       # Trans/Proc builders, @action decorator, _to_node()
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
    test_async.py
examples/           # runnable demos
```

## Conventions

- Actions are `State -> dict` (partial state update), matching LangGraph's node signature
- Compiler methods return `(entry_node_id, exit_node_id)` tuples
- Internal bookkeeping goes in `metadata["_langtrans"]` to avoid user key collisions
- Node IDs are auto-generated with `_unique_id()` to avoid LangGraph name collisions
- `Trans` is the top-level builder (has `state_schema` and `compile()`); `Proc("name")` is for named groups
- Top-level functions (`sequential`, `concurrent`, `optional`, `loop`, `switch`) are the preferred way to build sub-workflows — they delegate to `Proc()` internally
- The builder accepts plain callables, `Proc`/top-level function results, `Node` dataclasses, and LangGraph `Runnable` objects

## Testing

```bash
make test                         # full suite (79 tests)
uv run pytest tests/test_switch.py   # just switch tests
uv run pytest tests/test_compiler.py # just compiler tests
```

## Key design decisions

- `Trans.compile(**kwargs)` passes all kwargs through to LangGraph's `graph.compile()` — no wrapping
- Rollback/compensation wraps sequential steps in a single LangGraph node that runs them in order with try/catch
- `ConcurrentNode` compiles to a fork node, one LangGraph node per branch, and a join node (fan-out / fan-in). Parallel updates to the same state field need a reducer in the state schema (for example `Annotated[list, operator.add]`); a plain `dict` channel such as `metadata` can see last-writer-wins behavior across branches
- Control flow is `sequential` / `concurrent` / `optional` / `switch` / `loop` and nesting — the builder does not add ad-hoc edges; reach for a nested `Proc`, `Runnable`, or raw LangGraph when you need a topology the primitives do not cover
- No `RetryNode` — users should use `tenacity` or `backoff` for retry logic

## Tooling

- **uv** — package management and virtual env
- **ruff** — linting and formatting
- **mypy** — strict type checking on `langtrans/`
- **pre-commit** — ruff + mypy hooks run on every commit
- **Makefile** — `make test`, `make check`, `make lint`, `make format`, `make fix`

## Documentation

When you change public API, architecture, install steps, or project facts, update **both** `CLAUDE.md` (this file) and `README.md` so they stay consistent. Prefer editing them in the same change as the code when the behavior or story has shifted.

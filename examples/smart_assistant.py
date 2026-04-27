"""
Example: Smart Assistant with domain-driven routing

An assistant that classifies user requests and routes to the right
handler — no artificial _state variable. The data itself drives routing.

Demonstrates: loop + switch with a domain classifier function.

Usage:
    python examples/smart_assistant.py
"""

import operator
from typing import Annotated, TypedDict

from langtrans import Trans


class State(TypedDict):
    messages: Annotated[list, operator.add]
    metadata: dict


# ── Domain classifier — reads actual data to decide next step ────────

def classify(state) -> str:
    meta = state.get("metadata", {})

    if not meta.get("query_understood"):
        return "understand"

    if meta.get("needs_calculation") and not meta.get("calc_result"):
        return "calculate"

    if meta.get("needs_lookup") and not meta.get("lookup_result"):
        return "lookup"

    if not meta.get("response_drafted"):
        return "draft"

    if meta.get("confidence", 1.0) < 0.7 and not meta.get("verified"):
        return "verify"

    return "deliver"


def is_done(state) -> bool:
    return state.get("metadata", {}).get("delivered", False)


# ── Actions — each modifies domain data, which drives next routing ───

def understand_query(state):
    """Parse the user query and figure out what's needed."""
    meta = dict(state.get("metadata", {}))
    query = meta.get("query", "")

    meta["query_understood"] = True
    meta["needs_calculation"] = any(op in query for op in ["+", "-", "*", "/", "calculate"])
    meta["needs_lookup"] = any(kw in query.lower() for kw in ["who", "what", "when", "where", "search"])

    log = list(meta.get("log", []))
    log.append(f"Understood query: calc={meta['needs_calculation']}, lookup={meta['needs_lookup']}")
    meta["log"] = log
    return {"metadata": meta}


def do_calculation(state):
    """Perform calculation if the query requires it."""
    meta = dict(state.get("metadata", {}))
    meta["calc_result"] = "42"
    log = list(meta.get("log", []))
    log.append("Calculated result: 42")
    meta["log"] = log
    return {"metadata": meta}


def do_lookup(state):
    """Search for information if the query requires it."""
    meta = dict(state.get("metadata", {}))
    meta["lookup_result"] = "Python was created by Guido van Rossum in 1991"
    log = list(meta.get("log", []))
    log.append("Looked up information")
    meta["log"] = log
    return {"metadata": meta}


def draft_response(state):
    """Draft a response from the gathered information."""
    meta = dict(state.get("metadata", {}))

    parts = []
    if meta.get("calc_result"):
        parts.append(f"Calculation: {meta['calc_result']}")
    if meta.get("lookup_result"):
        parts.append(f"Info: {meta['lookup_result']}")
    if not parts:
        parts.append("I understood your query but didn't need any tools.")

    meta["draft"] = " | ".join(parts)
    meta["response_drafted"] = True
    meta["confidence"] = 0.5 if meta.get("needs_lookup") else 0.95

    log = list(meta.get("log", []))
    log.append(f"Drafted response (confidence={meta['confidence']})")
    meta["log"] = log
    return {"metadata": meta}


def verify_response(state):
    """Double-check a low-confidence response."""
    meta = dict(state.get("metadata", {}))
    meta["verified"] = True
    meta["confidence"] = 0.95
    meta["draft"] = meta["draft"] + " [verified]"

    log = list(meta.get("log", []))
    log.append("Verified response — confidence boosted")
    meta["log"] = log
    return {"metadata": meta}


def deliver_response(state):
    """Send the final response to the user."""
    meta = dict(state.get("metadata", {}))
    meta["final_response"] = meta["draft"]
    meta["delivered"] = True

    log = list(meta.get("log", []))
    log.append("Delivered response")
    meta["log"] = log
    return {"metadata": meta}


# ── Build the graph ──────────────────────────────────────────────────

app = (
    Trans(state_schema=State)
    .loop(
        until=is_done,
        body=Trans().switch(
            key=classify,
            cases={
                "understand": understand_query,
                "calculate":  do_calculation,
                "lookup":     do_lookup,
                "draft":      draft_response,
                "verify":     verify_response,
                "deliver":    deliver_response,
            },
        ),
    )
    .compile()
)


# ── Run examples ─────────────────────────────────────────────────────

if __name__ == "__main__":
    scenarios = [
        ("Simple greeting", "Hello there!"),
        ("Math question", "Please calculate 6 * 7"),
        ("Knowledge question", "Who created Python?"),
        ("Both needed", "What is Python and calculate 6 * 7"),
    ]

    for title, query in scenarios:
        print(f"\n{'=' * 60}")
        print(f"  {title}: \"{query}\"")
        print(f"{'=' * 60}")

        result = app.invoke({"messages": [], "metadata": {"query": query}})
        meta = result["metadata"]

        print(f"\n  Route taken:")
        for step in meta["log"]:
            print(f"    → {step}")

        print(f"\n  Response: {meta['final_response']}")
        print(f"  Confidence: {meta['confidence']}")

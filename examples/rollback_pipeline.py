"""Example 4: Multi-step pipeline with Saga-pattern rollback."""

import operator
from typing import Annotated, TypedDict

from langtrans.builder import Trans, action


class State(TypedDict):
    messages: Annotated[list, operator.add]
    metadata: dict


rollback_log = []


def undo_write_db(state):
    rollback_log.append("undo_write_db")
    print("Rolling back: undo_write_db")
    return {}


def undo_notify(state):
    rollback_log.append("undo_notify")
    print("Rolling back: undo_notify")
    return {}


def validate_input(state):
    meta = dict(state.get("metadata", {}))
    meta["validated"] = True
    return {"metadata": meta}


@action(rollback=undo_write_db)
def write_db(state):
    meta = dict(state.get("metadata", {}))
    meta["db_written"] = True
    return {"metadata": meta}


@action(rollback=undo_notify)
def notify_service(state):
    if state.get("metadata", {}).get("should_fail", False):
        raise ValueError("Notification service unavailable")
    meta = dict(state.get("metadata", {}))
    meta["notified"] = True
    return {"metadata": meta}


def send_confirmation(state):
    meta = dict(state.get("metadata", {}))
    meta["confirmed"] = True
    return {"metadata": meta}


app = (
    Trans(state_schema=State)
    .sequential(validate_input, write_db, notify_service, send_confirmation)
    .compile()
)

if __name__ == "__main__":
    print("=== Success case ===")
    rollback_log.clear()
    result = app.invoke({"messages": [], "metadata": {}})
    print(f"Result: {result['metadata']}")
    print(f"Rollbacks: {rollback_log}")

    print("\n=== Failure case ===")
    rollback_log.clear()
    try:
        result = app.invoke({"messages": [], "metadata": {"should_fail": True}})
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Rollbacks: {rollback_log}")

"""Pure functions that turn a call snapshot into what the page displays.

Kept free of Streamlit so the logic (which step is active, how many attempts
are left, what was chosen and how) is unit-tested like any other code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ui.session_client import Snapshot


# --------------------------------------------------------------------------- #
def analyse(snapshot: Snapshot) -> dict:
    """Fold the event stream into what the page needs to show."""

    language: str | None = None
    method = confidence = reason = None
    rejected = 0
    attempts_remaining: int | None = None
    selection_started: float | None = None
    selection_seconds: float | None = None
    welcome_played = False
    created = False
    failure: str | None = None
    last_transcript: dict | None = None

    for entry in snapshot.entries:
        kind, payload = entry.type, entry.payload

        if entry.source == "client":
            if kind == "reset":
                language = method = confidence = reason = None
                rejected = 0
                attempts_remaining = None
                selection_seconds = None
                selection_started = entry.at
            continue

        if kind == "session.created":
            created = True
        elif kind == "prompt.audio" and payload.get("prompt") == "welcome":
            welcome_played = True
        elif kind == "state.changed" and payload.get("state") == "language_selection":
            if selection_started is None:
                selection_started = entry.at
        elif kind == "transcript":
            last_transcript = payload
        elif kind == "language.rejected":
            rejected = payload.get("attempt", rejected + 1)
            attempts_remaining = payload.get("attempts_remaining")
        elif kind == "language.selected":
            language = payload.get("language")
            method = payload.get("method")
            confidence = payload.get("confidence")
            reason = payload.get("reason")
            if selection_started is not None:
                selection_seconds = entry.at - selection_started
        elif kind == "language.failed":
            failure = payload.get("message", "Language selection failed.")
            rejected = payload.get("attempts", rejected)
            attempts_remaining = 0
        elif kind == "error" and payload.get("fatal"):
            failure = payload.get("message", "Fatal error.")

    return {
        "created": created,
        "welcome_played": welcome_played,
        "language": language,
        "method": method,
        "confidence": confidence,
        "reason": reason,
        "rejected": rejected,
        "attempts_remaining": attempts_remaining,
        "selection_seconds": selection_seconds,
        "failure": failure,
        "last_transcript": last_transcript,
    }


def step_states(snapshot: Snapshot, view: dict) -> dict[str, str]:
    state = snapshot.state
    failed = view["failure"] is not None or snapshot.connect_error is not None
    selected = view["language"] is not None

    connect = "done" if view["created"] else ("error" if failed else "active")

    if not view["created"]:
        welcome = "pending"
    elif state in ("connecting", "welcome") and not failed:
        welcome = "active"
    elif view["welcome_played"]:
        welcome = "done"
    else:
        welcome = "error" if failed else "pending"

    if selected:
        language = "done"
    elif state == "language_selection":
        language = "active"
    elif failed and view["created"]:
        language = "error"
    else:
        language = "pending"

    ready = "done" if selected and state in ("active", "thinking") else "pending"
    if selected and state not in ("active", "thinking", "closed"):
        ready = "active"

    return {"connect": connect, "welcome": welcome, "language": language, "ready": ready}

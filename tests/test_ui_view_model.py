"""The Streamlit page's decision logic, tested without Streamlit or a socket."""

from __future__ import annotations

from types import SimpleNamespace

from ui.view_model import analyse, step_states


def server(at: float, kind: str, **payload) -> SimpleNamespace:
    return SimpleNamespace(at=at, source="server", type=kind, payload=payload)


def client(at: float, kind: str, **payload) -> SimpleNamespace:
    return SimpleNamespace(at=at, source="client", type=kind, payload=payload)


def snapshot(entries, *, state="connecting", connect_error=None) -> SimpleNamespace:
    return SimpleNamespace(entries=entries, state=state, connect_error=connect_error)


GREETING = [
    server(0.0, "session.created", session_id="abc"),
    server(0.0, "state.changed", state="welcome"),
    server(0.1, "prompt.audio", prompt="welcome"),
    server(0.1, "state.changed", state="language_selection"),
]


def test_stepper_follows_the_greeting() -> None:
    entries = GREETING[:2]
    snap = snapshot(entries, state="welcome")
    assert step_states(snap, analyse(snap)) == {
        "connect": "done",
        "welcome": "active",
        "language": "pending",
        "ready": "pending",
    }

    snap = snapshot(GREETING, state="language_selection")
    assert step_states(snap, analyse(snap)) == {
        "connect": "done",
        "welcome": "done",
        "language": "active",
        "ready": "pending",
    }


def test_confirmed_call_reports_how_the_language_was_chosen() -> None:
    entries = [
        *GREETING,
        server(3.0, "language.rejected", attempt=1, attempts_remaining=2),
        server(
            9.0,
            "language.selected",
            language="si-LK",
            method="voice",
            confidence=0.95,
            reason="keyword_exact",
        ),
        server(9.0, "state.changed", state="active"),
    ]
    snap = snapshot(entries, state="active")
    view = analyse(snap)

    assert view["language"] == "si-LK"
    assert view["method"] == "voice"
    assert view["rejected"] == 1
    assert view["selection_seconds"] == 9.0 - 0.1
    assert step_states(snap, view)["ready"] == "done"


def test_exhausted_attempts_mark_the_language_step_failed() -> None:
    entries = [
        *GREETING,
        server(2.0, "language.rejected", attempt=1, attempts_remaining=2),
        server(3.0, "language.rejected", attempt=2, attempts_remaining=1),
        server(4.0, "language.failed", attempts=3, message="Maximum attempts."),
    ]
    snap = snapshot(entries, state="closed")
    view = analyse(snap)

    assert view["failure"] == "Maximum attempts."
    assert (view["rejected"], view["attempts_remaining"]) == (3, 0)
    assert step_states(snap, view)["language"] == "error"


def test_reset_clears_the_previous_choice() -> None:
    entries = [
        *GREETING,
        server(2.0, "language.selected", language="en-US", method="ui"),
        client(5.0, "reset"),
        server(5.0, "state.changed", state="language_selection"),
    ]
    snap = snapshot(entries, state="language_selection")
    view = analyse(snap)

    assert view["language"] is None
    assert step_states(snap, view)["language"] == "active"


def test_a_refused_connection_is_an_error_on_the_first_step() -> None:
    snap = snapshot([], connect_error="Server refused the connection (HTTP 403).")

    assert step_states(snap, analyse(snap))["connect"] == "error"


def test_fatal_server_error_is_surfaced() -> None:
    entries = [
        *GREETING,
        server(5.0, "error", code="idle_timeout", message="Bye", fatal=True),
    ]
    snap = snapshot(entries, state="closed")

    assert analyse(snap)["failure"] == "Bye"

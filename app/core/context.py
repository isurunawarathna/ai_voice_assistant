"""Request/session scoped context propagated to every log record."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from uuid import uuid4

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)


def new_trace_id() -> str:
    return uuid4().hex


def get_trace_id() -> str | None:
    return _trace_id.get()


def get_session_id() -> str | None:
    return _session_id.get()


@contextmanager
def bind_context(
    *,
    trace_id: str | None = None,
    session_id: str | None = None,
) -> Iterator[str]:
    """Bind correlation ids for the duration of the block."""

    resolved_trace = trace_id or get_trace_id() or new_trace_id()
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = [
        (_trace_id, _trace_id.set(resolved_trace))
    ]

    if session_id is not None:
        tokens.append((_session_id, _session_id.set(session_id)))

    try:
        yield resolved_trace
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def set_session_id(session_id: str) -> None:
    """Attach a session id to the already-running context."""

    _session_id.set(session_id)

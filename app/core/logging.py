"""Structured logging configured once at startup."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from app.core.config import LoggingSettings
from app.core.context import get_session_id, get_trace_id

_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


class ContextFilter(logging.Filter):
    """Copy the contextvars onto each record so both formatters can use them."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id() or "-"
        record.session_id = get_session_id() or "-"
        return True


class JsonFormatter(logging.Formatter):
    def __init__(self, *, include_timestamp: bool = True) -> None:
        super().__init__()
        self.include_timestamp = include_timestamp

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", "-"),
            "session_id": getattr(record, "session_id", "-"),
        }

        if self.include_timestamp:
            payload["timestamp"] = datetime.fromtimestamp(
                record.created, tz=UTC
            ).isoformat(timespec="milliseconds")

        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload.setdefault(key, value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


_TEXT_FORMAT = (
    "%(asctime)s %(levelname)-8s [%(name)s] "
    "trace=%(trace_id)s session=%(session_id)s | %(message)s"
)


def configure_logging(settings: LoggingSettings) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(ContextFilter())
    handler.setFormatter(
        JsonFormatter(include_timestamp=settings.include_timestamp)
        if settings.json_format
        else logging.Formatter(_TEXT_FORMAT)
    )

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)

    root.addHandler(handler)
    root.setLevel(settings.level)

    # Uvicorn installs its own handlers; route them through ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

    logging.getLogger("faster_whisper").setLevel(logging.WARNING)

"""Admission control for the voice WebSocket.

Runs before the socket is accepted, because a rejected handshake costs almost
nothing while an accepted one streams megabytes of welcome audio.

Three independent checks:

* **Origin** - browsers always send ``Origin`` on a WebSocket handshake and
  cross-site pages can open sockets to any host, so a browser origin must be on
  the allow-list. Requests without an ``Origin`` (scripts, Streamlit, SIP
  gateways) are not browsers and cannot be a cross-site attack, so they pass.
* **Concurrent connections per client IP.**
* **New connections per minute per client IP** (sliding window).

State is per process. Behind several workers each applies the limit locally;
put the same limits on the load balancer for a global ceiling.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import StrEnum

from fastapi import WebSocket

from app.core.config import Settings


class Admission(StrEnum):
    ALLOWED = "accepted"
    ORIGIN = "rejected_origin"
    RATE_LIMIT = "rejected_rate_limit"
    CONCURRENCY = "rejected_concurrency"


@dataclass(slots=True, frozen=True)
class Decision:
    admission: Admission
    client_ip: str

    @property
    def allowed(self) -> bool:
        return self.admission is Admission.ALLOWED


class ConnectionGuard:
    WINDOW_SECONDS = 60.0

    def __init__(self, settings: Settings, *, clock=time.monotonic) -> None:
        self._settings = settings
        self._clock = clock
        self._open: dict[str, int] = defaultdict(int)
        self._recent: dict[str, deque[float]] = defaultdict(deque)

    # ------------------------------------------------------------------ #
    def client_ip(self, websocket: WebSocket) -> str:
        if self._settings.security.trust_forwarded_for:
            forwarded = websocket.headers.get("x-forwarded-for", "")
            first = forwarded.split(",")[0].strip()
            if first:
                return first
        client = websocket.client
        return client.host if client else "unknown"

    def check(self, websocket: WebSocket) -> Decision:
        """Decide whether to accept. On success the caller must ``release``."""

        ip = self.client_ip(websocket)

        if not self._origin_allowed(websocket.headers.get("origin")):
            return Decision(Admission.ORIGIN, ip)

        security = self._settings.security
        now = self._clock()

        window = self._recent[ip]
        while window and now - window[0] > self.WINDOW_SECONDS:
            window.popleft()

        if (
            security.max_connects_per_minute_per_ip
            and len(window) >= security.max_connects_per_minute_per_ip
        ):
            return Decision(Admission.RATE_LIMIT, ip)

        if (
            security.max_connections_per_ip
            and self._open[ip] >= security.max_connections_per_ip
        ):
            return Decision(Admission.CONCURRENCY, ip)

        window.append(now)
        self._open[ip] += 1
        return Decision(Admission.ALLOWED, ip)

    def release(self, ip: str) -> None:
        self._open[ip] = max(0, self._open[ip] - 1)
        if self._open[ip] == 0:
            del self._open[ip]
            if not self._recent.get(ip):
                self._recent.pop(ip, None)

    # ------------------------------------------------------------------ #
    def _origin_allowed(self, origin: str | None) -> bool:
        if not self._settings.origin_enforced or origin is None:
            return True
        allowed = {item.rstrip("/").lower() for item in self._settings.websocket_origins}
        return "*" in allowed or origin.rstrip("/").lower() in allowed

    @property
    def open_connections(self) -> int:
        return sum(self._open.values())

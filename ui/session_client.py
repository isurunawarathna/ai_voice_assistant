"""Thread-backed WebSocket client for the Streamlit UI.

Streamlit re-executes the script on every interaction, so a connection cannot
live in a local variable. This client owns a private asyncio loop on a daemon
thread, keeps the socket open across reruns (the UI stores one instance in
``st.session_state``) and exposes a cheap, thread-safe ``snapshot()`` that the
page renders.

It speaks the same wire protocol as the Angular client and
``scripts/voice_client.py`` and de
liberately imports nothing from ``app``: the
UI exercises the server exactly as an outside client would.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

MAX_MESSAGE_BYTES = 16 * 1024 * 1024


@dataclass(slots=True)
class AudioItem:
    """A binary frame together with the header that described it."""

    index: int
    at: float
    header: dict[str, Any]
    data: bytes

    @property
    def label(self) -> str:
        prompt = self.header.get("prompt") or self.header.get("type", "audio")
        language = self.header.get("language")
        return f"{prompt}" + (f" · {language}" if language else "")

    @property
    def duration(self) -> float:
        return float(self.header.get("duration_seconds") or 0.0)


@dataclass(slots=True)
class Entry:
    """One line of the timeline: a server event, a client action or audio."""

    at: float  # seconds since the call started
    source: str  # "server" | "client"
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    audio: AudioItem | None = None


@dataclass(slots=True)
class Snapshot:
    """Immutable-enough view of the call for one render pass."""

    connected: bool
    closed: bool
    close_code: int | None
    close_reason: str
    connect_error: str | None
    session_id: str | None
    state: str
    entries: list[Entry]
    audio: list[AudioItem]
    started_at: float | None

    @property
    def elapsed(self) -> float:
        return 0.0 if self.started_at is None else time.monotonic() - self.started_at


class VoiceSessionClient:
    def __init__(self, url: str, *, origin: str | None = None) -> None:
        self.url = url
        self._origin = origin
        self._lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="voice-ui-ws", daemon=True
        )
        self._socket: Any = None
        self._runner: Future[None] | None = None

        self._connected = False
        self._closed = False
        self._close_code: int | None = None
        self._close_reason = ""
        self._connect_error: str | None = None
        self._session_id: str | None = None
        self._state = "connecting"
        self._entries: list[Entry] = []
        self._audio: list[AudioItem] = []
        self._pending_header: dict[str, Any] | None = None
        self._started_at: float | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._thread.start()
        self._runner = asyncio.run_coroutine_threadsafe(self._session(), self._loop)

    def close(self) -> None:
        if self._loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._connected and not self._closed

    # ------------------------------------------------------------------ #
    # Caller actions
    # ------------------------------------------------------------------ #
    def send_json(self, message: dict[str, Any], *, note: str | None = None) -> None:
        self._record_client(message["type"], {"note": note, **message})
        self._submit(self._send_text(json.dumps(message, ensure_ascii=False)))

    def send_audio(self, data: bytes, *, seconds: float | None = None) -> None:
        self._record_client(
            "audio", {"bytes": len(data), "seconds": seconds, "note": "recording sent"}
        )
        self._submit(self._send_bytes(data))

    def select_language(self, language: str) -> None:
        self.send_json({"type": "select_language", "language": language})

    def say(self, text: str) -> None:
        self.send_json({"type": "text", "text": text})

    def reset(self) -> None:
        self.send_json({"type": "reset"})

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(
                connected=self._connected,
                closed=self._closed,
                close_code=self._close_code,
                close_reason=self._close_reason,
                connect_error=self._connect_error,
                session_id=self._session_id,
                state=self._state,
                entries=list(self._entries),
                audio=list(self._audio),
                started_at=self._started_at,
            )

    # ------------------------------------------------------------------ #
    # Internals (event-loop thread)
    # ------------------------------------------------------------------ #
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coroutine) -> None:
        if self._loop.is_closed():
            coroutine.close()
            return
        asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    async def _send_text(self, payload: str) -> None:
        if self._socket is not None and not self._closed:
            try:
                await self._socket.send(payload)
            except ConnectionClosed:
                pass

    async def _send_bytes(self, payload: bytes) -> None:
        if self._socket is not None and not self._closed:
            try:
                await self._socket.send(payload)
            except ConnectionClosed:
                pass

    async def _shutdown(self) -> None:
        if self._socket is not None and not self._closed:
            try:
                await self._socket.close(1000, "ui closed")
            except Exception:
                pass

    async def _session(self) -> None:
        headers = {"Origin": self._origin} if self._origin else None
        try:
            async with connect(
                self.url,
                additional_headers=headers,
                open_timeout=10,
                max_size=MAX_MESSAGE_BYTES,
                ping_interval=20,
                ping_timeout=60,
            ) as socket:
                self._socket = socket
                with self._lock:
                    self._connected = True
                    self._started_at = time.monotonic()
                await self._read(socket)
        except InvalidStatus as exc:
            self._fail(
                f"Server refused the connection (HTTP {exc.response.status_code})."
            )
        except (OSError, TimeoutError) as exc:
            self._fail(f"Could not reach {self.url}: {exc}")
        except WebSocketException as exc:
            self._fail(f"WebSocket error: {exc}")
        finally:
            with self._lock:
                self._closed = True
                self._connected = False

    async def _read(self, socket) -> None:
        try:
            async for message in socket:
                if isinstance(message, bytes | bytearray):
                    self._on_binary(bytes(message))
                else:
                    self._on_text(message)
        except ConnectionClosed as closed:
            frame = closed.rcvd
            with self._lock:
                self._close_code = frame.code if frame else None
                self._close_reason = (frame.reason if frame else "") or ""
            return

        frame = getattr(socket, "close_code", None)
        with self._lock:
            self._close_code = frame
            self._close_reason = getattr(socket, "close_reason", "") or ""

    def _on_text(self, raw: str) -> None:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            event = {"type": "invalid_json", "raw": raw[:200]}

        kind = event.get("type", "unknown")

        if kind in ("prompt.audio", "agent.audio"):
            self._pending_header = event
            return

        with self._lock:
            if kind == "session.created":
                self._session_id = event.get("session_id")
            elif kind == "state.changed":
                self._state = event.get("state", self._state)
            elif kind == "language.selected":
                self._state = event.get("state", self._state)
            elif kind in ("language.failed", "session.closed"):
                self._state = "closed"
            self._entries.append(Entry(self._elapsed(), "server", kind, event))

    def _on_binary(self, data: bytes) -> None:
        header = self._pending_header or {"type": "audio"}
        self._pending_header = None

        with self._lock:
            item = AudioItem(len(self._audio), self._elapsed(), header, data)
            self._audio.append(item)
            self._entries.append(
                Entry(item.at, "server", header.get("type", "audio"), header, item)
            )

    def _record_client(self, kind: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._entries.append(Entry(self._elapsed(), "client", kind, payload))

    def _fail(self, message: str) -> None:
        with self._lock:
            self._connect_error = message

    def _elapsed(self) -> float:
        return 0.0 if self._started_at is None else time.monotonic() - self._started_at


def http_base(ws_url: str) -> str:
    """``ws://host:8000/api/v1/voice`` -> ``http://host:8000``."""

    scheme, _, rest = ws_url.partition("://")
    host = rest.split("/", 1)[0]
    return f"{'https' if scheme == 'wss' else 'http'}://{host}"

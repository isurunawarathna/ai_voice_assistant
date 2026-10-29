"""Starlette WebSocket implementation of :class:`VoiceTransport`."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.conversation.transport import InboundFrame, VoiceTransport
from app.domain.models import AudioClip
from app.domain.protocol import AudioHeader, OutboundEvent, inbound_adapter

logger = logging.getLogger(__name__)


class WebSocketTransport(VoiceTransport):
    def __init__(self, websocket: WebSocket, *, max_control_chars: int = 8_192) -> None:
        self._ws = websocket
        self._max_control_chars = max_control_chars
        # Serialises writes so an audio header and its binary frame always
        # arrive back to back, even if something else tries to send.
        self._send_lock = asyncio.Lock()
        self._closed = False

    @property
    def peer(self) -> str:
        client = self._ws.client
        return f"{client.host}:{client.port}" if client else "unknown"

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def send_event(self, event: OutboundEvent) -> None:
        await self._send_text(event.model_dump_json(exclude_none=True))

    async def send_clip(self, header: AudioHeader, clip: AudioClip) -> None:
        if self._closed:
            return

        async with self._send_lock:
            try:
                await self._ws.send_text(header.model_dump_json(exclude_none=True))
                await self._ws.send_bytes(clip.data)
            except (WebSocketDisconnect, RuntimeError) as exc:
                self._closed = True
                logger.debug("Dropped audio frame, socket closed: %s", exc)

    async def receive(self) -> InboundFrame:
        try:
            message = await self._ws.receive()
        except (WebSocketDisconnect, RuntimeError):
            self._closed = True
            return InboundFrame.disconnect_frame()

        if message["type"] == "websocket.disconnect":
            self._closed = True
            return InboundFrame.disconnect_frame()

        payload = message.get("bytes")
        if payload is not None:
            return InboundFrame.audio_frame(payload)

        text = message.get("text")
        if text is None:
            return InboundFrame.invalid_frame("Empty frame received.")

        if len(text) > self._max_control_chars:
            return InboundFrame.invalid_frame("Control message is too large.")

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return InboundFrame.invalid_frame(
                "Text frames must be JSON control messages."
            )

        try:
            return InboundFrame.control_frame(inbound_adapter.validate_python(data))
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else {}
            return InboundFrame.invalid_frame(
                f"Unsupported control message: {first.get('msg', 'invalid payload')}"
            )

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self._closed:
            return

        self._closed = True
        try:
            await self._ws.close(code=code, reason=reason)
        except (WebSocketDisconnect, RuntimeError) as exc:
            logger.debug("Close on an already-closed socket: %s", exc)

    async def _send_text(self, payload: str) -> None:
        if self._closed:
            return

        async with self._send_lock:
            try:
                await self._ws.send_text(payload)
            except (WebSocketDisconnect, RuntimeError) as exc:
                self._closed = True
                logger.debug("Dropped event, socket closed: %s", exc)

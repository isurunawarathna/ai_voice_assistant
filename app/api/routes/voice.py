"""Voice WebSocket endpoint.

The route is deliberately thin: accept the socket, wrap it in a transport and
hand it to the orchestrator. All conversation logic lives in
:mod:`app.conversation.orchestrator`.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket

from app.api.deps import ContainerDep, SettingsDep, WSContainerDep
from app.api.ws.channel import WebSocketTransport
from app.api.ws.guard import Admission
from app.domain.enums import Language
from app.domain.protocol import PROTOCOL_VERSION

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Voice Assistant"])

WS_POLICY_VIOLATION = 1008
WS_SERVICE_UNAVAILABLE = 1013


@router.websocket("/voice")
async def voice_websocket(websocket: WebSocket, container: WSContainerDep) -> None:
    # Admission runs before accept(): a refused handshake is answered with an
    # HTTP 403 and never streams the welcome audio.
    decision = container.guard.check(websocket)
    container.metrics.connections.inc(result=decision.admission.value)

    if not decision.allowed:
        logger.warning(
            "Refused WebSocket",
            extra={
                "reason": decision.admission.value,
                "client_ip": decision.client_ip,
                "origin": websocket.headers.get("origin"),
            },
        )
        code = (
            WS_POLICY_VIOLATION
            if decision.admission is Admission.ORIGIN
            else WS_SERVICE_UNAVAILABLE
        )
        await websocket.close(code, decision.admission.value)
        return

    try:
        await websocket.accept()

        try:
            orchestrator = container.require_orchestrator()
        except RuntimeError:
            logger.error("Rejected WebSocket: service not ready")
            await websocket.close(WS_SERVICE_UNAVAILABLE, "service starting")
            return

        transport = WebSocketTransport(websocket)

        try:
            await orchestrator.run(transport)
        finally:
            await transport.close()
    finally:
        container.guard.release(decision.client_ip)


@router.get("/voice/status")
async def voice_status(
    settings: SettingsDep, container: ContainerDep
) -> dict[str, object]:
    return {
        "status": "online" if container.orchestrator is not None else "starting",
        "websocket_url": f"{settings.api_prefix}/voice",
        "protocol_version": PROTOCOL_VERSION,
        "supported_languages": [
            {"tag": language.value, "name": language.display_name}
            for language in Language
        ],
        "active_sessions": await container.sessions.active_count(),
        "limits": {
            "max_audio_bytes": settings.audio.max_bytes,
            "max_audio_seconds": settings.audio.max_seconds,
            "target_sample_rate": settings.audio.target_sample_rate,
            "language_max_retries": settings.conversation.language_max_retries,
            "idle_timeout_seconds": settings.conversation.idle_timeout_seconds,
            "language_selection_deadline_seconds": (
                settings.conversation.language_selection_deadline_seconds
            ),
            "max_invalid_inputs": settings.conversation.max_invalid_inputs,
            "max_resets": settings.conversation.max_resets,
        },
    }

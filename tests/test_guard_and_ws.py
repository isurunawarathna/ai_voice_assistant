"""Admission control and the real WebSocket route."""

from __future__ import annotations

import json

import httpx
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.ws.guard import Admission, ConnectionGuard
from app.conversation.orchestrator import ConversationOrchestrator
from app.core.config import Environment, Settings
from app.main import create_app
from app.services.language import LanguageDetector
from app.services.tts.base import NullTextToSpeech
from tests.conftest import ScriptedSTT, StubAgent, StubPrompts


class FakeSocket:
    def __init__(self, origin: str | None = None, ip: str = "10.0.0.1", xff: str = ""):
        self.headers = {}
        if origin is not None:
            self.headers["origin"] = origin
        if xff:
            self.headers["x-forwarded-for"] = xff
        self.client = type("Client", (), {"host": ip, "port": 1})()


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make_settings(**security) -> Settings:
    settings = Settings(environment=Environment.PRODUCTION)
    settings.cors_origins = ["https://campus.example"]
    for key, value in security.items():
        setattr(settings.security, key, value)
    return settings


# --------------------------------------------------------------------------- #
# ConnectionGuard
# --------------------------------------------------------------------------- #
def test_browser_origin_must_be_allowed_in_production() -> None:
    guard = ConnectionGuard(make_settings())

    assert guard.check(FakeSocket("https://campus.example")).allowed
    assert guard.check(FakeSocket("https://campus.example/")).allowed
    assert guard.check(FakeSocket("https://evil.example")).admission is Admission.ORIGIN


def test_clients_without_an_origin_header_are_not_browsers() -> None:
    assert ConnectionGuard(make_settings()).check(FakeSocket(None)).allowed


def test_origin_is_not_enforced_locally() -> None:
    settings = Settings(environment=Environment.LOCAL)

    assert ConnectionGuard(settings).check(FakeSocket("http://anything")).allowed


def test_origin_enforcement_can_be_forced_either_way() -> None:
    local = Settings(environment=Environment.LOCAL)
    local.security.enforce_origin = True
    assert not ConnectionGuard(local).check(FakeSocket("http://anything")).allowed

    prod = make_settings(enforce_origin=False)
    assert ConnectionGuard(prod).check(FakeSocket("https://evil.example")).allowed


def test_security_allowed_origins_override_cors_origins() -> None:
    guard = ConnectionGuard(make_settings(allowed_origins=["https://kiosk.example"]))

    assert guard.check(FakeSocket("https://kiosk.example")).allowed
    assert not guard.check(FakeSocket("https://campus.example")).allowed


def test_concurrent_connections_per_ip_are_capped_and_released() -> None:
    guard = ConnectionGuard(
        make_settings(max_connections_per_ip=2, max_connects_per_minute_per_ip=0)
    )

    assert guard.check(FakeSocket()).allowed
    assert guard.check(FakeSocket()).allowed
    assert guard.check(FakeSocket()).admission is Admission.CONCURRENCY
    assert guard.check(FakeSocket(ip="10.0.0.2")).allowed  # other clients unaffected

    guard.release("10.0.0.1")
    assert guard.check(FakeSocket()).allowed


def test_connection_rate_window_slides() -> None:
    clock = Clock()
    guard = ConnectionGuard(
        make_settings(max_connections_per_ip=0, max_connects_per_minute_per_ip=3),
        clock=clock,
    )

    for _ in range(3):
        assert guard.check(FakeSocket()).allowed
        guard.release("10.0.0.1")

    assert guard.check(FakeSocket()).admission is Admission.RATE_LIMIT

    clock.now = 61.0
    assert guard.check(FakeSocket()).allowed


def test_forwarded_for_is_ignored_unless_trusted() -> None:
    socket = FakeSocket(ip="10.0.0.1", xff="203.0.113.9, 10.0.0.1")

    assert ConnectionGuard(make_settings()).client_ip(socket) == "10.0.0.1"
    trusted = ConnectionGuard(make_settings(trust_forwarded_for=True))
    assert trusted.client_ip(socket) == "203.0.113.9"


def test_release_never_goes_negative() -> None:
    guard = ConnectionGuard(make_settings())

    guard.release("10.0.0.1")

    assert guard.open_connections == 0


# --------------------------------------------------------------------------- #
# The real route, driven through Starlette's TestClient
# --------------------------------------------------------------------------- #
def build_app(settings: Settings, stt: ScriptedSTT | None = None):
    """The real FastAPI app with the lifespan bypassed and fakes injected."""

    settings.session.sweep_interval_seconds = 0
    app = create_app(settings)
    container = app.state.container
    container.orchestrator = ConversationOrchestrator(
        settings=settings,
        sessions=container.sessions,
        prompts=StubPrompts(settings.audio),
        stt=stt or ScriptedSTT(),
        agent=StubAgent(),
        tts=NullTextToSpeech(),
        language_detector=LanguageDetector(),
        metrics=container.metrics,
    )
    return app


def test_websocket_greets_then_accepts_a_language_choice() -> None:
    app = build_app(Settings())
    client = TestClient(app)

    with client.websocket_connect("/api/v1/voice") as ws:
        seen: list[str] = []
        while True:
            message = ws.receive()
            if message.get("bytes") is not None:
                seen.append("<audio>")
                continue
            event = json.loads(message["text"])
            seen.append(event["type"])
            if (
                event["type"] == "state.changed"
                and event["state"] == "language_selection"
            ):
                break

        assert seen == [
            "session.created",
            "state.changed",
            "prompt.audio",
            "<audio>",
            "state.changed",
        ]

        ws.send_json({"type": "select_language", "language": "en-US"})
        selected = ws.receive_json()
        assert selected["type"] == "language.selected"
        assert selected["method"] == "ui"
        assert selected["language"] == "en-US"


def test_websocket_from_a_foreign_browser_origin_is_refused() -> None:
    settings = make_settings()
    app = build_app(settings)
    client = TestClient(app)

    with pytest.raises(WebSocketDisconnect) as refused:
        with client.websocket_connect(
            "/api/v1/voice", headers={"origin": "https://evil.example"}
        ):
            pass

    assert refused.value.code == 1008
    assert app.state.container.metrics.connections.value(result="rejected_origin") == 1
    assert app.state.container.guard.open_connections == 0


def test_connection_slot_is_released_when_the_socket_closes() -> None:
    settings = make_settings(max_connections_per_ip=1, max_connects_per_minute_per_ip=0)
    settings.security.enforce_origin = False
    app = build_app(settings)
    client = TestClient(app)

    for _ in range(3):  # would fail on the 2nd if the slot leaked
        with client.websocket_connect("/api/v1/voice") as ws:
            assert ws.receive_json()["type"] == "session.created"

    assert app.state.container.guard.open_connections == 0


# --------------------------------------------------------------------------- #
# /metrics
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_metrics_endpoint_serves_prometheus_text() -> None:
    app = build_app(Settings())
    app.state.container.metrics.connections.inc(result="accepted")
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert 'voice_connections_total{result="accepted"} 1' in response.text


@pytest.mark.asyncio
async def test_metrics_endpoint_can_be_disabled() -> None:
    settings = Settings()
    settings.metrics_enabled = False
    transport = httpx.ASGITransport(app=build_app(settings))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/metrics")).status_code == 404


# --------------------------------------------------------------------------- #
# Configuration guard rails
# --------------------------------------------------------------------------- #
def test_production_refuses_lenient_prompt_loading() -> None:
    with pytest.raises(ValueError, match="STRICT_PROMPT_LOADING"):
        Settings(
            environment=Environment.PRODUCTION,
            audio={"strict_prompt_loading": False},
        )


def test_production_requires_sane_selection_timeouts() -> None:
    with pytest.raises(ValueError, match="DEADLINE"):
        Settings(
            environment=Environment.PRODUCTION,
            conversation={
                "language_selection_timeout_seconds": 400,
                "language_selection_deadline_seconds": 300,
            },
        )


def test_conversation_limits_are_validated() -> None:
    with pytest.raises(ValueError):
        Settings(conversation={"language_max_retries": 0})

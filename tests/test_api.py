"""HTTP surface tests.

These run without the lifespan so no Whisper weights are loaded: the container
is deliberately un-started, which is also the state a readiness probe must
report as unavailable.
"""

from __future__ import annotations

import httpx
import pytest

from app.core.config import Settings
from app.main import create_app

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def client():
    app = create_app(Settings())
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_health_reports_service_metadata(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["version"]
    assert "uptime_seconds" in body


async def test_liveness_never_touches_dependencies(client: httpx.AsyncClient) -> None:
    response = await client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


async def test_readiness_is_503_before_startup(client: httpx.AsyncClient) -> None:
    response = await client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert set(body["components"]) == {"prompts", "stt", "tts", "sessions", "agent"}


async def test_voice_status_publishes_the_contract(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/voice/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "starting"
    assert body["websocket_url"] == "/api/v1/voice"
    assert body["protocol_version"] == 1
    assert [item["tag"] for item in body["supported_languages"]] == ["si-LK", "en-US"]
    assert body["limits"]["target_sample_rate"] == 16_000


async def test_trace_id_is_echoed_back(client: httpx.AsyncClient) -> None:
    response = await client.get("/health/live", headers={"X-Trace-Id": "abc123"})

    assert response.headers["X-Trace-Id"] == "abc123"

from __future__ import annotations

import pytest

from app.core.config import SessionSettings
from app.core.errors import CapacityExceededError
from app.domain.enums import ConversationState, Language, Speaker
from app.session.memory_store import InMemorySessionStore

pytestmark = pytest.mark.asyncio


async def test_create_get_and_delete() -> None:
    store = InMemorySessionStore(SessionSettings(sweep_interval_seconds=0))

    session = await store.create()
    assert await store.get(session.session_id) is session
    assert await store.active_count() == 1

    await store.delete(session.session_id)
    assert await store.get(session.session_id) is None
    assert await store.active_count() == 0


async def test_delete_is_idempotent() -> None:
    store = InMemorySessionStore(SessionSettings(sweep_interval_seconds=0))

    await store.delete("does-not-exist")


async def test_save_persists_mutations() -> None:
    store = InMemorySessionStore(SessionSettings(sweep_interval_seconds=0))
    session = await store.create()

    session.language = Language.SINHALA
    session.transition(ConversationState.ACTIVE)
    await store.save(session)

    stored = await store.get(session.session_id)
    assert stored is not None
    assert stored.language is Language.SINHALA
    assert stored.state is ConversationState.ACTIVE


async def test_expired_sessions_are_dropped() -> None:
    store = InMemorySessionStore(
        SessionSettings(ttl_seconds=0, sweep_interval_seconds=0)
    )
    session = await store.create()

    assert await store.get(session.session_id) is None
    assert await store.active_count() == 0


async def test_capacity_is_enforced() -> None:
    store = InMemorySessionStore(
        SessionSettings(max_active_sessions=2, sweep_interval_seconds=0)
    )

    await store.create()
    await store.create()

    with pytest.raises(CapacityExceededError):
        await store.create()


async def test_history_is_bounded() -> None:
    store = InMemorySessionStore(SessionSettings(sweep_interval_seconds=0))
    session = await store.create()

    for index in range(10):
        session.add_turn(Speaker.USER, f"question {index}", max_turns=4)

    assert len(session.turns) == 4
    assert session.turns[0].text == "question 6"
    assert session.turns[-1].text == "question 9"


async def test_health_report() -> None:
    store = InMemorySessionStore(SessionSettings(sweep_interval_seconds=0))
    await store.create()

    report = await store.health()
    assert report == {"name": "memory", "ready": True, "active": 1}

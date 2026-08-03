"""Memory TTL — "for how long" a memory is kept, and whether it's still valid.

The extraction model tags time-sensitive memories with ``ttl_days`` (permanent =
omit, recurring habit ≈ 30, time-bound note ≈ 7). The callback converts that to an
``expires_at`` via ``_expires_at_from_ttl``, and reads drop memories past their
window. These tests pin that conversion + the expiry contract deterministically —
no model needed. The MODEL'S judgment about *which* ttl to assign is covered
separately by the extraction eval (tools/memory_extraction_eval.py).
"""
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base
from app.services.memory_service import MemoryService
from app.services.memory_extraction_service import _expires_at_from_ttl


class TestTtlToExpiresAt:
    def test_none_is_permanent(self):
        assert _expires_at_from_ttl(None) is None

    def test_time_bound_seven_days(self):
        exp = _expires_at_from_ttl(7)
        assert exp is not None and timedelta(days=6, hours=23) < (exp - datetime.utcnow()) <= timedelta(days=7)

    def test_recurring_thirty_days(self):
        exp = _expires_at_from_ttl(30)
        assert exp is not None and timedelta(days=29, hours=23) < (exp - datetime.utcnow()) <= timedelta(days=30)

    def test_int_like_string_is_coerced(self):
        exp = _expires_at_from_ttl("5")
        assert exp is not None and timedelta(days=4, hours=23) < (exp - datetime.utcnow()) <= timedelta(days=5)

    def test_unparseable_is_permanent_fail_safe(self):
        # A bad value must NOT expire the memory instantly — keep it (permanent).
        assert _expires_at_from_ttl("soon") is None
        assert _expires_at_from_ttl([1, 2]) is None

    def test_zero_expires_immediately(self):
        exp = _expires_at_from_ttl(0)
        assert exp is not None and abs((exp - datetime.utcnow()).total_seconds()) < 5


@pytest.fixture()
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    s = sessionmaker(bind=engine)()
    try:
        yield s
    finally:
        s.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def service(db):
    return MemoryService(db)


class TestTtlExpiryContract:
    """Ties the ttl window to what reads actually return."""

    def test_permanent_memory_is_always_returned(self, service):
        service.save_memory(1, "h1", "name is Jordan", expires_at=_expires_at_from_ttl(None))
        assert any("Jordan" in m.content for m in service.get_active_memories(1, "h1"))

    def test_time_bound_memory_is_live_within_its_window(self, service):
        service.save_memory(1, "h1", "dentist appointment Friday", expires_at=_expires_at_from_ttl(7))
        assert any("dentist" in m.content for m in service.get_active_memories(1, "h1"))

    def test_memory_past_its_window_drops_out(self, service, db):
        m = service.save_memory(1, "h1", "grilling steaks tonight", expires_at=_expires_at_from_ttl(7))
        m.expires_at = datetime.utcnow() - timedelta(hours=1)  # fast-forward past the window
        db.commit()
        assert service.get_active_memories(1, "h1") == []

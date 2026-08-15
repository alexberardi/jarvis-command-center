"""Unit tests for SignalService — the Signal Bus store (upsert, TTL, cacheable gate).

Runs on in-memory SQLite (the ``signals`` table has no Vector column), mirroring
tests/test_memory_service.py. This file is the executable spec for Phase 1's
storage layer: a self-describing Signal, UPSERT-on-``source_key``, TTL cleanup,
and the cacheable write-boundary rule that protects the prefix cache / TTFS.
"""
import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Signal
from app.services.signal_service import SignalService, CacheableSignalError


@pytest.fixture()
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def service(db):
    return SignalService(db)


class TestSaveSignal:
    def test_save_basic(self, service, db):
        sig = service.save_signal(
            household_id="h1", source_key="presence:alex", kind="presence.seen",
            subject="user:alex", summary="Alex is home",
            facts={"user": "alex", "room": "kitchen"}, user_id=7, ttl_seconds=900,
        )
        assert sig.id is not None
        assert sig.household_id == "h1"
        assert sig.source_key == "presence:alex"
        assert sig.kind == "presence.seen"
        assert sig.subject == "user:alex"
        assert sig.user_id == 7
        assert sig.is_active is True
        # facts round-trips as JSON
        assert json.loads(sig.facts) == {"user": "alex", "room": "kitchen"}
        # expires ~ now + ttl
        assert sig.expires_at is not None
        delta = (sig.expires_at - datetime.utcnow()).total_seconds()
        assert 800 < delta <= 901

    def test_no_ttl_never_expires(self, service):
        sig = service.save_signal(household_id="h1", source_key="k", kind="x")
        assert sig.expires_at is None

    def test_household_wide_null_user(self, service):
        sig = service.save_signal(household_id="h1", source_key="k", kind="x", user_id=None)
        assert sig.user_id is None


class TestUpsert:
    def test_same_source_key_updates_in_place(self, service, db):
        a = service.save_signal(household_id="h1", source_key="presence:alex",
                                kind="presence.seen", summary="first")
        b = service.save_signal(household_id="h1", source_key="presence:alex",
                                kind="presence.seen", summary="second")
        assert db.query(Signal).count() == 1          # one row, not two — the flap collapsed
        assert a.id == b.id
        assert b.summary == "second"

    def test_scoped_per_household(self, service, db):
        service.save_signal(household_id="h1", source_key="presence:alex", kind="presence.seen")
        service.save_signal(household_id="h2", source_key="presence:alex", kind="presence.seen")
        assert db.query(Signal).count() == 2          # same source_key, different households


class TestCleanup:
    def test_cleanup_removes_only_expired(self, service, db):
        now = datetime.utcnow()
        db.add(Signal(household_id="h1", source_key="expired", kind="x",
                      expires_at=now - timedelta(seconds=1), is_active=True))
        db.add(Signal(household_id="h1", source_key="future", kind="x",
                      expires_at=now + timedelta(hours=1), is_active=True))
        db.add(Signal(household_id="h1", source_key="forever", kind="x",
                      expires_at=None, is_active=True))
        db.commit()

        removed = service.cleanup_expired()

        assert removed == 1
        remaining = {s.source_key for s in db.query(Signal).all()}
        assert remaining == {"future", "forever"}      # NULL-expiry never swept


class TestCacheableWriteBoundary:
    """cacheable=True Signals may ride the byte-stable prefix, so they must be
    quantized/absolute. A live float / relative time / seconds-precision stamp
    would poison the prefix cache and blow TTFS — reject at the write boundary."""

    def test_rejects_live_float(self, service):
        with pytest.raises(CacheableSignalError):
            service.save_signal(household_id="h1", source_key="w", kind="weather",
                                cacheable=True, summary="It's 72.5 degrees right now")

    def test_rejects_relative_time(self, service):
        with pytest.raises(CacheableSignalError):
            service.save_signal(household_id="h1", source_key="p", kind="presence",
                                cacheable=True, summary="Alex left 3 minutes ago")

    def test_rejects_seconds_precision_in_facts(self, service):
        with pytest.raises(CacheableSignalError):
            service.save_signal(household_id="h1", source_key="p", kind="presence",
                                cacheable=True, summary="arrived", facts={"at": "18:42:07"})

    def test_allows_quantized_cacheable(self, service):
        sig = service.save_signal(household_id="h1", source_key="home",
                                  kind="presence.home_base", cacheable=True,
                                  summary="Alex has a home base configured",
                                  facts={"state": "configured"})
        assert sig.cacheable is True

    def test_non_cacheable_allows_anything(self, service):
        # the boundary only applies to cacheable=True; transient signals ride the
        # trailing block and may carry live values.
        sig = service.save_signal(household_id="h1", source_key="w", kind="weather",
                                  cacheable=False, summary="It's 72.5 degrees right now")
        assert sig.cacheable is False

"""Tests for TranscriptService rating API (Phase 1)."""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base, ConversationTranscript
from app.services.transcript_service import TranscriptService


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
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
    return TranscriptService(db)


def _log(service, *, user_id=1, household_id="h1", message="hi", tool_calls=None):
    return service.log_transcript(
        user_id=user_id,
        household_id=household_id,
        conversation_id="c1",
        user_message=message,
        assistant_message="ok",
        tool_calls=tool_calls,
    )


class TestListRecentForUser:
    def test_returns_newest_first(self, service, db):
        a = _log(service, message="first")
        # Backdate to make ordering deterministic
        a.created_at = datetime.utcnow() - timedelta(minutes=5)
        db.commit()
        b = _log(service, message="second")
        rows = service.list_recent_for_user(user_id=1)
        assert [r.user_message for r in rows] == ["second", "first"]

    def test_filters_by_user(self, service):
        _log(service, user_id=1, message="alice")
        _log(service, user_id=2, message="bob")
        rows = service.list_recent_for_user(user_id=1)
        assert [r.user_message for r in rows] == ["alice"]

    def test_respects_limit(self, service):
        for i in range(5):
            _log(service, message=f"msg{i}")
        rows = service.list_recent_for_user(user_id=1, limit=3)
        assert len(rows) == 3

    def test_since_filter(self, service, db):
        old = _log(service, message="old")
        old.created_at = datetime.utcnow() - timedelta(days=2)
        db.commit()
        _log(service, message="new")
        cutoff = datetime.utcnow() - timedelta(days=1)
        rows = service.list_recent_for_user(user_id=1, since=cutoff)
        assert [r.user_message for r in rows] == ["new"]


class TestRateTranscript:
    def test_sets_rating_and_timestamp(self, service):
        row = _log(service, message="rate me")
        updated = service.rate_transcript(row.id, user_id=1, rating=1, notes="good parse")
        assert updated is not None
        assert updated.user_rating == 1
        assert updated.rating_notes == "good parse"
        assert updated.rated_at is not None

    def test_negative_rating(self, service):
        row = _log(service, message="bad parse")
        updated = service.rate_transcript(row.id, user_id=1, rating=-1)
        assert updated is not None
        assert updated.user_rating == -1

    def test_overwrites_previous_rating(self, service):
        row = _log(service, message="change mind")
        service.rate_transcript(row.id, user_id=1, rating=1)
        updated = service.rate_transcript(row.id, user_id=1, rating=-1, notes="actually wrong")
        assert updated is not None
        assert updated.user_rating == -1
        assert updated.rating_notes == "actually wrong"

    def test_returns_none_for_other_user(self, service):
        row = _log(service, user_id=1, message="alice's row")
        updated = service.rate_transcript(row.id, user_id=2, rating=1)
        assert updated is None

    def test_returns_none_for_missing_row(self, service):
        updated = service.rate_transcript(999_999, user_id=1, rating=1)
        assert updated is None

    def test_rejects_invalid_rating(self, service):
        row = _log(service, message="x")
        with pytest.raises(ValueError):
            service.rate_transcript(row.id, user_id=1, rating=5)

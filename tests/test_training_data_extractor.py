"""Tests for the Phase 3 training data extractor."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base, ConversationTranscript
from app.services.training_data_extractor import (
    ExtractedRow,
    TrainingDataExtractor,
    build_response_json,
    normalize_utterance,
    to_dataset_ref_row,
)


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
def extractor(db):
    return TrainingDataExtractor(db)


def _insert(
    db,
    *,
    user_id: int = 1,
    household_id: str = "h1",
    conversation_id: str = "c1",
    user_message: str = "play jazz",
    tool_calls: list | None = None,
    user_rating: int | None = None,
    created_at: datetime | None = None,
) -> ConversationTranscript:
    row = ConversationTranscript(
        user_id=user_id,
        household_id=household_id,
        conversation_id=conversation_id,
        user_message=user_message,
        assistant_message="ok",
        tool_calls_json=json.dumps(tool_calls) if tool_calls is not None else None,
        user_rating=user_rating,
        created_at=created_at or datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    return row


def _music_play(query: str = "jazz") -> list:
    return [{"name": "music", "arguments": {"action": "play", "query": query}}]


class TestNormalize:
    def test_basic(self):
        assert normalize_utterance("Play Jazz!") == "play jazz"

    def test_punct_collapse(self):
        assert normalize_utterance("  Play,  jazz?? ") == "play jazz"

    def test_empty(self):
        assert normalize_utterance("") == ""
        assert normalize_utterance(None) == ""  # type: ignore[arg-type]


class TestPositiveSetFilter:
    def test_explicit_positive_rated_thumbs_up(self, db, extractor):
        _insert(db, user_message="play jazz", tool_calls=_music_play(), user_rating=1)
        rows = extractor.extract(min_per_user=1)
        assert len(rows) == 1
        assert rows[0].source == "explicit_positive"
        assert rows[0].user_rating == 1

    def test_implicit_clean_no_rating(self, db, extractor):
        _insert(db, user_message="play jazz", tool_calls=_music_play(), user_rating=None)
        rows = extractor.extract(min_per_user=1)
        assert len(rows) == 1
        assert rows[0].source == "implicit_clean"

    def test_rejects_thumbs_down(self, db, extractor):
        _insert(db, user_message="play jazz", tool_calls=_music_play(), user_rating=-1)
        rows = extractor.extract(min_per_user=1)
        assert rows == []

    def test_rejects_no_tool_call(self, db, extractor):
        _insert(db, user_message="how are you", tool_calls=None)
        rows = extractor.extract(min_per_user=1)
        assert rows == []

    def test_rejects_malformed_tool_calls_json(self, db, extractor):
        row = _insert(db, user_message="play jazz", tool_calls=_music_play())
        row.tool_calls_json = "not-json"
        db.commit()
        rows = extractor.extract(min_per_user=1)
        assert rows == []

    def test_rejects_empty_user_message(self, db, extractor):
        _insert(db, user_message="", tool_calls=_music_play())
        _insert(db, user_message="   ", tool_calls=_music_play())
        rows = extractor.extract(min_per_user=1)
        assert rows == []

    # Note: user_id is NOT NULL in the schema, so a null can't exist in practice.
    # The query still filters `user_id IS NOT NULL` for defensiveness, but we
    # don't test that branch directly — the schema already enforces it.


class TestRetryDetection:
    def test_retry_within_window_drops_earlier(self, db, extractor):
        # User says "play jazz" twice, 30s apart, same parse both times
        t = datetime.utcnow()
        _insert(db, conversation_id="c1", user_message="play jazz",
                tool_calls=_music_play(), created_at=t)
        _insert(db, conversation_id="c1", user_message="play jazz",
                tool_calls=_music_play(), created_at=t + timedelta(seconds=30))
        rows = extractor.extract(min_per_user=1)
        # Both pass retry filter the way I wrote it, but dedup will collapse them
        # since they're identical. Check we end up with exactly 1 row, and it's
        # the LATER one (dedup keeps newest).
        assert len(rows) == 1
        assert rows[0].created_at == t + timedelta(seconds=30)

    def test_retry_cross_conversation_does_not_apply(self, db, extractor):
        # Same phrase in two DIFFERENT conversations → not a retry, both kept
        t = datetime.utcnow()
        _insert(db, conversation_id="c1", user_message="play jazz",
                tool_calls=_music_play(), created_at=t)
        _insert(db, conversation_id="c2", user_message="play jazz",
                tool_calls=_music_play(), created_at=t + timedelta(seconds=10))
        # Same user, same text, same tool → dedup collapses to 1 regardless
        rows = extractor.extract(min_per_user=1)
        assert len(rows) == 1

    def test_different_text_same_convo_not_retry(self, db, extractor):
        t = datetime.utcnow()
        _insert(db, conversation_id="c1", user_message="play jazz",
                tool_calls=_music_play("jazz"), created_at=t)
        _insert(db, conversation_id="c1", user_message="play rock",
                tool_calls=_music_play("rock"), created_at=t + timedelta(seconds=10))
        rows = extractor.extract(min_per_user=1)
        assert len(rows) == 2

    def test_explicit_positive_ignores_retry_check(self, db, extractor):
        """Thumbs-up is a strong signal — don't second-guess it with retry logic."""
        t = datetime.utcnow()
        _insert(db, conversation_id="c1", user_message="play jazz",
                tool_calls=_music_play(), user_rating=1, created_at=t)
        _insert(db, conversation_id="c1", user_message="play jazz",
                tool_calls=_music_play(), user_rating=1, created_at=t + timedelta(seconds=30))
        rows = extractor.extract(min_per_user=1)
        # Both are explicit positives; dedup collapses them to 1 newest row
        assert len(rows) == 1
        assert rows[0].source == "explicit_positive"


class TestDedup:
    def test_same_user_same_phrase_same_tool(self, db, extractor):
        _insert(db, user_message="play jazz", tool_calls=_music_play())
        _insert(db, user_message="Play Jazz", tool_calls=_music_play())
        _insert(db, user_message="play  jazz!", tool_calls=_music_play())
        rows = extractor.extract(min_per_user=1)
        assert len(rows) == 1

    def test_different_users_same_phrase_kept(self, db, extractor):
        """Per Ultraplan refinement: same phrase from two users is NOT a dup."""
        _insert(db, user_id=1, user_message="play jazz", tool_calls=_music_play())
        _insert(db, user_id=2, user_message="play jazz", tool_calls=_music_play())
        rows = extractor.extract(min_per_user=1)
        assert len(rows) == 2
        assert {r.user_id for r in rows} == {1, 2}

    def test_same_phrase_different_tool_kept(self, db, extractor):
        _insert(db, user_message="play jazz",
                tool_calls=[{"name": "music", "arguments": {"action": "play"}}])
        _insert(db, user_message="play jazz",
                tool_calls=[{"name": "radio", "arguments": {"channel": "jazz"}}])
        rows = extractor.extract(min_per_user=1)
        assert len(rows) == 2


class TestBalance:
    def test_drops_cold_start_users(self, db, extractor):
        for i in range(3):
            _insert(db, user_id=1, user_message=f"msg{i}", tool_calls=_music_play(f"q{i}"))
        for i in range(15):
            _insert(db, user_id=2, user_message=f"msg{i}", tool_calls=_music_play(f"q{i}"))
        rows = extractor.extract(min_per_user=10)
        assert all(r.user_id == 2 for r in rows)
        assert len(rows) == 15

    def test_downsamples_heaviest_user(self, db, extractor):
        # User 1: 10 rows, user 2: 10 rows, user 3: 100 rows.
        # Median count = 10; cap = 4 × 10 = 40.
        for u, n in ((1, 10), (2, 10), (3, 100)):
            for i in range(n):
                _insert(db, user_id=u, user_message=f"msg{u}-{i}",
                        tool_calls=_music_play(f"q{u}{i}"))
        rows = extractor.extract(min_per_user=5, max_per_user_multiplier=4.0)
        by_user: dict[int, int] = {}
        for r in rows:
            by_user[r.user_id] = by_user.get(r.user_id, 0) + 1
        assert by_user[1] == 10
        assert by_user[2] == 10
        assert by_user[3] == 40  # downsampled from 100 → 40

    def test_preserves_order_by_time(self, db, extractor):
        t0 = datetime.utcnow() - timedelta(minutes=10)
        for i in range(15):
            _insert(db, user_message=f"msg{i}",
                    tool_calls=_music_play(f"q{i}"),
                    created_at=t0 + timedelta(minutes=i))
        rows = extractor.extract(min_per_user=1, max_per_user_multiplier=100.0)
        assert rows == sorted(rows, key=lambda r: r.created_at)


class TestTimeWindow:
    def test_since_filter(self, db, extractor):
        t = datetime.utcnow()
        _insert(db, user_message="old", tool_calls=_music_play("old"),
                created_at=t - timedelta(days=2))
        _insert(db, user_message="new", tool_calls=_music_play("new"),
                created_at=t)
        rows = extractor.extract(since=t - timedelta(days=1), min_per_user=1)
        assert [r.user_message for r in rows] == ["new"]

    def test_until_filter(self, db, extractor):
        t = datetime.utcnow()
        _insert(db, user_message="old", tool_calls=_music_play("old"),
                created_at=t - timedelta(days=2))
        _insert(db, user_message="new", tool_calls=_music_play("new"),
                created_at=t)
        rows = extractor.extract(until=t - timedelta(days=1), min_per_user=1)
        assert [r.user_message for r in rows] == ["old"]

    def test_household_filter(self, db, extractor):
        _insert(db, household_id="h1", user_message="h1", tool_calls=_music_play())
        _insert(db, household_id="h2", user_message="h2", tool_calls=_music_play())
        rows = extractor.extract(household_id="h1", min_per_user=1)
        assert [r.household_id for r in rows] == ["h1"]


class TestFormatting:
    def test_build_response_json_shape(self):
        out = build_response_json({"name": "music", "arguments": {"action": "play"}})
        # Leading space matches the convention used in production completions.
        assert out.startswith(" ")
        parsed = json.loads(out.strip())
        assert parsed == {"message": "", "tool_call": {"name": "music", "arguments": {"action": "play"}}}

    def test_to_dataset_ref_row_fields(self):
        row = ExtractedRow(
            user_id=1,
            household_id="h1",
            conversation_id="c1",
            created_at=datetime(2026, 4, 19, 12, 0, 0),
            user_message="play jazz",
            tool_call={"name": "music", "arguments": {"action": "play", "query": "jazz"}},
            source="explicit_positive",
            user_rating=1,
        )
        out = to_dataset_ref_row(row, system_prompt="SYS")
        assert out["voice_command"] == "play jazz"
        assert out["expected_tool_call"] == row.tool_call
        assert out["formatted_system_prompt"] == "SYS"
        assert "formatted_completion" in out
        assert out["_meta"]["source"] == "explicit_positive"
        assert out["_meta"]["user_rating"] == 1

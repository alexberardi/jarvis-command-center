"""Tests for the characterization synthesis layer (Slice 1).

Covers the pure parsing helpers, the CharacterizationService persistence
(create / version-bump / stale-retirement / scoping), and the synthesis callback
glue. Uses an in-memory SQLite DB (same pattern as test_memory_service) — no
Postgres or LLM needed; the model output is supplied as canned content.
"""

import asyncio
import json
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base, PersonCharacterization
from app.services.characterization_service import CharacterizationService
from app.services import characterization_synthesis_service as css


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine():
    # StaticPool → a single shared in-memory connection, so a session opened
    # inside the callback sees what the test wrote and vice-versa.
    eng = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)


@pytest.fixture()
def db(engine):
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def service(db):
    return CharacterizationService(db)


def _run_callback(engine, payload, monkeypatch):
    """Invoke the async callback with get_session_local pointed at the test engine."""
    Session = sessionmaker(bind=engine)
    import app.db as app_db

    monkeypatch.setattr(app_db, "get_session_local", lambda: Session, raising=False)
    asyncio.run(css.handle_synthesis_callback(payload))
    return Session


# ---------------------------------------------------------------------------
# _parse_synthesis_response
# ---------------------------------------------------------------------------


class TestParseSynthesisResponse:
    def test_raw_json_object(self):
        out = css._parse_synthesis_response('{"summary": "hi", "confidence": 0.5}')
        assert out == {"summary": "hi", "confidence": 0.5}

    def test_markdown_fenced(self):
        assert css._parse_synthesis_response('```json\n{"summary": "hi"}\n```') == {"summary": "hi"}

    def test_think_preamble_stripped(self):
        content = "<think>let me reason about this</think>\n{\"summary\": \"hi\"}"
        assert css._parse_synthesis_response(content) == {"summary": "hi"}

    def test_object_embedded_in_prose(self):
        content = 'Here is the result: {"summary": "hi"} — done.'
        assert css._parse_synthesis_response(content) == {"summary": "hi"}

    def test_json_array_is_rejected(self):
        # Valid JSON, but a characterization must be an object, not a list.
        assert css._parse_synthesis_response("[1, 2, 3]") is None

    def test_garbage_returns_none(self):
        assert css._parse_synthesis_response("not json at all") is None

    def test_empty_returns_none(self):
        assert css._parse_synthesis_response("") is None


# ---------------------------------------------------------------------------
# _parse_iso
# ---------------------------------------------------------------------------


class TestParseIso:
    def test_naive(self):
        dt = css._parse_iso("2026-07-20T10:00:00")
        assert dt == datetime(2026, 7, 20, 10, 0, 0)
        assert dt.tzinfo is None

    def test_tz_aware_converted_to_naive_utc(self):
        # 10:00 at +02:00 == 08:00 UTC, stored naive to match the schema.
        dt = css._parse_iso("2026-07-20T10:00:00+02:00")
        assert dt == datetime(2026, 7, 20, 8, 0, 0)
        assert dt.tzinfo is None

    def test_none(self):
        assert css._parse_iso(None) is None

    def test_garbage(self):
        assert css._parse_iso("not-a-date") is None


# ---------------------------------------------------------------------------
# CharacterizationService
# ---------------------------------------------------------------------------


class TestCharacterizationService:
    def test_create_and_get(self, service):
        body = {"summary": "A builder.", "current_focus": [{"item": "voice latency", "confidence": 0.8}]}
        row = service.upsert(1, "h1", body=body, rendered="Knows Alex.", confidence=0.7)
        assert row.id is not None
        assert row.version == 1
        got = service.get(1, "h1")
        assert got is not None
        assert json.loads(got.body)["summary"] == "A builder."
        assert got.rendered == "Knows Alex."
        assert got.confidence == 0.7

    def test_get_missing_returns_none(self, service):
        assert service.get(999, "h1") is None

    def test_upsert_bumps_version_single_row(self, service, db):
        service.upsert(1, "h1", body={"summary": "v1"})
        row2 = service.upsert(1, "h1", body={"summary": "v2"})
        assert row2.version == 2
        rows = db.query(PersonCharacterization).filter_by(user_id=1, household_id="h1").all()
        assert len(rows) == 1  # unique (user, household) — one evolving row
        assert json.loads(rows[0].body)["summary"] == "v2"

    def test_stale_focus_is_retired_not_accumulated(self, service):
        # The whole point: the view EVOLVES; it must not just grow.
        service.upsert(1, "h1", body={"current_focus": [{"item": "old thing"}]})
        service.upsert(1, "h1", body={"current_focus": [{"item": "new thing"}]})
        body = json.loads(service.get(1, "h1").body)
        assert [f["item"] for f in body["current_focus"]] == ["new thing"]

    def test_scoped_by_user_and_household(self, service):
        service.upsert(1, "h1", body={"summary": "alex"})
        service.upsert(2, "h1", body={"summary": "kaitlyn"})
        service.upsert(1, "h2", body={"summary": "alex-other-home"})
        assert json.loads(service.get(1, "h1").body)["summary"] == "alex"
        assert json.loads(service.get(2, "h1").body)["summary"] == "kaitlyn"
        assert json.loads(service.get(1, "h2").body)["summary"] == "alex-other-home"


# ---------------------------------------------------------------------------
# handle_synthesis_callback (glue: parse → field-map → upsert)
# ---------------------------------------------------------------------------


class TestHandleSynthesisCallback:
    def test_writes_characterization(self, engine, monkeypatch):
        doc = {
            "summary": "A builder mid-shipping-stretch.",
            "current_focus": [{"item": "characterization memory", "confidence": 0.9}],
            "confidence": 0.66,
            "rendered": "You're briefing Jarvis on Alex.",
        }
        payload = {
            "job_id": "job-1",
            "status": "succeeded",
            "metadata": {"user_id": 1, "household_id": "h1", "last_transcript_at": "2026-07-20T10:00:00"},
            "result": {"content": json.dumps(doc)},
        }
        Session = _run_callback(engine, payload, monkeypatch)
        row = Session().query(PersonCharacterization).filter_by(user_id=1, household_id="h1").first()
        assert row is not None
        assert row.version == 1
        assert row.confidence == 0.66
        assert row.rendered == "You're briefing Jarvis on Alex."
        assert row.last_transcript_at == datetime(2026, 7, 20, 10, 0, 0)
        assert row.model == "background"
        assert json.loads(row.body)["summary"].startswith("A builder")

    def test_failed_status_writes_nothing(self, engine, monkeypatch):
        payload = {
            "job_id": "j",
            "status": "failed",
            "metadata": {"user_id": 1, "household_id": "h1"},
            "error": {"message": "boom"},
        }
        Session = _run_callback(engine, payload, monkeypatch)
        assert Session().query(PersonCharacterization).count() == 0

    def test_unparseable_content_writes_nothing(self, engine, monkeypatch):
        payload = {
            "job_id": "j",
            "status": "succeeded",
            "metadata": {"user_id": 1, "household_id": "h1"},
            "result": {"content": "totally not json"},
        }
        Session = _run_callback(engine, payload, monkeypatch)
        assert Session().query(PersonCharacterization).count() == 0

    def test_non_numeric_confidence_coerced_to_none(self, engine, monkeypatch):
        doc = {"summary": "x", "confidence": "high", "rendered": "r"}
        payload = {
            "job_id": "j",
            "status": "succeeded",
            "metadata": {"user_id": 5, "household_id": "h1", "last_transcript_at": None},
            "result": {"content": json.dumps(doc)},
        }
        Session = _run_callback(engine, payload, monkeypatch)
        row = Session().query(PersonCharacterization).filter_by(user_id=5).first()
        assert row is not None
        assert row.confidence is None
        assert row.last_transcript_at is None

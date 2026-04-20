"""Tests for the /api/v0/transcripts endpoints (Phase 1)."""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base
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
def client(db):
    """TestClient with get_db + verify_user_jwt overridden."""
    from app.main import app
    from app.deps import AuthenticatedUser, get_db, verify_user_jwt

    def override_get_db():
        try:
            yield db
        finally:
            pass

    def override_user():
        return AuthenticatedUser(user_id=1, email="alex@test", is_superuser=False)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_user_jwt] = override_user

    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


@pytest.fixture()
def seed(db):
    def _seed(
        user_id: int = 1,
        message: str = "hi",
        tool_calls: list | None = None,
        household_id: str = "h1",
    ):
        return TranscriptService(db).log_transcript(
            user_id=user_id,
            household_id=household_id,
            conversation_id="c1",
            user_message=message,
            assistant_message="ok",
            tool_calls=tool_calls,
        )
    return _seed


class TestListRecent:
    def test_returns_own_rows(self, client, seed):
        seed(user_id=1, message="alice")
        seed(user_id=2, message="bob")
        res = client.get("/api/v0/transcripts/recent")
        assert res.status_code == 200
        messages = [r["user_message"] for r in res.json()]
        assert messages == ["alice"]

    def test_newest_first(self, client, db, seed):
        a = seed(message="first")
        a.created_at = datetime.utcnow() - timedelta(minutes=5)
        db.commit()
        seed(message="second")
        res = client.get("/api/v0/transcripts/recent")
        messages = [r["user_message"] for r in res.json()]
        assert messages == ["second", "first"]

    def test_limit_enforced(self, client, seed):
        for i in range(5):
            seed(message=f"m{i}")
        res = client.get("/api/v0/transcripts/recent?limit=2")
        assert len(res.json()) == 2

    def test_limit_max(self, client):
        # Project maps FastAPI 422 to 400 via a custom handler
        res = client.get("/api/v0/transcripts/recent?limit=9999")
        assert res.status_code in (400, 422)

    def test_tool_calls_deserialized(self, client, seed):
        seed(message="turn on lights", tool_calls=[{"name": "lights", "args": {"state": "on"}}])
        body = client.get("/api/v0/transcripts/recent").json()
        assert body[0]["tool_calls"] == [{"name": "lights", "args": {"state": "on"}}]


class TestRateTranscript:
    def test_sets_rating(self, client, seed):
        row = seed(message="parse me")
        res = client.post(
            f"/api/v0/transcripts/{row.id}/rate",
            json={"rating": 1, "notes": "good parse"},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["user_rating"] == 1
        assert body["rating_notes"] == "good parse"
        assert body["rated_at"] is not None

    def test_negative_rating(self, client, seed):
        row = seed(message="bad parse")
        res = client.post(f"/api/v0/transcripts/{row.id}/rate", json={"rating": -1})
        assert res.status_code == 200
        assert res.json()["user_rating"] == -1

    def test_other_users_row_returns_404(self, client, seed):
        row = seed(user_id=2, message="bob's row")
        res = client.post(f"/api/v0/transcripts/{row.id}/rate", json={"rating": 1})
        assert res.status_code == 404

    def test_missing_row_returns_404(self, client):
        res = client.post("/api/v0/transcripts/999999/rate", json={"rating": 1})
        assert res.status_code == 404

    def test_invalid_rating_value(self, client, seed):
        row = seed(message="x")
        res = client.post(f"/api/v0/transcripts/{row.id}/rate", json={"rating": 5})
        assert res.status_code == 400

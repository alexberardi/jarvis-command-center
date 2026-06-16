"""Tests for the self-scoped account-data purge (DELETE /api/v0/me/data).

Covers the account-deletion contract slice owned by command-center:

  - a user's memories, transcripts, user-scoped settings, and auth-sessions are
    all deleted
  - OTHER users' rows (and household-/node-scoped rows) are left untouched
  - the endpoint is idempotent (no rows → still 204)
  - a missing / invalid JWT is rejected with 401
"""

from contextlib import contextmanager
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import (
    AuthSession,
    Base,
    ConversationTranscript,
    Node,
    Setting,
    UserMemory,
)

HH = "hh-test"
ME = 1
OTHER = 2


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
def app_with_overrides(db):
    """FastAPI app with get_db overridden; auth overrides applied per-test."""
    from app.main import app
    from app.deps import get_db

    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    yield app
    app.dependency_overrides.clear()


@contextmanager
def _as_user(app, user_id: int):
    """Override the JWT dep to act as a specific user."""
    from app.deps import AuthenticatedUser, verify_user_jwt

    def fake_jwt():
        return AuthenticatedUser(user_id=user_id, email=f"u{user_id}@test")

    app.dependency_overrides[verify_user_jwt] = fake_jwt
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(verify_user_jwt, None)


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _seed_node(db, node_id: str = "node-1") -> Node:
    node = Node(node_id=node_id, api_key="k", room="office", household_id=HH)
    db.add(node)
    db.commit()
    return node


def _seed_for_user(db, user_id: int | None, node_id: str) -> None:
    """Create one row in each purgeable table for a given user_id."""
    db.add(
        UserMemory(
            user_id=user_id,
            household_id=HH,
            content=f"memory for {user_id}",
        )
    )
    if user_id is not None:
        # conversation_transcripts.user_id is NOT NULL
        db.add(
            ConversationTranscript(
                user_id=user_id,
                household_id=HH,
                conversation_id="conv-1",
                user_message="hello",
            )
        )
    db.add(
        Setting(
            key="some.pref",
            value="v",
            household_id=HH,
            user_id=user_id,
        )
    )
    db.add(
        AuthSession(
            provider="home_assistant",
            node_id=node_id,
            state=f"state-{user_id}",
            client_id="cid",
            user_id=user_id,
            expires_at=datetime.utcnow() + timedelta(minutes=10),
        )
    )
    db.commit()


def _counts_for(db, user_id: int) -> dict[str, int]:
    return {
        "memories": db.query(UserMemory).filter(UserMemory.user_id == user_id).count(),
        "transcripts": db.query(ConversationTranscript)
        .filter(ConversationTranscript.user_id == user_id)
        .count(),
        "settings": db.query(Setting).filter(Setting.user_id == user_id).count(),
        "auth_sessions": db.query(AuthSession)
        .filter(AuthSession.user_id == user_id)
        .count(),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPurge:
    def test_deletes_all_caller_rows(self, app_with_overrides, db):
        _seed_node(db)
        _seed_for_user(db, ME, "node-1")

        # Sanity: the caller has data before the purge.
        assert _counts_for(db, ME) == {
            "memories": 1,
            "transcripts": 1,
            "settings": 1,
            "auth_sessions": 1,
        }

        with _as_user(app_with_overrides, ME) as client:
            resp = client.delete("/api/v0/me/data")

        assert resp.status_code == 204
        assert resp.content == b""
        assert _counts_for(db, ME) == {
            "memories": 0,
            "transcripts": 0,
            "settings": 0,
            "auth_sessions": 0,
        }

    def test_leaves_other_users_rows_untouched(self, app_with_overrides, db):
        _seed_node(db)
        _seed_for_user(db, ME, "node-1")
        _seed_for_user(db, OTHER, "node-1")

        with _as_user(app_with_overrides, ME) as client:
            resp = client.delete("/api/v0/me/data")

        assert resp.status_code == 204
        # The other user keeps every row.
        assert _counts_for(db, OTHER) == {
            "memories": 1,
            "transcripts": 1,
            "settings": 1,
            "auth_sessions": 1,
        }

    def test_leaves_household_and_node_scoped_rows_untouched(
        self, app_with_overrides, db
    ):
        _seed_node(db)
        _seed_for_user(db, ME, "node-1")
        # household-/node-scoped rows have user_id IS NULL
        db.add(UserMemory(user_id=None, household_id=HH, content="household-wide"))
        db.add(Setting(key="hh.pref", value="v", household_id=HH, user_id=None))
        db.add(
            AuthSession(
                provider="home_assistant",
                node_id="node-1",
                state="state-shared",
                client_id="cid",
                user_id=None,
                expires_at=datetime.utcnow() + timedelta(minutes=10),
            )
        )
        db.commit()

        with _as_user(app_with_overrides, ME) as client:
            resp = client.delete("/api/v0/me/data")

        assert resp.status_code == 204
        assert (
            db.query(UserMemory).filter(UserMemory.user_id.is_(None)).count() == 1
        )
        assert db.query(Setting).filter(Setting.user_id.is_(None)).count() == 1
        assert (
            db.query(AuthSession).filter(AuthSession.user_id.is_(None)).count() == 1
        )

    def test_idempotent_with_no_rows(self, app_with_overrides, db):
        # No seeding for ME at all.
        with _as_user(app_with_overrides, ME) as client:
            resp = client.delete("/api/v0/me/data")
        assert resp.status_code == 204

        # Calling it again is still a clean 204.
        with _as_user(app_with_overrides, ME) as client:
            resp = client.delete("/api/v0/me/data")
        assert resp.status_code == 204

    def test_missing_jwt_rejected(self, app_with_overrides):
        # No _as_user override → real verify_user_jwt runs and sees no header.
        with TestClient(app_with_overrides) as client:
            resp = client.delete("/api/v0/me/data")
        assert resp.status_code == 401

    def test_invalid_jwt_rejected(self, app_with_overrides):
        with TestClient(app_with_overrides) as client:
            resp = client.delete(
                "/api/v0/me/data",
                headers={"Authorization": "Bearer not-a-real-token"},
            )
        assert resp.status_code == 401

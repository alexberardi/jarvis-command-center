"""Tests for the mobile memory CRUD API (/api/v0/mobile/memories).

Covers the role-scoped permission matrix:

  MEMBER         sees / writes own only
  POWER_USER     sees own + household-wide; non-agent household-wide writable
  ADMIN          full read+write including agent_context
  is_superuser   same as ADMIN globally
"""

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base, UserMemory
from app.services.memory_service import MemoryService

HH = "hh-test"
OTHER_HH = "hh-other"


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
def _as_user(app, user_id: int, role: str, is_superuser: bool = False):
    """Override auth deps to act as a specific (user, role)."""
    from app.deps import AuthenticatedUser, verify_user_jwt

    def fake_jwt():
        return AuthenticatedUser(
            user_id=user_id, email=f"u{user_id}@test", is_superuser=is_superuser,
        )

    def fake_role(_uid: int, _hid: str):
        return role

    app.dependency_overrides[verify_user_jwt] = fake_jwt
    # resolve_household_role is called directly (not as a dep) — patch via monkeypatch
    import app.api.mobile_memories as mm
    original = mm.resolve_household_role
    mm.resolve_household_role = fake_role
    try:
        with TestClient(app) as client:
            yield client
    finally:
        mm.resolve_household_role = original
        app.dependency_overrides.pop(verify_user_jwt, None)


@pytest.fixture()
def seed(db):
    """Create a memory directly with sensible defaults."""
    def _seed(
        user_id: int | None = 1,
        household_id: str = HH,
        content: str = "fact",
        category: str = "general",
        source: str = "ui",
        key: str | None = None,
        is_pinned: bool = False,
    ) -> UserMemory:
        return MemoryService(db).save_memory(
            user_id=user_id,
            household_id=household_id,
            content=content,
            category=category,
            key=key,
            source=source,
            is_pinned=is_pinned,
        )
    return _seed


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


class TestListScope:
    def test_member_sees_only_own(self, app_with_overrides, seed):
        seed(user_id=1, content="mine")
        seed(user_id=2, content="not mine")
        seed(user_id=None, content="household-wide", source="agent", category="agent_context")

        with _as_user(app_with_overrides, user_id=1, role="member") as client:
            resp = client.get("/api/v0/mobile/memories", params={"household_id": HH})
            assert resp.status_code == 200
            contents = {m["content"] for m in resp.json()}
            assert contents == {"mine"}

    def test_power_user_sees_own_and_household(self, app_with_overrides, seed):
        seed(user_id=1, content="mine")
        seed(user_id=2, content="not mine")
        seed(user_id=None, content="household-wide")

        with _as_user(app_with_overrides, user_id=1, role="power_user") as client:
            resp = client.get("/api/v0/mobile/memories", params={"household_id": HH})
            assert resp.status_code == 200
            contents = {m["content"] for m in resp.json()}
            assert contents == {"mine", "household-wide"}

    def test_admin_same_as_power_user_for_list(self, app_with_overrides, seed):
        seed(user_id=1, content="mine")
        seed(user_id=None, content="household-wide")

        with _as_user(app_with_overrides, user_id=1, role="admin") as client:
            resp = client.get("/api/v0/mobile/memories", params={"household_id": HH})
            assert resp.status_code == 200
            contents = {m["content"] for m in resp.json()}
            assert contents == {"mine", "household-wide"}

    def test_superuser_with_member_role_sees_household(self, app_with_overrides, seed):
        seed(user_id=1, content="mine")
        seed(user_id=None, content="household-wide")

        with _as_user(
            app_with_overrides, user_id=1, role="member", is_superuser=True,
        ) as client:
            resp = client.get("/api/v0/mobile/memories", params={"household_id": HH})
            contents = {m["content"] for m in resp.json()}
            assert contents == {"mine", "household-wide"}

    def test_excludes_other_household(self, app_with_overrides, seed):
        seed(user_id=1, household_id=HH, content="here")
        seed(user_id=1, household_id=OTHER_HH, content="there")

        with _as_user(app_with_overrides, user_id=1, role="admin") as client:
            resp = client.get("/api/v0/mobile/memories", params={"household_id": HH})
            contents = {m["content"] for m in resp.json()}
            assert contents == {"here"}

    def test_include_household_false_hides_household_wide(self, app_with_overrides, seed):
        seed(user_id=1, content="mine")
        seed(user_id=None, content="household-wide")

        with _as_user(app_with_overrides, user_id=1, role="power_user") as client:
            resp = client.get(
                "/api/v0/mobile/memories",
                params={"household_id": HH, "include_household": False},
            )
            contents = {m["content"] for m in resp.json()}
            assert contents == {"mine"}

    def test_excludes_inactive(self, app_with_overrides, seed, db):
        m = seed(user_id=1, content="gone")
        m.is_active = False
        db.commit()

        with _as_user(app_with_overrides, user_id=1, role="member") as client:
            resp = client.get("/api/v0/mobile/memories", params={"household_id": HH})
            assert resp.json() == []

    def test_editable_flag_reflects_permission(self, app_with_overrides, seed):
        seed(user_id=1, content="mine")
        seed(user_id=None, content="ai context", source="agent", category="agent_context")

        with _as_user(app_with_overrides, user_id=1, role="power_user") as client:
            resp = client.get("/api/v0/mobile/memories", params={"household_id": HH})
            by_content = {m["content"]: m for m in resp.json()}
            assert by_content["mine"]["editable"] is True
            assert by_content["ai context"]["editable"] is False


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


class TestCreate:
    def test_member_can_create_own(self, app_with_overrides):
        with _as_user(app_with_overrides, user_id=1, role="member") as client:
            resp = client.post(
                "/api/v0/mobile/memories",
                params={"household_id": HH},
                json={"content": "likes tea", "category": "preference"},
            )
            assert resp.status_code == 201
            body = resp.json()
            assert body["user_id"] == 1
            assert body["source"] == "ui"
            assert body["editable"] is True

    def test_member_blocked_from_household_scope(self, app_with_overrides):
        with _as_user(app_with_overrides, user_id=1, role="member") as client:
            resp = client.post(
                "/api/v0/mobile/memories",
                params={"household_id": HH},
                json={"content": "x", "scope": "household"},
            )
            assert resp.status_code == 403

    def test_power_user_can_create_household_scope(self, app_with_overrides):
        with _as_user(app_with_overrides, user_id=1, role="power_user") as client:
            resp = client.post(
                "/api/v0/mobile/memories",
                params={"household_id": HH},
                json={"content": "kitchen lights bright", "scope": "household"},
            )
            assert resp.status_code == 201
            assert resp.json()["user_id"] is None

    def test_non_admin_blocked_from_agent_context_category(self, app_with_overrides):
        with _as_user(app_with_overrides, user_id=1, role="power_user") as client:
            resp = client.post(
                "/api/v0/mobile/memories",
                params={"household_id": HH},
                json={"content": "x", "category": "agent_context", "scope": "household"},
            )
            assert resp.status_code == 403

    def test_admin_can_create_agent_context(self, app_with_overrides):
        with _as_user(app_with_overrides, user_id=1, role="admin") as client:
            resp = client.post(
                "/api/v0/mobile/memories",
                params={"household_id": HH},
                json={"content": "x", "category": "agent_context", "scope": "household"},
            )
            assert resp.status_code == 201

    def test_source_always_forced_to_ui(self, app_with_overrides):
        with _as_user(app_with_overrides, user_id=1, role="admin") as client:
            resp = client.post(
                "/api/v0/mobile/memories",
                params={"household_id": HH},
                json={"content": "x"},
            )
            assert resp.json()["source"] == "ui"


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_own(self, app_with_overrides, seed):
        m = seed(user_id=1, content="mine")
        with _as_user(app_with_overrides, user_id=1, role="member") as client:
            resp = client.get(
                f"/api/v0/mobile/memories/{m.id}", params={"household_id": HH},
            )
            assert resp.status_code == 200
            assert resp.json()["content"] == "mine"

    def test_member_cannot_get_other_user(self, app_with_overrides, seed):
        m = seed(user_id=2, content="not mine")
        with _as_user(app_with_overrides, user_id=1, role="member") as client:
            resp = client.get(
                f"/api/v0/mobile/memories/{m.id}", params={"household_id": HH},
            )
            assert resp.status_code == 404

    def test_member_cannot_get_household_wide(self, app_with_overrides, seed):
        m = seed(user_id=None, content="household-wide")
        with _as_user(app_with_overrides, user_id=1, role="member") as client:
            resp = client.get(
                f"/api/v0/mobile/memories/{m.id}", params={"household_id": HH},
            )
            assert resp.status_code == 404

    def test_power_user_can_get_household_wide(self, app_with_overrides, seed):
        m = seed(user_id=None, content="household-wide")
        with _as_user(app_with_overrides, user_id=1, role="power_user") as client:
            resp = client.get(
                f"/api/v0/mobile/memories/{m.id}", params={"household_id": HH},
            )
            assert resp.status_code == 200

    def test_cross_household_returns_404(self, app_with_overrides, seed):
        m = seed(user_id=1, household_id=OTHER_HH, content="elsewhere")
        with _as_user(app_with_overrides, user_id=1, role="admin") as client:
            resp = client.get(
                f"/api/v0/mobile/memories/{m.id}", params={"household_id": HH},
            )
            assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_own(self, app_with_overrides, seed):
        m = seed(user_id=1, content="old")
        with _as_user(app_with_overrides, user_id=1, role="member") as client:
            resp = client.put(
                f"/api/v0/mobile/memories/{m.id}",
                params={"household_id": HH},
                json={"content": "new"},
            )
            assert resp.status_code == 200
            assert resp.json()["content"] == "new"

    def test_member_cannot_update_household_wide(self, app_with_overrides, seed):
        m = seed(user_id=None, content="x")
        with _as_user(app_with_overrides, user_id=1, role="member") as client:
            resp = client.put(
                f"/api/v0/mobile/memories/{m.id}",
                params={"household_id": HH},
                json={"content": "y"},
            )
            assert resp.status_code == 404  # hidden, not 403

    def test_power_user_can_update_household_non_agent(self, app_with_overrides, seed):
        m = seed(user_id=None, content="lights bright", source="ui")
        with _as_user(app_with_overrides, user_id=1, role="power_user") as client:
            resp = client.put(
                f"/api/v0/mobile/memories/{m.id}",
                params={"household_id": HH},
                json={"content": "lights dim"},
            )
            assert resp.status_code == 200

    def test_power_user_blocked_on_agent_source(self, app_with_overrides, seed):
        m = seed(
            user_id=None, content="weather: sunny", source="agent", category="general",
        )
        with _as_user(app_with_overrides, user_id=1, role="power_user") as client:
            resp = client.put(
                f"/api/v0/mobile/memories/{m.id}",
                params={"household_id": HH},
                json={"content": "x"},
            )
            assert resp.status_code == 403
            assert "read-only" in resp.json()["detail"].lower()

    def test_power_user_blocked_on_agent_context_category(self, app_with_overrides, seed):
        m = seed(user_id=None, content="x", source="ui", category="agent_context")
        with _as_user(app_with_overrides, user_id=1, role="power_user") as client:
            resp = client.put(
                f"/api/v0/mobile/memories/{m.id}",
                params={"household_id": HH},
                json={"content": "y"},
            )
            assert resp.status_code == 403

    def test_admin_can_update_agent_context(self, app_with_overrides, seed):
        m = seed(user_id=None, content="x", source="agent", category="agent_context")
        with _as_user(app_with_overrides, user_id=1, role="admin") as client:
            resp = client.put(
                f"/api/v0/mobile/memories/{m.id}",
                params={"household_id": HH},
                json={"content": "y"},
            )
            assert resp.status_code == 200

    def test_power_user_cannot_set_category_to_agent_context(self, app_with_overrides, seed):
        m = seed(user_id=None, content="x", source="ui", category="general")
        with _as_user(app_with_overrides, user_id=1, role="power_user") as client:
            resp = client.put(
                f"/api/v0/mobile/memories/{m.id}",
                params={"household_id": HH},
                json={"category": "agent_context"},
            )
            assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_own(self, app_with_overrides, seed):
        m = seed(user_id=1, content="bye")
        with _as_user(app_with_overrides, user_id=1, role="member") as client:
            resp = client.delete(
                f"/api/v0/mobile/memories/{m.id}", params={"household_id": HH},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "deleted"

    def test_member_cannot_delete_other_user(self, app_with_overrides, seed):
        m = seed(user_id=2, content="theirs")
        with _as_user(app_with_overrides, user_id=1, role="member") as client:
            resp = client.delete(
                f"/api/v0/mobile/memories/{m.id}", params={"household_id": HH},
            )
            assert resp.status_code == 404

    def test_power_user_blocked_from_deleting_agent_injected(self, app_with_overrides, seed):
        m = seed(
            user_id=None, content="x", source="agent", category="agent_context",
        )
        with _as_user(app_with_overrides, user_id=1, role="power_user") as client:
            resp = client.delete(
                f"/api/v0/mobile/memories/{m.id}", params={"household_id": HH},
            )
            assert resp.status_code == 403

    def test_admin_can_delete_agent_injected(self, app_with_overrides, seed):
        m = seed(
            user_id=None, content="x", source="agent", category="agent_context",
        )
        with _as_user(app_with_overrides, user_id=1, role="admin") as client:
            resp = client.delete(
                f"/api/v0/mobile/memories/{m.id}", params={"household_id": HH},
            )
            assert resp.status_code == 200

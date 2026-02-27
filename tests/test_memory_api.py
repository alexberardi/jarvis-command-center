"""Tests for memory CRUD API endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base, UserMemory
from app.services.memory_service import MemoryService


@pytest.fixture()
def db():
    """Create an in-memory SQLite database for testing."""
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
def memory_client(db):
    """Test client with auth + DB overridden for memory endpoints."""
    from app.main import app
    from app.deps import get_db
    from app.provisioning import verify_provisioning_auth, ProvisioningAuthContext

    def override_get_db():
        try:
            yield db
        finally:
            pass

    def override_auth():
        return ProvisioningAuthContext(auth_type="admin_key")

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_provisioning_auth] = override_auth

    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture()
def seed_memory(db):
    """Helper to create a memory directly in the database."""
    def _create(
        user_id: int = 1,
        household_id: str = "hh-test",
        content: str = "likes black coffee",
        category: str = "preference",
        key: str | None = None,
        source: str = "voice",
    ) -> UserMemory:
        service = MemoryService(db)
        return service.save_memory(
            user_id=user_id,
            household_id=household_id,
            content=content,
            category=category,
            key=key,
            source=source,
        )
    return _create


class TestListMemories:
    def test_list_empty(self, memory_client):
        resp = memory_client.get("/api/v0/memories", params={"user_id": 1, "household_id": "hh-test"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_returns_memories(self, memory_client, seed_memory):
        seed_memory(content="likes black coffee")
        seed_memory(content="morning person", category="fact")

        resp = memory_client.get("/api/v0/memories", params={"user_id": 1, "household_id": "hh-test"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        contents = {m["content"] for m in data}
        assert "likes black coffee" in contents
        assert "morning person" in contents

    def test_list_filters_by_category(self, memory_client, seed_memory):
        seed_memory(content="likes black coffee", category="preference")
        seed_memory(content="morning person", category="fact")

        resp = memory_client.get(
            "/api/v0/memories",
            params={"user_id": 1, "household_id": "hh-test", "category": "preference"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["content"] == "likes black coffee"

    def test_list_excludes_inactive(self, memory_client, seed_memory, db):
        m = seed_memory(content="old memory")
        m.is_active = False
        db.commit()

        resp = memory_client.get("/api/v0/memories", params={"user_id": 1, "household_id": "hh-test"})
        assert resp.status_code == 200
        assert resp.json() == []


class TestCreateMemory:
    def test_create_memory(self, memory_client):
        resp = memory_client.post(
            "/api/v0/memories",
            params={"user_id": 1, "household_id": "hh-test"},
            json={"content": "allergic to peanuts", "category": "fact"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["content"] == "allergic to peanuts"
        assert body["category"] == "fact"
        assert body["source"] == "ui"
        assert body["is_active"] is True
        assert "id" in body

    def test_create_with_key_upserts(self, memory_client):
        params = {"user_id": 1, "household_id": "hh-test"}

        resp1 = memory_client.post(
            "/api/v0/memories",
            params=params,
            json={"content": "prefers dark mode", "key": "theme"},
        )
        assert resp1.status_code == 201
        id1 = resp1.json()["id"]

        resp2 = memory_client.post(
            "/api/v0/memories",
            params=params,
            json={"content": "prefers light mode", "key": "theme"},
        )
        assert resp2.status_code == 201
        id2 = resp2.json()["id"]
        assert id1 == id2  # same record updated
        assert resp2.json()["content"] == "prefers light mode"


class TestGetMemory:
    def test_get_memory(self, memory_client, seed_memory):
        m = seed_memory(content="test memory")

        resp = memory_client.get(f"/api/v0/memories/{m.id}")
        assert resp.status_code == 200
        assert resp.json()["content"] == "test memory"

    def test_get_not_found(self, memory_client):
        resp = memory_client.get("/api/v0/memories/99999")
        assert resp.status_code == 404


class TestUpdateMemory:
    def test_update_content(self, memory_client, seed_memory):
        m = seed_memory(content="old content")

        resp = memory_client.put(
            f"/api/v0/memories/{m.id}",
            json={"content": "new content"},
        )
        assert resp.status_code == 200
        assert resp.json()["content"] == "new content"

    def test_update_category(self, memory_client, seed_memory):
        m = seed_memory(content="some note", category="note")

        resp = memory_client.put(
            f"/api/v0/memories/{m.id}",
            json={"category": "preference"},
        )
        assert resp.status_code == 200
        assert resp.json()["category"] == "preference"
        assert resp.json()["content"] == "some note"  # unchanged

    def test_update_not_found(self, memory_client):
        resp = memory_client.put("/api/v0/memories/99999", json={"content": "x"})
        assert resp.status_code == 404


class TestDeleteMemory:
    def test_soft_delete(self, memory_client, seed_memory):
        m = seed_memory(content="to be deleted")

        resp = memory_client.delete(f"/api/v0/memories/{m.id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify it's soft-deleted (no longer in active list)
        list_resp = memory_client.get(
            "/api/v0/memories",
            params={"user_id": 1, "household_id": "hh-test"},
        )
        assert list_resp.status_code == 200
        assert list_resp.json() == []

    def test_delete_not_found(self, memory_client):
        resp = memory_client.delete("/api/v0/memories/99999")
        assert resp.status_code == 404

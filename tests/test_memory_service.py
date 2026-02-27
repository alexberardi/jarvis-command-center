"""Tests for MemoryService."""
import pytest
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, UserMemory
from app.services.memory_service import MemoryService


@pytest.fixture()
def db():
    """Create an in-memory SQLite database for testing."""
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
    return MemoryService(db)


class TestSaveMemory:
    def test_save_basic(self, service, db):
        mem = service.save_memory(1, "h1", "likes black coffee", category="preference")
        assert mem.id is not None
        assert mem.content == "likes black coffee"
        assert mem.category == "preference"
        assert mem.user_id == 1
        assert mem.household_id == "h1"
        assert mem.is_active is True
        assert mem.source == "voice"

    def test_save_with_key(self, service, db):
        mem = service.save_memory(1, "h1", "black", category="preference", key="coffee_preference")
        assert mem.key == "coffee_preference"

    def test_upsert_on_same_key(self, service, db):
        m1 = service.save_memory(1, "h1", "black", key="coffee")
        m2 = service.save_memory(1, "h1", "with cream", key="coffee")
        # Same row updated
        assert m1.id == m2.id
        assert m2.content == "with cream"

    def test_no_upsert_without_key(self, service, db):
        m1 = service.save_memory(1, "h1", "fact one")
        m2 = service.save_memory(1, "h1", "fact two")
        assert m1.id != m2.id

    def test_different_users_same_key(self, service, db):
        m1 = service.save_memory(1, "h1", "black", key="coffee")
        m2 = service.save_memory(2, "h1", "latte", key="coffee")
        assert m1.id != m2.id


class TestGetActiveMemories:
    def test_returns_active_only(self, service, db):
        service.save_memory(1, "h1", "active memory")
        inactive = service.save_memory(1, "h1", "inactive memory")
        inactive.is_active = False
        db.commit()

        result = service.get_active_memories(1, "h1")
        assert len(result) == 1
        assert result[0].content == "active memory"

    def test_filters_by_category(self, service, db):
        service.save_memory(1, "h1", "pref", category="preference")
        service.save_memory(1, "h1", "note", category="note")

        prefs = service.get_active_memories(1, "h1", categories=["preference"])
        assert len(prefs) == 1
        assert prefs[0].category == "preference"

    def test_excludes_expired(self, service, db):
        mem = service.save_memory(1, "h1", "temporary fact")
        mem.expires_at = datetime.utcnow() - timedelta(hours=1)
        db.commit()

        result = service.get_active_memories(1, "h1")
        assert len(result) == 0

    def test_household_scoped(self, service, db):
        service.save_memory(1, "h1", "household 1 memory")
        service.save_memory(1, "h2", "household 2 memory")

        result = service.get_active_memories(1, "h1")
        assert len(result) == 1
        assert result[0].household_id == "h1"


class TestForgetMemory:
    def test_forget_by_key(self, service, db):
        service.save_memory(1, "h1", "black coffee", key="coffee")
        count = service.forget_memory(1, "h1", key="coffee")
        assert count == 1

        active = service.get_active_memories(1, "h1")
        assert len(active) == 0

    def test_forget_by_content_match(self, service, db):
        service.save_memory(1, "h1", "I like black coffee")
        service.save_memory(1, "h1", "I prefer tea")

        count = service.forget_memory(1, "h1", content_match="coffee")
        assert count == 1

        active = service.get_active_memories(1, "h1")
        assert len(active) == 1
        assert "tea" in active[0].content

    def test_forget_case_insensitive(self, service, db):
        service.save_memory(1, "h1", "I like BLACK coffee")
        count = service.forget_memory(1, "h1", content_match="black")
        assert count == 1

    def test_forget_no_filter_returns_zero(self, service, db):
        service.save_memory(1, "h1", "something")
        count = service.forget_memory(1, "h1")
        assert count == 0

    def test_forget_nonexistent(self, service, db):
        count = service.forget_memory(1, "h1", key="nonexistent")
        assert count == 0


class TestGetMemoriesForPrompt:
    def test_formats_as_bullet_list(self, service, db):
        service.save_memory(1, "h1", "likes coffee black", category="preference")
        service.save_memory(1, "h1", "works from home", category="fact")

        result = service.get_memories_for_prompt(1, "h1")
        assert "- likes coffee black" in result
        assert "- works from home" in result

    def test_respects_char_budget(self, service, db):
        for i in range(20):
            service.save_memory(1, "h1", f"memory number {i} with some content padding here")

        result = service.get_memories_for_prompt(1, "h1", max_chars=100)
        assert len(result) <= 100

    def test_prioritizes_preferences(self, service, db):
        service.save_memory(1, "h1", "a general note", category="note")
        service.save_memory(1, "h1", "prefers tea", category="preference")

        result = service.get_memories_for_prompt(1, "h1")
        lines = result.strip().split("\n")
        # Preference should be first
        assert "prefers tea" in lines[0]

    def test_empty_when_no_memories(self, service, db):
        result = service.get_memories_for_prompt(1, "h1")
        assert result == ""


class TestCleanupExpired:
    def test_deactivates_expired(self, service, db):
        mem = service.save_memory(1, "h1", "temp fact")
        mem.expires_at = datetime.utcnow() - timedelta(hours=1)
        db.commit()

        count = service.cleanup_expired()
        assert count == 1

        db.refresh(mem)
        assert mem.is_active is False

    def test_ignores_non_expired(self, service, db):
        mem = service.save_memory(1, "h1", "future fact")
        mem.expires_at = datetime.utcnow() + timedelta(hours=1)
        db.commit()

        count = service.cleanup_expired()
        assert count == 0

        db.refresh(mem)
        assert mem.is_active is True

    def test_ignores_already_inactive(self, service, db):
        mem = service.save_memory(1, "h1", "old fact")
        mem.expires_at = datetime.utcnow() - timedelta(hours=1)
        mem.is_active = False
        db.commit()

        count = service.cleanup_expired()
        assert count == 0

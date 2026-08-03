"""Tests for MemoryService."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

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


class TestEmbedMissing:
    """The sweep that makes any unembedded memory (passive extraction, agent
    contributions, a failed remember-embed) recallable — semantic recall filters
    embedding IS NOT NULL, so an unembedded memory is invisible to it."""

    def test_embeds_pending_and_returns_count(self, service, db):
        service.save_memory(1, "h1", "likes tea", key="k1")
        service.save_memory(1, "h1", "lives in Denver", key="k2")
        assert len(service.get_memories_without_embeddings()) == 2

        fake = MagicMock()
        fake.create_embeddings_sync.return_value = [[0.1] * 384, [0.2] * 384]
        with patch("app.core.llm_proxy_client.LLMProxyClient", return_value=fake):
            n = service.embed_missing()

        assert n == 2
        fake.create_embeddings_sync.assert_called_once()  # one batched call, not N
        assert service.get_memories_without_embeddings() == []

    def test_noop_when_nothing_pending(self, service, db):
        with patch("app.core.llm_proxy_client.LLMProxyClient") as cls:
            assert service.embed_missing() == 0
            cls.assert_not_called()  # don't even construct the client with no work

    def test_skips_rows_whose_embedding_failed(self, service, db):
        service.save_memory(1, "h1", "keep", key="k1")
        service.save_memory(1, "h1", "retry-me", key="k2")
        fake = MagicMock()
        fake.create_embeddings_sync.return_value = [[0.1] * 384, []]  # 2nd came back empty
        with patch("app.core.llm_proxy_client.LLMProxyClient", return_value=fake):
            n = service.embed_missing()

        assert n == 1
        remaining = service.get_memories_without_embeddings()
        assert [m.content for m in remaining] == ["retry-me"]  # still pending for next sweep


class TestEmbeddingInvalidationOnUpdate:
    """A3: when an upsert changes the text, the stale vector must be dropped so
    recall can't keep matching the OLD content; the sweep re-embeds it."""

    def _row(self, db, mem_id):
        return db.query(UserMemory).filter(UserMemory.id == mem_id).first()

    def test_content_change_nulls_the_embedding(self, service, db):
        m = service.save_memory(1, "h1", "likes tea", key="k1")
        service.update_embedding(m.id, [0.1] * 384)
        db.expire_all()
        assert self._row(db, m.id).embedding is not None

        service.save_memory(1, "h1", "likes green tea", key="k1")  # same key, new text
        db.expire_all()
        assert self._row(db, m.id).embedding is None

    def test_same_content_keeps_the_embedding(self, service, db):
        m = service.save_memory(1, "h1", "likes tea", key="k1")
        service.update_embedding(m.id, [0.1] * 384)
        service.save_memory(1, "h1", "likes tea", key="k1")  # identical upsert
        db.expire_all()
        assert self._row(db, m.id).embedding is not None

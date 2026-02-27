"""Tests for Remember and Forget server tools."""
import pytest
from unittest.mock import patch, MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, UserMemory
from app.core.tools.remember_tool import RememberTool
from app.core.tools.forget_tool import ForgetTool


@pytest.fixture()
def db_engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def mock_session_local(db_engine):
    """Patch get_session_local to return our test session factory."""
    Session = sessionmaker(bind=db_engine)
    with patch("app.db.get_session_local", return_value=Session):
        yield Session


def _mock_node_context(speaker_user_id=42, household_id="h1", speaker_name="Alice"):
    return {
        "speaker_user_id": speaker_user_id,
        "household_id": household_id,
        "speaker_name": speaker_name,
        "room": "kitchen",
    }


class TestRememberTool:
    def test_properties(self):
        tool = RememberTool()
        assert tool.name == "remember"
        assert "content" in tool.parameters["properties"]
        assert tool.enabled is True

    @patch("app.core.tools.remember_tool.conversation_cache")
    def test_saves_memory(self, mock_cache, mock_session_local):
        mock_cache.get_node_context.return_value = _mock_node_context()

        tool = RememberTool()
        result = tool.execute(
            content="likes coffee black",
            category="preference",
            conversation_id="conv-123",
        )

        assert result["status"] == "remembered"
        assert result["content"] == "likes coffee black"

        # Verify it's in the DB
        db = mock_session_local()
        memories = db.query(UserMemory).all()
        assert len(memories) == 1
        assert memories[0].content == "likes coffee black"
        assert memories[0].user_id == 42
        db.close()

    @patch("app.core.tools.remember_tool.conversation_cache")
    def test_no_speaker_error(self, mock_cache):
        mock_cache.get_node_context.return_value = {"household_id": "h1"}

        tool = RememberTool()
        result = tool.execute(content="test", conversation_id="conv-123")

        assert result["error"] == "no_speaker"

    @patch("app.core.tools.remember_tool.conversation_cache")
    def test_no_conversation_error(self, mock_cache):
        tool = RememberTool()
        result = tool.execute(content="test")

        assert result["error"] == "no_conversation"

    @patch("app.core.tools.remember_tool.conversation_cache")
    def test_with_key_upserts(self, mock_cache, mock_session_local):
        mock_cache.get_node_context.return_value = _mock_node_context()

        tool = RememberTool()
        tool.execute(content="black", key="coffee", conversation_id="conv-1")
        tool.execute(content="with cream", key="coffee", conversation_id="conv-1")

        db = mock_session_local()
        memories = db.query(UserMemory).filter(UserMemory.is_active == True).all()  # noqa: E712
        assert len(memories) == 1
        assert memories[0].content == "with cream"
        db.close()


class TestForgetTool:
    def test_properties(self):
        tool = ForgetTool()
        assert tool.name == "forget"
        assert "content_match" in tool.parameters["properties"]

    @patch("app.core.tools.forget_tool.conversation_cache")
    def test_forgets_matching_memory(self, mock_cache, mock_session_local):
        mock_cache.get_node_context.return_value = _mock_node_context()

        # First save a memory
        remember = RememberTool()
        with patch("app.core.tools.remember_tool.conversation_cache", mock_cache):
            remember.execute(content="likes coffee black", conversation_id="conv-1")

        # Now forget it
        forget = ForgetTool()
        result = forget.execute(content_match="coffee", conversation_id="conv-1")

        assert result["status"] == "forgotten"
        assert result["count"] == 1

        # Verify it's deactivated
        db = mock_session_local()
        active = db.query(UserMemory).filter(UserMemory.is_active == True).all()  # noqa: E712
        assert len(active) == 0
        db.close()

    @patch("app.core.tools.forget_tool.conversation_cache")
    def test_not_found(self, mock_cache, mock_session_local):
        mock_cache.get_node_context.return_value = _mock_node_context()

        tool = ForgetTool()
        result = tool.execute(content_match="nonexistent", conversation_id="conv-1")

        assert result["status"] == "not_found"
        assert result["count"] == 0

    @patch("app.core.tools.forget_tool.conversation_cache")
    def test_no_speaker_error(self, mock_cache):
        mock_cache.get_node_context.return_value = {"household_id": "h1"}

        tool = ForgetTool()
        result = tool.execute(content_match="test", conversation_id="conv-1")

        assert result["error"] == "no_speaker"

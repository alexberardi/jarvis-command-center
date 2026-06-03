"""Tests for the IdentifySpeakerTool — the "who am I?" server tool."""

from unittest.mock import patch

import pytest

from app.core.tools.identify_speaker_tool import IdentifySpeakerTool


def _node_context(speaker_user_id=42, speaker_name="Alice", household_id="h1"):
    return {
        "speaker_user_id": speaker_user_id,
        "speaker_name": speaker_name,
        "household_id": household_id,
        "room": "kitchen",
    }


class TestIdentifySpeakerProperties:
    def test_name(self):
        assert IdentifySpeakerTool().name == "identify_speaker"

    def test_no_parameters(self):
        params = IdentifySpeakerTool().parameters
        assert params["properties"] == {}
        assert params["required"] == []

    def test_default_enabled(self):
        assert IdentifySpeakerTool().enabled is True

    def test_included_prompt_text_mentions_positive_evidence(self):
        # The whole point of the tool is letting the strict not_for_me
        # prompt stay strict — the system-prompt text must explicitly
        # bind identity questions to this tool so the LLM stops hedging.
        text = IdentifySpeakerTool().included_system_prompt_text
        assert text is not None
        assert "identify_speaker" in text
        assert "not_for_me" in text


class TestIdentifySpeakerExecute:
    @patch("app.core.tools.identify_speaker_tool.conversation_cache")
    def test_returns_speaker_name_when_recognized(self, mock_cache):
        mock_cache.get_node_context.return_value = _node_context(
            speaker_user_id=42, speaker_name="Alice"
        )

        result = IdentifySpeakerTool().execute(conversation_id="conv-1")

        assert result == {"speaker_name": "Alice"}

    @patch("app.core.tools.identify_speaker_tool.conversation_cache")
    def test_trims_whitespace_around_name(self, mock_cache):
        mock_cache.get_node_context.return_value = _node_context(
            speaker_user_id=42, speaker_name="  Alice  "
        )

        result = IdentifySpeakerTool().execute(conversation_id="conv-1")

        assert result == {"speaker_name": "Alice"}

    @patch("app.core.tools.identify_speaker_tool.conversation_cache")
    def test_unknown_when_no_speaker_user_id(self, mock_cache):
        mock_cache.get_node_context.return_value = _node_context(
            speaker_user_id=None, speaker_name="Alice"
        )

        result = IdentifySpeakerTool().execute(conversation_id="conv-1")

        assert result["speaker_name"] is None
        assert "message" in result

    @patch("app.core.tools.identify_speaker_tool.conversation_cache")
    @pytest.mark.parametrize("placeholder", ["default", "user", "", "  "])
    def test_unknown_when_speaker_name_is_placeholder(self, mock_cache, placeholder):
        # The system-prompt builder falls back to "default" / "user" when
        # nothing is set. Treat those as unrecognized — speaking "you are
        # default" would be worse than the recovery message.
        mock_cache.get_node_context.return_value = _node_context(
            speaker_user_id=42, speaker_name=placeholder
        )

        result = IdentifySpeakerTool().execute(conversation_id="conv-1")

        assert result["speaker_name"] is None
        assert "message" in result

    @patch("app.core.tools.identify_speaker_tool.conversation_cache")
    def test_error_when_no_conversation_id(self, mock_cache):
        result = IdentifySpeakerTool().execute()

        assert result["speaker_name"] is None
        assert result["error"] == "no_conversation"
        mock_cache.get_node_context.assert_not_called()

    @patch("app.core.tools.identify_speaker_tool.conversation_cache")
    def test_error_when_no_node_context(self, mock_cache):
        mock_cache.get_node_context.return_value = None

        result = IdentifySpeakerTool().execute(conversation_id="conv-1")

        assert result["speaker_name"] is None
        assert result["error"] == "no_context"

    @patch("app.core.tools.identify_speaker_tool.conversation_cache")
    def test_ignores_extra_kwargs(self, mock_cache):
        # Tool executor may pass user_utterance, conversation_id, etc.;
        # the tool must not blow up on unknown keys.
        mock_cache.get_node_context.return_value = _node_context()

        result = IdentifySpeakerTool().execute(
            conversation_id="conv-1",
            user_utterance="who am i",
            unexpected_kwarg="ignored",
        )

        assert result == {"speaker_name": "Alice"}


class TestAutoRegistration:
    """Verify the registry picks it up via auto-discovery."""

    def test_registry_finds_tool(self):
        from app.core.tool_registry import ToolRegistry

        registry = ToolRegistry()
        assert registry.has_tool("identify_speaker")
        tool = registry.get_tool("identify_speaker")
        assert isinstance(tool, IdentifySpeakerTool)

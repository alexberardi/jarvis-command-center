"""
Tests for LlamaSmallUntrained prompt provider.

Verifies:
- Provider implements IJarvisPromptProvider correctly
- build_system_prompt produces valid output with expected sections
- use_tool_classifier returns True
- get_capabilities returns correct metadata
- Node context (room, user, voice_mode) is included
- Direct answer policy is rendered when applicable
"""

import pytest

from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider
from app.core.prompt_providers.small.untrained.llama_small_untrained import (
    LlamaSmallUntrained,
)


@pytest.fixture
def provider() -> LlamaSmallUntrained:
    return LlamaSmallUntrained()


@pytest.fixture
def sample_tools() -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_timer",
                "description": "Set a countdown timer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "duration_minutes": {
                            "type": "integer",
                            "description": "Timer duration in minutes",
                        }
                    },
                    "required": ["duration_minutes"],
                },
            },
        },
    ]


@pytest.fixture
def sample_node_context() -> dict:
    return {
        "room": "kitchen",
        "user": "alex",
        "voice_mode": "brief",
    }


@pytest.fixture
def sample_commands() -> list:
    return [
        {
            "command_name": "get_weather",
            "allow_direct_answer": False,
        },
        {
            "command_name": "calculate",
            "allow_direct_answer": True,
        },
    ]


class TestLlamaSmallUntrainedInterface:
    """Test that the provider implements IJarvisPromptProvider correctly."""

    def test_is_instance_of_interface(self, provider: LlamaSmallUntrained):
        assert isinstance(provider, IJarvisPromptProvider)

    def test_name_property(self, provider: LlamaSmallUntrained):
        assert provider.name == "LlamaSmallUntrained"

    def test_use_tool_classifier_true(self, provider: LlamaSmallUntrained):
        assert provider.use_tool_classifier is True

    def test_get_response_format_default(self, provider: LlamaSmallUntrained):
        assert provider.get_response_format() is None

    def test_parse_response_quirks_default(self, provider: LlamaSmallUntrained):
        assert provider.parse_response_quirks("some content") is None

    def test_get_capabilities(self, provider: LlamaSmallUntrained):
        caps = provider.get_capabilities()
        assert caps["provider_name"] == "LlamaSmallUntrained"
        assert caps["model_family"] == "llama"
        assert caps["size_tier"] == "small"
        assert caps["training_tier"] == "untrained"
        assert caps["use_tool_classifier"] is True


class TestLlamaSmallUntrainedPrompt:
    """Test system prompt generation."""

    def test_prompt_includes_identity(
        self,
        provider: LlamaSmallUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "Jarvis" in prompt
        assert "voice assistant" in prompt

    def test_prompt_includes_node_context(
        self,
        provider: LlamaSmallUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "room=kitchen" in prompt
        assert "user=alex" in prompt
        assert "style=brief" in prompt

    def test_prompt_includes_json_format(
        self,
        provider: LlamaSmallUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "JSON ONLY" in prompt
        assert "tool_calls" in prompt

    def test_prompt_includes_tools_section(
        self,
        provider: LlamaSmallUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "Tools:" in prompt
        # Tool names should appear somewhere in the prompt
        assert "get_weather" in prompt
        assert "set_timer" in prompt

    def test_prompt_with_direct_answer_policy(
        self,
        provider: LlamaSmallUntrained,
        sample_tools: list,
        sample_node_context: dict,
        sample_commands: list,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context,
            "America/New_York",
            sample_tools,
            sample_commands,
        )
        assert "Direct Answer Policy" in prompt
        assert "MUST call tools for" in prompt
        assert "get_weather" in prompt
        assert "Direct answers allowed for" in prompt
        assert "calculate" in prompt

    def test_prompt_without_commands(
        self,
        provider: LlamaSmallUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools, []
        )
        assert "Direct Answer Policy" not in prompt

    def test_prompt_with_empty_context(
        self,
        provider: LlamaSmallUntrained,
        sample_tools: list,
    ):
        prompt = provider.build_system_prompt({}, None, sample_tools)
        assert "room=unknown" in prompt
        assert "user=default" in prompt
        assert "style=brief" in prompt

    def test_prompt_returns_string(
        self,
        provider: LlamaSmallUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        result = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert isinstance(result, str)
        assert len(result) > 0

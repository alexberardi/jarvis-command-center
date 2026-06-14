"""
Tests for Qwen25MediumUntrained prompt provider.

Verifies:
- Provider implements IJarvisPromptProvider correctly
- build_system_prompt produces valid output with expected sections
- Tools are wrapped in <tools> XML tags (Qwen 2.5 chat template format)
- Tools are one-per-line JSON (not pretty-printed)
- Agent context (HA devices) is included when present
- use_tool_classifier returns True
- get_capabilities returns correct metadata
- get_response_format returns text mode
- parse_response transforms <tool_call> tags into Jarvis JSON
"""

import json

import pytest

from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider
from app.core.prompt_providers.medium.untrained.qwen25_medium_untrained import (
    Qwen25MediumUntrained,
)


@pytest.fixture
def provider() -> Qwen25MediumUntrained:
    return Qwen25MediumUntrained()


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


@pytest.fixture
def sample_node_context_with_agents() -> dict:
    return {
        "room": "living_room",
        "user": "alex",
        "voice_mode": "brief",
        "agents": {
            "home_assistant": {
                "light_controls": {
                    "Living Room": {
                        "entity_id": "light.living_room",
                        "state": "on",
                    }
                },
                "device_controls": {
                    "switch": [
                        {
                            "entity_id": "switch.fan",
                            "name": "Ceiling Fan",
                            "state": "off",
                        }
                    ],
                    "scene": [
                        {
                            "entity_id": "scene.movie_time",
                            "name": "Movie Time",
                        }
                    ],
                },
            }
        },
    }


class TestQwen25MediumUntrainedInterface:
    """Test that the provider implements IJarvisPromptProvider correctly."""

    def test_is_instance_of_interface(self, provider: Qwen25MediumUntrained):
        assert isinstance(provider, IJarvisPromptProvider)

    def test_name_property(self, provider: Qwen25MediumUntrained):
        assert provider.name == "Qwen25MediumUntrained"

    def test_use_tool_classifier_true(self, provider: Qwen25MediumUntrained):
        assert provider.use_tool_classifier is True

    def test_get_response_format_text_mode(self, provider: Qwen25MediumUntrained):
        fmt = provider.get_response_format()
        assert fmt == {"type": "text"}

    def test_get_capabilities(self, provider: Qwen25MediumUntrained):
        caps = provider.get_capabilities()
        assert caps["provider_name"] == "Qwen25MediumUntrained"
        assert caps["model_family"] == "qwen"
        assert caps["size_tier"] == "medium"
        assert caps["training_tier"] == "untrained"
        assert caps["use_tool_classifier"] is True
        assert caps["supports_native_tools"] is False

    def test_supports_native_tools_false(self, provider: Qwen25MediumUntrained):
        assert provider.supports_native_tools is False


class TestQwen25MediumUntrainedPrompt:
    """Test system prompt generation."""

    def test_prompt_includes_identity(
        self,
        provider: Qwen25MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "Jarvis" in prompt
        assert "function calling" in prompt

    def test_prompt_includes_node_context(
        self,
        provider: Qwen25MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "room=kitchen" in prompt
        # Speaker is no longer baked into the cached prefix — it's injected
        # per-turn as a trailing system message (build_speaker_context).
        assert "You are speaking with" not in prompt
        assert "style=brief" in prompt

    def test_prompt_includes_tools_xml_tags(
        self,
        provider: Qwen25MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "<tools>" in prompt
        assert "</tools>" in prompt

    def test_prompt_includes_tool_call_format(
        self,
        provider: Qwen25MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "<tool_call>" in prompt
        assert "</tool_call>" in prompt
        assert "brief spoken reply" in prompt
        # Should NOT fight the model's native format
        assert "JSON object ONLY" not in prompt

    def test_tools_are_one_per_line(
        self,
        provider: Qwen25MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        """Qwen's chat template expects one tool per line, not pretty-printed."""
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        # Find the <tools> block
        start = prompt.index("<tools>\n") + len("<tools>\n")
        end = prompt.index("\n</tools>")
        tools_content = prompt[start:end]
        lines = tools_content.strip().split("\n")
        assert len(lines) == 2  # Two tools, one per line
        # Each line should be valid JSON
        for line in lines:
            parsed = json.loads(line)
            assert parsed["type"] == "function"

    def test_prompt_includes_tool_schemas(
        self,
        provider: Qwen25MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert '"get_weather"' in prompt
        assert '"set_timer"' in prompt

    def test_prompt_includes_tools_section(
        self,
        provider: Qwen25MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "Tools:" in prompt
        assert "get_weather" in prompt
        assert "set_timer" in prompt

    def test_prompt_with_direct_answer_policy(
        self,
        provider: Qwen25MediumUntrained,
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
        provider: Qwen25MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools, []
        )
        assert "Direct Answer Policy" not in prompt

    def test_prompt_with_empty_context(
        self,
        provider: Qwen25MediumUntrained,
        sample_tools: list,
    ):
        prompt = provider.build_system_prompt({}, None, sample_tools)
        assert "room=unknown" in prompt
        # Default user is now omitted — the "speaking with" line is suppressed
        assert "You are speaking with" not in prompt
        assert "style=brief" in prompt

    def test_prompt_includes_agent_context(
        self,
        provider: Qwen25MediumUntrained,
        sample_tools: list,
        sample_node_context_with_agents: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context_with_agents, "America/New_York", sample_tools
        )
        assert "Living Room" in prompt
        assert "light.living_room" in prompt
        assert "Ceiling Fan" in prompt
        assert "switch.fan" in prompt
        assert "Movie Time" in prompt
        assert "scene.movie_time" in prompt

    def test_prompt_without_agent_context(
        self,
        provider: Qwen25MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "Available Light Controls" not in prompt
        assert "Available Switches" not in prompt

    def test_prompt_returns_string(
        self,
        provider: Qwen25MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        result = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_prompt_includes_anti_hallucination(
        self,
        provider: Qwen25MediumUntrained,
        sample_tools: list,
        sample_node_context: dict,
    ):
        prompt = provider.build_system_prompt(
            sample_node_context, "America/New_York", sample_tools
        )
        assert "NEVER fabricate" in prompt


class TestQwen25MediumUntrainedParseResponse:
    """Test parse_response for Qwen 2.5 output → Jarvis JSON transformation."""

    def test_single_tool_call(self, provider: Qwen25MediumUntrained):
        raw = '<tool_call>{"name":"get_weather","arguments":{"location":"NYC"}}</tool_call>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert parsed["message"] == ""
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "get_weather"
        assert parsed["tool_calls"][0]["arguments"] == {"location": "NYC"}
        assert parsed["error"] is None

    def test_tool_call_with_whitespace(self, provider: Qwen25MediumUntrained):
        raw = '<tool_call>\n  {"name":"set_timer","arguments":{"duration_minutes":5}}\n</tool_call>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "set_timer"

    def test_multiple_tool_calls(self, provider: Qwen25MediumUntrained):
        raw = (
            '<tool_call>{"name":"get_weather","arguments":{"location":"NYC"}}</tool_call>\n'
            '<tool_call>{"name":"set_timer","arguments":{"duration_minutes":5}}</tool_call>'
        )
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 2
        assert parsed["tool_calls"][0]["name"] == "get_weather"
        assert parsed["tool_calls"][1]["name"] == "set_timer"

    def test_tool_call_with_surrounding_text(self, provider: Qwen25MediumUntrained):
        """Qwen may emit text before tool call."""
        raw = 'I\'ll check the weather for you.\n<tool_call>{"name":"get_weather","arguments":{"location":"NYC"}}</tool_call>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "get_weather"

    def test_returns_none_for_clean_jarvis_json(self, provider: Qwen25MediumUntrained):
        raw = '{"message":"hello","tool_calls":[],"error":null}'
        result = provider.parse_response(raw)
        assert result is None

    def test_returns_none_for_jarvis_json_with_whitespace(self, provider: Qwen25MediumUntrained):
        raw = '  {"message":"hello","tool_calls":[],"error":null}  '
        result = provider.parse_response(raw)
        assert result is None

    def test_plain_text_wrapped_as_jarvis_json(self, provider: Qwen25MediumUntrained):
        raw = "The weather in NYC is sunny and 72 degrees."
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert parsed["message"] == raw
        assert parsed["tool_calls"] == []
        assert parsed["error"] is None

    def test_invalid_tool_call_json_skipped(self, provider: Qwen25MediumUntrained):
        raw = '<tool_call>not valid json</tool_call>'
        result = provider.parse_response(raw)
        # No valid tool calls parsed, falls through to plain text wrap
        parsed = json.loads(result)
        assert parsed["tool_calls"] == []

    def test_resolved_datetimes_string_normalized_to_array(self, provider: Qwen25MediumUntrained):
        raw = '<tool_call>{"name":"get_weather","arguments":{"city":"Miami","resolved_datetimes":"today"}}</tool_call>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert parsed["tool_calls"][0]["arguments"]["resolved_datetimes"] == ["today"]

    def test_resolved_datetimes_array_unchanged(self, provider: Qwen25MediumUntrained):
        raw = '<tool_call>{"name":"get_weather","arguments":{"city":"Miami","resolved_datetimes":["today","tomorrow"]}}</tool_call>'
        result = provider.parse_response(raw)
        parsed = json.loads(result)
        assert parsed["tool_calls"][0]["arguments"]["resolved_datetimes"] == ["today", "tomorrow"]


class TestSpeakerAgnosticPrompt:
    """The system prompt (cached prefix) must be byte-identical across speakers
    — speaker name + memories now live in a per-turn trailing block
    (build_speaker_context), not in the prompt. This is the prefix-cache-hit
    guarantee that removes the speaker-switch warmup/mismatch bug."""

    def test_prompt_identical_across_speakers(self, provider, sample_tools):
        ctx_alex = {
            "room": "kitchen", "speaker_name": "alex", "voice_mode": "brief",
            "user_memories": "- Likes coffee black",
        }
        ctx_bob = {
            "room": "kitchen", "speaker_name": "bob", "voice_mode": "brief",
            "user_memories": "- Allergic to peanuts",
        }
        p_alex = provider.build_system_prompt(ctx_alex, "America/New_York", sample_tools)
        p_bob = provider.build_system_prompt(ctx_bob, "America/New_York", sample_tools)
        assert p_alex == p_bob, "system prompt must not depend on the speaker"
        assert "alex" not in p_alex and "bob" not in p_alex
        # The actual memory CONTENT must not be in the cached prefix (it moves
        # to the per-turn speaker block). ("User Profile" as a phrase appears
        # in a static rule, so we check the memory text itself, not the label.)
        assert "Likes coffee black" not in p_alex
        assert "Allergic to peanuts" not in p_bob

    def test_speaker_context_carries_name_and_memories(self, provider):
        block = provider.build_speaker_context(
            {"speaker_name": "alex", "user_memories": "- Likes coffee black"}
        )
        assert "You are speaking with alex." in block
        assert "Likes coffee black" in block

    def test_speaker_context_empty_for_unknown(self, provider):
        assert provider.build_speaker_context({"voice_mode": "brief"}) == ""

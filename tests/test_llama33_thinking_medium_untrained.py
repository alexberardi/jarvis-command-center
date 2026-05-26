"""Tests for Llama33ThinkingMediumUntrained provider.

Focus areas:
- think_delimiters match what DavidAU's Claude-Haiku LoRA actually emits
  (``<think>``/``</think>``, NOT ``[[[thinking start]]]``)
- parse_response strips think blocks that the model wraps inside the JSON
  ``message`` field
- truncated mid-think output (finish_reason=length leaving ``{"message": "``)
  is collapsed to an empty envelope so TTS doesn't speak the JSON wrapper
- sanitize_text catches the same broken-JSON scaffolding when called
  on tool_call_parser's plain-text fallback
- user_message_suffix is /no_think so voice turns stay in token budget
"""

import json

import pytest

from app.core.prompt_providers.medium.untrained.llama33_thinking_medium_untrained import (
    Llama33ThinkingMediumUntrained,
)


@pytest.fixture
def provider() -> Llama33ThinkingMediumUntrained:
    return Llama33ThinkingMediumUntrained()


class TestThinkDelimiters:
    def test_uses_standard_think_markers(self, provider):
        # The model emits <think>/</think>, not the [[[...]]] form that
        # earlier model notes referenced. Pin this to catch regressions.
        assert provider.think_delimiters == ("<think>", "</think>")


class TestUserMessageSuffix:
    def test_appends_no_think(self, provider):
        # Disables thinking on tool-selection turns so the 256-token budget
        # isn't burnt mid-reasoning.
        assert provider.user_message_suffix == "/no_think"


class TestParseResponseTruncatedThink:
    def test_collapses_unclosed_think_in_message_field(self, provider):
        # Real-world capture: model puts <think> inside the JSON message
        # field, then hits max_tokens mid-thought. Without intervention,
        # the broken JSON falls through to plain-text fallback and TTS
        # speaks "open brace quote message colon".
        raw = (
            '{"message": "<think>This is a weather question about rain in '
            'the next 15 minutes. This matches get_weather, which handles'
        )
        result = provider.parse_response(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == {"message": "", "tool_calls": [], "error": None}

    def test_valid_json_with_balanced_think_passes_through(self, provider):
        # Happy path: model finished the think block AND closed the JSON.
        # parse_response should let the parent's passthrough handle it
        # (returns None so tool_call_parser extracts the message field).
        raw = (
            '{"message": "<think>weighing the question</think>It looks '
            'clear.", "tool_calls": [], "error": null}'
        )
        result = provider.parse_response(raw)
        # Parent returns None for already-Jarvis-JSON content (passthrough).
        assert result is None


class TestParseResponseToolCall:
    def test_extracts_tool_call_tag(self, provider):
        raw = (
            '<tool_call>{"name": "get_weather", '
            '"arguments": {"city": "Miami"}}</tool_call>'
        )
        result = provider.parse_response(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["tool_calls"] == [
            {"name": "get_weather", "arguments": {"city": "Miami"}}
        ]

    def test_strips_think_before_tool_call_extraction(self, provider):
        # If a tool call appears inside the reasoning block, it must not
        # be extracted as a real call — only the final tool_call should
        # win.
        raw = (
            "<think>I could call <tool_call>{\"name\": \"wrong\"}</tool_call>"
            " but actually...</think>"
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Miami"}}'
            '</tool_call>'
        )
        result = provider.parse_response(raw)
        assert result is not None
        parsed = json.loads(result)
        # Only the post-think call should be picked up
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "get_weather"


class TestSanitizeText:
    def test_strips_complete_think_block(self, provider):
        text = "<think>reasoning</think>The weather is clear."
        assert provider.sanitize_text(text) == "The weather is clear."

    def test_strips_unclosed_think_tail(self, provider):
        # If a downstream caller hands sanitize_text raw truncated content,
        # strip the unclosed <think>... tail.
        text = "Heads up: <think>cut off here"
        assert provider.sanitize_text(text) == "Heads up:"

    def test_collapses_broken_json_wrapper(self, provider):
        # Equivalent to parse_response's truncation guard, but for the
        # path where tool_call_parser already returned the raw string as
        # the plain-text fallback. Returning empty here means the
        # orchestrator hears "no message" instead of TTS-ing JSON.
        text = '{"message": "<think>truncated mid-thought'
        assert provider.sanitize_text(text) == ""

    def test_passes_clean_text_through(self, provider):
        text = "The weather is clear."
        assert provider.sanitize_text(text) == "The weather is clear."

    def test_strips_leaked_tool_call_scaffold(self, provider):
        text = 'Done. <tool_call>{"name": "foo"}</tool_call>'
        assert provider.sanitize_text(text) == "Done."

"""Tests for the ChatGPTOpenAI prompt provider (native tool calling).

This is the first provider with ``supports_native_tools=True`` — it exercises
command-center's native tool-calling surface (tools via the API ``tools`` param,
structured ``tool_calls`` in the response) rather than text ``<tool_call>`` tag
parsing. These are behavior-meaningful checks: they pin that the provider routes
tools natively, keeps tool *schemas* OUT of the (cached) system prompt, and is
discoverable so ``llm.interface='ChatGPTOpenAI'`` selects it.
"""

from __future__ import annotations

import pytest

from app.core.prompt_provider_factory import PromptProviderFactory
from app.core.prompt_providers.large.untrained.chatgpt_openai import ChatGPTOpenAI


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Set a countdown timer.",
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
    }
]

NODE_CONTEXT = {"room": "kitchen", "voice_mode": "brief", "speaker_name": "alex"}


class TestChatGPTOpenAIProvider:
    def test_name(self) -> None:
        assert ChatGPTOpenAI().name == "ChatGPTOpenAI"

    def test_supports_native_tools(self) -> None:
        # The whole point: native tool calling, not text-tag parsing.
        assert ChatGPTOpenAI().supports_native_tools is True

    def test_tool_classifier_disabled(self) -> None:
        # A capable cloud model routes natively; no fastText hints.
        assert ChatGPTOpenAI().use_tool_classifier is False

    def test_build_tools_returns_openai_function_format(self) -> None:
        built = ChatGPTOpenAI().build_tools(SAMPLE_TOOLS)
        assert isinstance(built, list) and len(built) == 1
        assert built[0]["type"] == "function"
        assert built[0]["function"]["name"] == "set_timer"

    def test_system_prompt_carries_room_context(self) -> None:
        prompt = ChatGPTOpenAI().build_system_prompt(
            NODE_CONTEXT, "America/New_York", SAMPLE_TOOLS, available_commands=[]
        )
        assert "room=kitchen" in prompt
        assert "style=brief" in prompt

    def test_system_prompt_does_not_embed_tool_schemas(self) -> None:
        # Tools are delivered via the native `tools` API param, so their schemas
        # must NOT be duplicated in the (cached) system prompt. Guard against a
        # regression that re-embeds them (token waste + cache busting).
        prompt = ChatGPTOpenAI().build_system_prompt(
            NODE_CONTEXT, "America/New_York", SAMPLE_TOOLS, available_commands=[]
        )
        assert "<tools>" not in prompt
        assert "duration_minutes" not in prompt  # a tool param name

    def test_capabilities_shape(self) -> None:
        caps = ChatGPTOpenAI().get_capabilities()
        assert caps["provider_name"] == "ChatGPTOpenAI"
        assert caps["model_family"] == "openai"
        assert caps["supports_native_tools"] is True
        assert caps["use_tool_classifier"] is False


class TestChatGPTOpenAIDiscovery:
    def test_factory_discovers_by_name(self) -> None:
        provider = PromptProviderFactory.create_provider("ChatGPTOpenAI")
        assert isinstance(provider, ChatGPTOpenAI)

    @pytest.mark.parametrize("alias", ["chatgptopenai", "CHATGPTOPENAI", "ChatGptOpenAI"])
    def test_factory_match_is_case_insensitive(self, alias: str) -> None:
        provider = PromptProviderFactory.create_provider(alias)
        assert isinstance(provider, ChatGPTOpenAI)

    def test_listed_in_available_providers(self) -> None:
        assert "ChatGPTOpenAI" in PromptProviderFactory.get_available_providers()

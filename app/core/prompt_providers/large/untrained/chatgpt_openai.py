"""
ChatGPTOpenAI - Prompt provider for OpenAI-compatible chat models.

The natural fit for a real OpenAI-compatible model (e.g. a gpt-4.1-nano-class
cloud model served via llm-proxy's REST backend): it uses NATIVE tool calling
rather than text ``<tool_call>`` tag parsing.

- ``supports_native_tools = True`` -> tools are passed to the model via the API
  ``tools`` parameter with ``tool_choice="auto"``; the model returns structured
  ``tool_calls`` (read from ``finish_reason="tool_calls"``).
- Because tools are passed natively, the system prompt is deliberately CONCISE:
  it carries identity/room context + routing rules, NOT the full tool schemas
  (those would just duplicate the native ``tools`` payload and waste tokens).
- ``use_tool_classifier = False``: a capable cloud model routes natively and
  does not need fastText hints.
- ``parse_response`` / ``get_response_format`` are unused on the native path
  (the engine reads structured ``tool_calls`` directly), so they keep defaults.

This is the FIRST provider to exercise command-center's native tool-calling
path end-to-end — historically only the text path was used.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider
from app.core.prompt_providers.shared.context_builders import (
    build_agent_context_summary,
    build_direct_answer_section,
)
from app.core.prompt_providers.shared.core_rules import (
    ANTI_HALLUCINATION_MANDATE,
    build_fallback_line,
    build_rules_block,
)
from app.core.tool_builder import ToolBuilder

logger = logging.getLogger("uvicorn")


class ChatGPTOpenAI(IJarvisPromptProvider):
    """Prompt provider for an OpenAI-compatible model using native tool calling."""

    @property
    def name(self) -> str:
        return "ChatGPTOpenAI"

    @property
    def use_tool_classifier(self) -> bool:
        # A capable cloud model decides routing from the native tool schemas;
        # fastText hints are unnecessary.
        return False

    @property
    def supports_native_tools(self) -> bool:
        return True

    def build_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return clean OpenAI-format function tool definitions.

        Called by the tool-execution engine (native path) with the
        jarvis-extension-stripped tools; ToolBuilder.build emits the standard
        ``{"type": "function", "function": {...}}`` shape OpenAI expects.
        """
        return ToolBuilder.build(tools)

    def build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build a concise system prompt.

        Tool *schemas* are delivered via the native ``tools`` API parameter, so
        they are intentionally NOT embedded here. The prompt carries identity +
        room/style context, the shared routing rules, agent (device) context,
        the direct-answer hints, and the fallback line.
        """
        available_commands = available_commands or []
        node_context = node_context or {}

        identity: str = self.build_context_header(node_context)
        rules: str = build_rules_block()
        fallback: str = build_fallback_line()
        agent_context_section: str = build_agent_context_summary(node_context)
        direct_answer_section: str = build_direct_answer_section(available_commands)

        system_prompt: str = f"""{identity}

You are a function-calling voice assistant. Use the provided functions to act on the user's request: call one or more functions when the request maps to them, and always include every required parameter — use sensible defaults from context when the user does not state them explicitly. {ANTI_HALLUCINATION_MANDATE}

{rules}
{agent_context_section}
{fallback}
{direct_answer_section}"""

        logger.info(
            "Built ChatGPTOpenAI system prompt: %d chars, %d tools (native)",
            len(system_prompt),
            len(tools),
        )
        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("System prompt (full):\n%s", system_prompt)

        return system_prompt

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "openai",
            "size_tier": "large",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }

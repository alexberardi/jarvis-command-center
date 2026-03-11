"""
HermesMediumUntrained - Prompt provider for Hermes 3 Llama 3.1 8B Instruct.

Optimized for NousResearch Hermes 3 (Q4_K_M GGUF) with text-based tool calling.

Key features:
- Tools presented in <tools> XML tags (Hermes's fine-tuned format)
- Concise rules leveraging Hermes's function-calling training
- parse_response transforms <tool_call> XML tags into Jarvis JSON
- supports_native_tools=False (text-based): set to True when backend model
  reliably uses structured tool_calls via llama-cpp-python's tools parameter
- build_tools() ready for native path via ToolBuilder
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider
from app.core.prompt_providers.shared.context_builders import (
    build_agent_context_summary,
    build_direct_answer_section,
)
from app.core.prompt_providers.shared.tool_formatters import format_tools_for_prompt
from app.core.tool_builder import ToolBuilder

logger = logging.getLogger("uvicorn")

# Patterns for stripping Hermes-native tags from responses
_TOOL_CALL_TAG_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)
_SCRATCH_PAD_RE = re.compile(
    r"<scratch_pad>.*?</scratch_pad>", re.DOTALL
)


class HermesMediumUntrained(IJarvisPromptProvider):
    """
    Prompt provider for Hermes 3 Llama 3.1 8B Instruct (untrained).

    Strategy:
    - Tools in <tools> XML tags (Hermes's fine-tuned format)
    - Concise rules leveraging Hermes's function-calling training
    - Agent context (HA devices) included for device awareness
    - Primary examples only to save context window
    - fastText classifier enabled for routing hints
    """

    @property
    def name(self) -> str:
        return "HermesMediumUntrained"

    @property
    def use_tool_classifier(self) -> bool:
        return True

    @property
    def supports_native_tools(self) -> bool:
        # chatml-function-calling causes extreme latency (15-30s vs 1.3s)
        # and hallucinated parameters. Text-based path is far superior
        # for Hermes 3 Q4_K_M via llama-cpp-python.
        return False

    def build_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build OpenAI-format tool definitions using ToolBuilder."""
        return ToolBuilder.build(tools)

    @staticmethod
    def _build_tools_xml(tools: List[Dict[str, Any]]) -> str:
        """Build Hermes-style <tools> XML block from tool definitions."""
        clean_tools: List[Dict[str, Any]] = ToolBuilder.build(tools)
        if not clean_tools:
            return "<tools>\n</tools>"
        tool_json: str = json.dumps(clean_tools, indent=2)
        return f"<tools>\n{tool_json}\n</tools>"

    def build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build system prompt using Hermes's native <tools> XML format.

        Tools are presented as OpenAI-compatible JSON schemas inside <tools>
        tags to leverage Hermes's function-calling fine-tuning, but responses
        are required in Jarvis's JSON format for ToolCallParser compatibility.
        """
        available_commands = available_commands or []
        node_context = node_context or {}

        room: str = node_context.get("room", "unknown")
        user: str = node_context.get("speaker_name") or node_context.get("user", "default")
        voice_mode: str = node_context.get("voice_mode", "brief")
        user_memories: str = node_context.get("user_memories", "")

        # Shared sections
        direct_answer_section: str = build_direct_answer_section(available_commands)
        agent_context_section: str = build_agent_context_summary(node_context)

        # Tool descriptions with primary examples only (for intent guidance)
        tools_section: str = format_tools_for_prompt(
            tools, available_commands, primary_examples_only=True
        )

        # Build <tools> XML block for Hermes's fine-tuned format
        tools_xml: str = HermesMediumUntrained._build_tools_xml(tools)

        # Build memory block
        memory_block: str = ""
        if user_memories:
            memory_block = f"\nAbout {user}:\n{user_memories}\n"

        system_prompt: str = f"""You are Jarvis, a function calling voice assistant.
Context: room={room}, user={user}, style={voice_mode}
{memory_block}

You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.

{tools_xml}

Rules:
- Call ONE tool at a time to fulfill requests.
- Pick the tool that best matches intent; use get_command_utterance_examples if unsure.
- Extract parameters from the user's words; only request validation if required params are truly missing/ambiguous.
- For date parameters like resolved_datetimes, use natural words: "today", "tomorrow", "day_after_tomorrow", "this_weekend", "this_year". NEVER convert to ISO dates or timestamps.
- Always populate required tool parameters from the user's request.
- For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{{"name": "<function-name>", "arguments": {{"<arg-name>": "<arg-value>"}}, "failure_message": "<brief spoken response if this call fails>"}}
</tool_call>
{direct_answer_section}
{agent_context_section}
For final answers with no tool needed, respond with a brief spoken reply.

Tools:
{tools_section}
"""

        logger.info(
            "Built HermesMediumUntrained system prompt: %d chars, %d tools",
            len(system_prompt),
            len(tools),
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("System prompt (full):\n%s", system_prompt)

        return system_prompt

    def get_response_format(self) -> Optional[Dict[str, Any]]:
        """Return text mode — Hermes outputs <tool_call> tags, not JSON."""
        return {"type": "text"}

    def parse_response(self, raw_content: str) -> Optional[str]:
        """
        Transform Hermes native output into Jarvis JSON format.

        Hermes emits tool calls as <tool_call>{"name":"x","arguments":{...}}</tool_call>
        and may include <scratch_pad> blocks for chain-of-thought. This method:
        1. Strips <scratch_pad> blocks
        2. Extracts ALL <tool_call> blocks and builds Jarvis JSON
        3. Wraps plain text responses as Jarvis JSON messages
        4. Returns None for content already in Jarvis JSON format (passthrough)

        Returns:
            Transformed Jarvis JSON string, or None if no transformation needed.
        """
        cleaned: str = raw_content

        # Strip <scratch_pad>...</scratch_pad> blocks
        had_scratch_pad: bool = bool(_SCRATCH_PAD_RE.search(cleaned))
        cleaned = _SCRATCH_PAD_RE.sub("", cleaned)

        # Extract ALL <tool_call>...</tool_call> blocks
        tool_call_matches = _TOOL_CALL_TAG_RE.findall(cleaned)
        if tool_call_matches:
            parsed_calls: list[Dict[str, Any]] = []
            for match in tool_call_matches:
                try:
                    call_obj = json.loads(match.strip())
                    parsed_calls.append(call_obj)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse tool_call JSON: %s", match[:100])
            if parsed_calls:
                jarvis_json: Dict[str, Any] = {
                    "message": "",
                    "tool_calls": parsed_calls,
                    "error": None,
                }
                return json.dumps(jarvis_json)

        # Content was modified (scratch_pad stripped) but no tool calls
        cleaned = cleaned.strip()
        if had_scratch_pad and cleaned:
            # Check if remaining content is already valid Jarvis JSON
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "tool_calls" in parsed:
                    return cleaned
            except json.JSONDecodeError:
                pass
            # Wrap plain text as Jarvis JSON message
            return json.dumps({
                "message": cleaned,
                "tool_calls": [],
                "error": None,
            })

        # Check if content is already Jarvis JSON (passthrough)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                return None
        except json.JSONDecodeError:
            pass

        # Plain text response (no tags, not JSON) — wrap as Jarvis message
        if cleaned and cleaned != raw_content.strip():
            return json.dumps({
                "message": cleaned,
                "tool_calls": [],
                "error": None,
            })

        # No JSON, no tags, unchanged — wrap plain text
        if cleaned and not cleaned.startswith("{"):
            return json.dumps({
                "message": cleaned,
                "tool_calls": [],
                "error": None,
            })

        return None

    def build_training_prompt(self, voice_command: str) -> str:
        """Build training prompt matching Hermes's inference system prompt."""
        return (
            "You are a function calling AI model. "
            "For each function call return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags as follows:\n"
            "<tool_call>\n"
            '{"name": "<function-name>", "arguments": {"<arg-name>": "<arg-value>"}, "failure_message": "<brief spoken response if this call fails>"}\n'
            "</tool_call>\n"
            f"User: {voice_command}\n"
            "Assistant:"
        )

    def build_training_completion(self, tool_call: Dict[str, Any]) -> str:
        """Format as <tool_call> XML tags matching Hermes's fine-tuned output."""
        return f" <tool_call>\n{json.dumps(tool_call)}\n</tool_call>"

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "hermes",
            "size_tier": "medium",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }


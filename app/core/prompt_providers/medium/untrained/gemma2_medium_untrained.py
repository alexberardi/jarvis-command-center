"""
Gemma2MediumUntrained - Prompt provider for Google Gemma 2 9B Instruct.

Optimized for Gemma 2 9B Instruct (Q4_K_M GGUF) with text-based tool calling.

Key features:
- Tools presented as pretty-printed JSON schemas (no native tool format)
- Model instructed to emit <tool_call>{"name": ..., "arguments": ...}</tool_call>
- parse_response transforms <tool_call> XML tags into Jarvis JSON
- supports_native_tools=False (text-based): model outputs tool_call tags, not
  structured tool_calls via llama-cpp-python's tools parameter
- build_tools() ready for native path via ToolBuilder
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider
from app.core.prompt_providers.shared.context_builders import (
    build_agent_context_section,
    build_direct_answer_section,
)
from app.core.prompt_providers.shared.core_rules import (
    ANTI_HALLUCINATION_MANDATE,
    build_fallback_line,
    build_identity_header,
    build_rules_block,
)
from app.core.prompt_providers.shared.tool_formatters import format_tools_for_prompt
from app.core.tool_builder import ToolBuilder

logger = logging.getLogger("uvicorn")

# Pattern for <tool_call>{"name":...,"arguments":...}</tool_call> output
_TOOL_CALL_TAG_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)

# Parameters that are always arrays — normalize string values to single-element lists
_ARRAY_PARAMS = frozenset({"resolved_datetimes"})


class Gemma2MediumUntrained(IJarvisPromptProvider):
    """
    Prompt provider for Google Gemma 2 9B Instruct (untrained).

    Strategy:
    - Tools as pretty-printed JSON schemas (Gemma 2 has no fine-tuned tool format)
    - Model instructed to emit <tool_call> tags for function calls
    - Agent context (HA devices) included for device awareness
    - Primary examples only to save context window
    - fastText classifier enabled for routing hints
    """

    @property
    def name(self) -> str:
        return "Gemma2MediumUntrained"

    @property
    def use_tool_classifier(self) -> bool:
        return True

    @property
    def supports_native_tools(self) -> bool:
        # Text-based <tool_call> format — Gemma 2 has no fine-tuned
        # function-calling format like Hermes or Llama 3.1.
        return False

    def build_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build OpenAI-format tool definitions using ToolBuilder."""
        return ToolBuilder.build(tools)

    def build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build system prompt for Gemma 2 9B Instruct.

        Tools are presented as pretty-printed JSON schemas (no native tool
        format wrapper). The model is instructed to emit calls using
        <tool_call> XML tags.
        """
        available_commands = available_commands or []
        node_context = node_context or {}

        room: str = node_context.get("room", "unknown")
        user: str = node_context.get("user", "default")
        voice_mode: str = node_context.get("voice_mode", "brief")

        # Shared sections
        direct_answer_section: str = build_direct_answer_section(available_commands)
        agent_context_section: str = build_agent_context_section(node_context)

        # Tool descriptions with primary examples only (for intent guidance)
        tools_section: str = format_tools_for_prompt(
            tools, available_commands, primary_examples_only=True
        )

        # Build pretty-printed JSON schemas for tools (no native wrapper)
        clean_tools: List[Dict[str, Any]] = ToolBuilder.build(tools)
        tool_json: str = json.dumps(clean_tools, indent=2) if clean_tools else "[]"

        # Shared header
        identity: str = build_identity_header(room, user, voice_mode)

        # Shared rules (Gemma2: default param_names_rule, default terminology)
        rules: str = build_rules_block()

        # Shared fallback
        fallback: str = build_fallback_line()

        system_prompt: str = f"""{identity}

You are a function calling AI model. You are provided with function signatures below. Always include all required parameters — use sensible defaults from context when the user does not state them explicitly. {ANTI_HALLUCINATION_MANDATE}

Available functions:
{tool_json}

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": {{"<arg-name>": "<arg-value>"}}}}
</tool_call>

{rules}
{direct_answer_section}
{agent_context_section}
{fallback}

Tools:
{tools_section}
"""

        logger.info(
            "Built Gemma2MediumUntrained system prompt: %d chars, %d tools",
            len(system_prompt),
            len(tools),
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("System prompt (full):\n%s", system_prompt)

        return system_prompt

    def get_response_format(self) -> Optional[Dict[str, Any]]:
        """Return text mode — Gemma 2 outputs <tool_call> tags, not JSON."""
        return {"type": "text"}

    def parse_response(self, raw_content: str) -> Optional[str]:
        """
        Transform Gemma 2 output into Jarvis JSON format.

        Gemma 2 is instructed to emit tool calls as
        <tool_call>{"name":"x","arguments":{...}}</tool_call>.
        This method:
        1. Extracts ALL <tool_call> blocks and builds Jarvis JSON
        2. Wraps plain text responses as Jarvis JSON messages
        3. Returns None for content already in Jarvis JSON format (passthrough)

        Returns:
            Transformed Jarvis JSON string, or None if no transformation needed.
        """
        cleaned: str = raw_content.strip()

        # Extract ALL <tool_call>...</tool_call> blocks
        tool_call_matches = _TOOL_CALL_TAG_RE.findall(cleaned)
        if tool_call_matches:
            parsed_calls: list[Dict[str, Any]] = []
            for match in tool_call_matches:
                try:
                    call_obj = json.loads(match.strip())
                except json.JSONDecodeError:
                    logger.warning("Failed to parse tool_call JSON: %s", match[:100])
                    continue
                # Normalize array parameters: wrap string values in a list
                arguments = call_obj.get("arguments", {})
                if isinstance(arguments, dict):
                    for key in _ARRAY_PARAMS:
                        if key in arguments and isinstance(arguments[key], str):
                            arguments[key] = [arguments[key]]
                parsed_calls.append(call_obj)
            if parsed_calls:
                jarvis_json: Dict[str, Any] = {
                    "message": "",
                    "tool_calls": parsed_calls,
                    "error": None,
                }
                return json.dumps(jarvis_json)

        # Check if content is already Jarvis JSON (passthrough)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                return None
        except json.JSONDecodeError:
            pass

        # Plain text response — wrap as Jarvis message
        if cleaned and not cleaned.startswith("{"):
            return json.dumps({
                "message": cleaned,
                "tool_calls": [],
                "error": None,
            })

        return None

    def build_training_prompt(self, voice_command: str) -> str:
        """Build training prompt matching Gemma 2's inference system prompt."""
        return (
            "You are a function calling AI model. "
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": "<function-name>", "arguments": {"<arg-name>": "<arg-value>"}}\n'
            "</tool_call>\n"
            f"User: {voice_command}\n"
            "Assistant:"
        )

    def build_training_completion(self, tool_call: Dict[str, Any]) -> str:
        """Format as <tool_call> XML tags matching Gemma 2's instructed output."""
        return f" <tool_call>\n{json.dumps(tool_call)}\n</tool_call>"

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "gemma",
            "size_tier": "medium",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }

"""
Qwen25MediumUntrained - Prompt provider for Qwen 2.5 7B Instruct.

Optimized for Qwen 2.5 7B Instruct (Q4_K_M GGUF) with text-based tool calling.

Key features:
- Tools presented as one-per-line JSON in <tools> XML tags (Qwen's chat template format)
- Model emits <tool_call>{"name": ..., "arguments": ...}</tool_call> responses
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
from app.core.prompt_providers.shared.tool_formatters import format_tools_for_prompt
from app.core.tool_builder import ToolBuilder

logger = logging.getLogger("uvicorn")

# Pattern for Qwen 2.5 tool call format: <tool_call>{"name":...,"arguments":...}</tool_call>
_TOOL_CALL_TAG_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)

# Parameters that are always arrays — normalize string values to single-element lists
_ARRAY_PARAMS = frozenset({"resolved_datetimes"})


class Qwen25MediumUntrained(IJarvisPromptProvider):
    """
    Prompt provider for Qwen 2.5 7B Instruct (untrained).

    Strategy:
    - Tools as one-per-line JSON in <tools> tags (Qwen's chat template format)
    - Model emits <tool_call> tags matching its fine-tuned function-calling training
    - Agent context (HA devices) included for device awareness
    - Primary examples only to save context window
    - fastText classifier enabled for routing hints
    """

    @property
    def name(self) -> str:
        return "Qwen25MediumUntrained"

    @property
    def use_tool_classifier(self) -> bool:
        return True

    @property
    def supports_native_tools(self) -> bool:
        # Text-based <tool_call> format is more reliable than
        # structured tool_calls via llama-cpp-python for this model.
        return False

    def build_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build OpenAI-format tool definitions using ToolBuilder."""
        return ToolBuilder.build(tools)

    @staticmethod
    def _build_tools_block(tools: List[Dict[str, Any]]) -> str:
        """Build Qwen-style <tools> block with one tool definition per line.

        Matches Qwen 2.5's chat template: each tool is a single-line JSON
        object inside <tools></tools> tags, not pretty-printed.
        """
        clean_tools: List[Dict[str, Any]] = ToolBuilder.build(tools)
        if not clean_tools:
            return "<tools>\n</tools>"
        lines: List[str] = [json.dumps(t, separators=(",", ":")) for t in clean_tools]
        return "<tools>\n" + "\n".join(lines) + "\n</tools>"

    def build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build system prompt using Qwen 2.5's native tool calling format.

        Tools are presented as one-per-line JSON inside <tools> tags,
        matching Qwen 2.5's chat template. The model is instructed to emit
        calls using <tool_call> XML tags.
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

        # Build <tools> block matching Qwen 2.5's chat template
        tools_block: str = Qwen25MediumUntrained._build_tools_block(tools)

        system_prompt: str = f"""You are Jarvis, a function calling voice assistant.
Context: room={room}, user={user}, style={voice_mode}

You are a function calling AI model. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. You MUST call a function for any request that matches an available tool — NEVER fabricate data, pretend to perform actions, or answer from memory for weather, sports, calendar, timers, searches, or any tool-covered domain.

You are provided with function signatures within <tools></tools> XML tags:
{tools_block}

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": {{"<arg-name>": "<arg-value>"}}}}
</tool_call>

Rules:
- Call ONE function at a time to fulfill requests.
- Use the actual parameter names from the function schema above.
- Pick the function that best matches intent; use get_command_utterance_examples if unsure.
- Extract parameters from the user's words; only request clarification if required params are truly missing/ambiguous.
- For date parameters like resolved_datetimes, use natural words: "today", "tomorrow", "day_after_tomorrow", "this_weekend", "this_year". NEVER convert to ISO dates or timestamps.
- Always populate required function parameters from the user's request.
{direct_answer_section}
{agent_context_section}
Only respond with a brief spoken reply for general knowledge questions, greetings, or jokes that have NO matching tool.

Tools:
{tools_section}
"""

        logger.info(
            "Built Qwen25MediumUntrained system prompt: %d chars, %d tools",
            len(system_prompt),
            len(tools),
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("System prompt (full):\n%s", system_prompt)

        return system_prompt

    def get_response_format(self) -> Optional[Dict[str, Any]]:
        """Return text mode — Qwen 2.5 outputs <tool_call> tags, not JSON."""
        return {"type": "text"}

    def parse_response(self, raw_content: str) -> Optional[str]:
        """
        Transform Qwen 2.5 native output into Jarvis JSON format.

        Qwen 2.5 emits tool calls as <tool_call>{"name":"x","arguments":{...}}</tool_call>.
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

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "qwen",
            "size_tier": "medium",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }

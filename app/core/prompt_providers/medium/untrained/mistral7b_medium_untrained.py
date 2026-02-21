"""
Mistral7bMediumUntrained - Prompt provider for Mistral 7B Instruct v0.3.

Optimized for Mistral 7B Instruct v0.3 (Q4_K_M GGUF) with text-based tool calling.

Key features:
- Tools presented using Mistral's [AVAILABLE_TOOLS] format
- Model emits [TOOL_CALLS] with JSON array of function calls
- parse_response transforms [TOOL_CALLS] output into Jarvis JSON
- supports_native_tools=False (text-based): model outputs tool call tokens,
  not structured tool_calls via llama-cpp-python's tools parameter
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

# Pattern for Mistral's native tool call format: [TOOL_CALLS] [{...}]
# The model emits a JSON array after the [TOOL_CALLS] token.
_TOOL_CALLS_RE = re.compile(
    r"\[TOOL_CALLS\]\s*(\[.*\])", re.DOTALL
)

# Fallback: model emits a single JSON object (not array) after [TOOL_CALLS]
_TOOL_CALLS_SINGLE_RE = re.compile(
    r"\[TOOL_CALLS\]\s*(\{.*\})", re.DOTALL
)

# Parameters that are always arrays — normalize string values to single-element lists
_ARRAY_PARAMS = frozenset({"resolved_datetimes"})


class Mistral7bMediumUntrained(IJarvisPromptProvider):
    """
    Prompt provider for Mistral 7B Instruct v0.3 (untrained).

    Strategy:
    - Tools in [AVAILABLE_TOOLS] format (Mistral's fine-tuned token format)
    - Model emits [TOOL_CALLS] with JSON array of calls
    - Agent context (HA devices) included for device awareness
    - Primary examples only to save context window
    - fastText classifier enabled for routing hints
    """

    @property
    def name(self) -> str:
        return "Mistral7bMediumUntrained"

    @property
    def use_tool_classifier(self) -> bool:
        return True

    @property
    def supports_native_tools(self) -> bool:
        # Text-based [TOOL_CALLS] format matches Mistral's fine-tuning.
        return False

    def build_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build OpenAI-format tool definitions using ToolBuilder."""
        return ToolBuilder.build(tools)

    @staticmethod
    def _build_available_tools(tools: List[Dict[str, Any]]) -> str:
        """Build Mistral-style [AVAILABLE_TOOLS] block.

        Matches Mistral v0.3's chat template: tools as a JSON array between
        [AVAILABLE_TOOLS] and [/AVAILABLE_TOOLS] tokens.
        """
        clean_tools: List[Dict[str, Any]] = ToolBuilder.build(tools)
        if not clean_tools:
            return "[AVAILABLE_TOOLS] [] [/AVAILABLE_TOOLS]"
        tools_json: str = json.dumps(clean_tools, separators=(",", ":"))
        return f"[AVAILABLE_TOOLS] {tools_json} [/AVAILABLE_TOOLS]"

    def build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build system prompt using Mistral's native tool calling format.

        Tools are presented in [AVAILABLE_TOOLS] blocks matching Mistral v0.3's
        fine-tuned function calling tokens.
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

        # Build [AVAILABLE_TOOLS] block matching Mistral's format
        available_tools: str = Mistral7bMediumUntrained._build_available_tools(tools)

        system_prompt: str = f"""You are Jarvis, a function calling voice assistant.
Context: room={room}, user={user}, style={voice_mode}

You are a function calling AI model. You are provided with function definitions below. Don't make assumptions about what values to plug into functions. You MUST call a function for any request that matches an available tool — NEVER fabricate data, pretend to perform actions, or answer from memory for weather, sports, calendar, timers, searches, or any tool-covered domain.

{available_tools}

To call a function, respond with [TOOL_CALLS] followed by a JSON array:
[TOOL_CALLS] [{{"name": "function_name", "arguments": {{"param": "value"}}}}]

Rules:
- Call ONE function at a time to fulfill requests.
- Use the actual parameter names from the function definitions above.
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
            "Built Mistral7bMediumUntrained system prompt: %d chars, %d tools",
            len(system_prompt),
            len(tools),
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("System prompt (full):\n%s", system_prompt)

        return system_prompt

    def get_response_format(self) -> Optional[Dict[str, Any]]:
        """Return text mode — Mistral outputs [TOOL_CALLS] tokens, not JSON."""
        return {"type": "text"}

    def parse_response(self, raw_content: str) -> Optional[str]:
        """
        Transform Mistral native output into Jarvis JSON format.

        Mistral v0.3 emits tool calls as:
            [TOOL_CALLS] [{"name": "fn", "arguments": {...}, "id": "abc123def"}]

        This method:
        1. Extracts [TOOL_CALLS] JSON array and builds Jarvis JSON
        2. Wraps plain text responses as Jarvis JSON messages
        3. Returns None for content already in Jarvis JSON format (passthrough)

        Returns:
            Transformed Jarvis JSON string, or None if no transformation needed.
        """
        cleaned: str = raw_content.strip()

        # Try to extract [TOOL_CALLS] [...] (JSON array)
        match = _TOOL_CALLS_RE.search(cleaned)
        if match:
            try:
                calls_array = json.loads(match.group(1))
                if isinstance(calls_array, list):
                    parsed_calls: list[Dict[str, Any]] = []
                    for call in calls_array:
                        if not isinstance(call, dict) or "name" not in call:
                            continue
                        arguments = call.get("arguments", {})
                        # Normalize array parameters
                        if isinstance(arguments, dict):
                            for key in _ARRAY_PARAMS:
                                if key in arguments and isinstance(arguments[key], str):
                                    arguments[key] = [arguments[key]]
                        parsed_calls.append({
                            "name": call["name"],
                            "arguments": arguments,
                        })
                    if parsed_calls:
                        return json.dumps({
                            "message": "",
                            "tool_calls": parsed_calls,
                            "error": None,
                        })
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse [TOOL_CALLS] JSON array: %s",
                    match.group(1)[:100],
                )

        # Fallback: single JSON object after [TOOL_CALLS]
        single_match = _TOOL_CALLS_SINGLE_RE.search(cleaned)
        if single_match:
            try:
                call = json.loads(single_match.group(1))
                if isinstance(call, dict) and "name" in call:
                    arguments = call.get("arguments", {})
                    if isinstance(arguments, dict):
                        for key in _ARRAY_PARAMS:
                            if key in arguments and isinstance(arguments[key], str):
                                arguments[key] = [arguments[key]]
                    return json.dumps({
                        "message": "",
                        "tool_calls": [{"name": call["name"], "arguments": arguments}],
                        "error": None,
                    })
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse [TOOL_CALLS] single JSON: %s",
                    single_match.group(1)[:100],
                )

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
            "model_family": "mistral",
            "size_tier": "medium",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }

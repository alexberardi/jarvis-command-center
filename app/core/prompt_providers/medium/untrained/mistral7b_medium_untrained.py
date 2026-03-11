"""
Mistral7bMediumUntrained - Prompt provider for Mistral 7B Instruct v0.3.

Optimized for Mistral 7B Instruct v0.3 (Q4_K_M GGUF) with text-based tool calling.

Key features:
- Uses chatml chat_format (mistral-instruct drops system messages)
- Tools presented in <tools> XML tags with <tool_call> output format
- parse_response handles both <tool_call> XML tags and Mistral's native
  [TOOL_CALLS] format for robustness
- supports_native_tools=False (text-based): model outputs tool call tags
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

# Primary: <tool_call>{"name":...,"arguments":...}</tool_call> output
_TOOL_CALL_TAG_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)

# Fallback: Mistral's native [TOOL_CALLS] [{...}] format
_TOOL_CALLS_RE = re.compile(
    r"\[TOOL_CALLS\]\s*(\[.*\])", re.DOTALL
)

# Fallback: single JSON object after [TOOL_CALLS]
_TOOL_CALLS_SINGLE_RE = re.compile(
    r"\[TOOL_CALLS\]\s*(\{.*\})", re.DOTALL
)

# Parameters that are always arrays — normalize string values to single-element lists
_ARRAY_PARAMS = frozenset({"resolved_datetimes"})


class Mistral7bMediumUntrained(IJarvisPromptProvider):
    """
    Prompt provider for Mistral 7B Instruct v0.3 (untrained).

    Strategy:
    - chatml chat_format (mistral-instruct drops system messages)
    - Tools in <tools> XML tags, same structure as Hermes/Gemma providers
    - Concrete example to demonstrate expected <tool_call> output
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
        # Text-based <tool_call> format. mistral-instruct chat format
        # drops system messages, so we use chatml instead.
        return False

    def build_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build OpenAI-format tool definitions using ToolBuilder."""
        return ToolBuilder.build(tools)

    @staticmethod
    def _build_tools_xml(tools: List[Dict[str, Any]]) -> str:
        """Build <tools> XML block from tool definitions."""
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
        Build system prompt for Mistral 7B Instruct v0.3.

        Mirrors the Hermes/Gemma prompt structure which achieves 91%+ accuracy:
        tools in <tools> XML tags, concise inline rules, <tool_call> output
        format. Requires chatml chat_format since mistral-instruct drops
        system messages.
        """
        available_commands = available_commands or []
        node_context = node_context or {}

        room: str = node_context.get("room", "unknown")
        user: str = node_context.get("user", "default")
        voice_mode: str = node_context.get("voice_mode", "brief")

        # Shared sections
        direct_answer_section: str = build_direct_answer_section(available_commands)
        agent_context_section: str = build_agent_context_summary(node_context)

        # Tool descriptions with primary examples only (for intent guidance)
        tools_section: str = format_tools_for_prompt(
            tools, available_commands, primary_examples_only=True
        )

        # Build <tools> XML block (same format as Hermes/Gemma)
        tools_xml: str = Mistral7bMediumUntrained._build_tools_xml(tools)

        system_prompt: str = f"""You are Jarvis, a function calling voice assistant.
Context: room={room}, user={user}, style={voice_mode}

You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You MUST call a function for any request that matches an available tool. Do not make assumptions about what values to plug into functions.

{tools_xml}

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{{"name": "<function-name>", "arguments": {{"<arg-name>": "<arg-value>"}}, "failure_message": "<brief spoken response if this call fails>"}}
</tool_call>

Example — if the user says "What's the weather in Miami?", respond ONLY with:
<tool_call>
{{"name": "get_weather", "arguments": {{"city": "Miami", "resolved_datetimes": ["today"]}}}}
</tool_call>

Rules:
- You MUST call a tool for weather, sports, calendar, timers, music, device control, web search, and all other tool-covered domains. NEVER answer these from memory.
- Call ONE tool at a time to fulfill requests.
- Pick the tool that best matches intent; use get_command_utterance_examples if unsure.
- Extract parameters from the user's words; only request validation if required params are truly missing/ambiguous.
- For date parameters like resolved_datetimes, use natural words: "today", "tomorrow", "day_after_tomorrow", "this_weekend", "this_year". NEVER convert to ISO dates or timestamps.
- Always populate required tool parameters from the user's request.
{direct_answer_section}
{agent_context_section}
For final answers with no tool needed, respond with a brief spoken reply.

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
        """Return text mode — Mistral outputs <tool_call> tags, not JSON."""
        return {"type": "text"}

    def parse_response(self, raw_content: str) -> Optional[str]:
        """
        Transform Mistral output into Jarvis JSON format.

        Handles two formats for robustness:
        1. Primary: <tool_call>{"name":"x","arguments":{...}}</tool_call>
        2. Fallback: [TOOL_CALLS] [{"name":"x","arguments":{...}}] (native Mistral)

        Returns:
            Transformed Jarvis JSON string, or None if no transformation needed.
        """
        cleaned: str = raw_content.strip()

        # Primary: extract ALL <tool_call>...</tool_call> blocks
        tool_call_matches = _TOOL_CALL_TAG_RE.findall(cleaned)
        if tool_call_matches:
            parsed_calls: list[Dict[str, Any]] = []
            for match in tool_call_matches:
                try:
                    call_obj = json.loads(match.strip())
                except json.JSONDecodeError:
                    logger.warning("Failed to parse tool_call JSON: %s", match[:100])
                    continue
                arguments = call_obj.get("arguments", {})
                if isinstance(arguments, dict):
                    for key in _ARRAY_PARAMS:
                        if key in arguments and isinstance(arguments[key], str):
                            arguments[key] = [arguments[key]]
                parsed_calls.append(call_obj)
            if parsed_calls:
                return json.dumps({
                    "message": "",
                    "tool_calls": parsed_calls,
                    "error": None,
                })

        # Fallback: [TOOL_CALLS] [...] (JSON array)
        match = _TOOL_CALLS_RE.search(cleaned)
        if match:
            try:
                calls_array = json.loads(match.group(1))
                if isinstance(calls_array, list):
                    parsed_calls = []
                    for call in calls_array:
                        if not isinstance(call, dict) or "name" not in call:
                            continue
                        arguments = call.get("arguments", {})
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

    def build_training_completion(self, tool_call: Dict[str, Any]) -> str:
        """Format as <tool_call> XML tags matching inference output format."""
        return f" <tool_call>\n{json.dumps(tool_call)}\n</tool_call>"

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "mistral",
            "size_tier": "medium",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }

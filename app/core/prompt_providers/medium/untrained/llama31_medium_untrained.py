"""
Llama31MediumUntrained - Prompt provider for Meta Llama 3.1 8B Instruct.

Optimized for Llama 3.1 8B Instruct (Q4_K_M GGUF) with text-based tool calling.

Key features:
- Tools presented as JSON schemas with <function=name>{args}</function> call format
- Leverages Llama 3.1's fine-tuned function-calling format
- parse_response transforms <function=name>{args}</function> tags into Jarvis JSON
- supports_native_tools=False (text-based): model outputs function tags, not
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

# Pattern for Llama 3.1 native function call format: <function=name>{...}</function>
# Also matches variant without '=' sign: <function>name{...}</function>
# Also matches malformed closing tag: <function=name>{...}<function> (missing slash)
_FUNCTION_CALL_RE = re.compile(
    r"<function[=>](\w+)>(.*?)</?\s*function>", re.DOTALL
)
# Fallback: model emits <function=name>{...} without closing </function> tag
_FUNCTION_CALL_UNCLOSED_RE = re.compile(
    r"<function[=>](\w+)>(\{.*)", re.DOTALL
)

# Parameters that are always arrays — normalize string values to single-element lists
_ARRAY_PARAMS = frozenset({"resolved_datetimes"})


class Llama31MediumUntrained(IJarvisPromptProvider):
    """
    Prompt provider for Meta Llama 3.1 8B Instruct (untrained).

    Strategy:
    - Tools as JSON schemas with <function=name>{args}</function> call format
    - Matches Llama 3.1's fine-tuned function-calling training
    - Agent context (HA devices) included for device awareness
    - Primary examples only to save context window
    - fastText classifier enabled for routing hints
    """

    @property
    def name(self) -> str:
        return "Llama31MediumUntrained"

    @property
    def use_tool_classifier(self) -> bool:
        return True

    @property
    def supports_native_tools(self) -> bool:
        # Text-based <function=...> format is more reliable than
        # structured tool_calls via llama-cpp-python for this model.
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
        Build system prompt using Llama 3.1's native <function=...> format.

        Tools are presented as JSON schemas and the model is instructed to
        emit calls using <function=name>{"param": "value"}</function> tags,
        matching Llama 3.1's function-calling fine-tuning.
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

        # Build JSON schemas for tools
        clean_tools: List[Dict[str, Any]] = ToolBuilder.build(tools)
        tool_json: str = json.dumps(clean_tools, indent=2) if clean_tools else "[]"

        system_prompt: str = f"""You are Jarvis, a function calling voice assistant.
Context: room={room}, user={user}, style={voice_mode}

You are a function calling AI model. You are provided with function signatures below. Don't make assumptions about what values to plug into functions. You MUST call a function for any request that matches an available tool — NEVER fabricate data, pretend to perform actions, or answer from memory for weather, sports, calendar, timers, searches, or any tool-covered domain.

To call a function, respond with:
<function=function_name>{{"arg_name": "value"}}</function>

For example, to get weather: <function=get_weather>{{"city": "Miami"}}</function>

Available functions:
{tool_json}

Rules:
- Call ONE function at a time to fulfill requests.
- Use the actual parameter names from the function schema above — NOT "param".
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
            "Built Llama31MediumUntrained system prompt: %d chars, %d tools",
            len(system_prompt),
            len(tools),
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("System prompt (full):\n%s", system_prompt)

        return system_prompt

    def get_response_format(self) -> Optional[Dict[str, Any]]:
        """Return text mode — Llama 3.1 outputs <function=...> tags, not JSON."""
        return {"type": "text"}

    def parse_response(self, raw_content: str) -> Optional[str]:
        """
        Transform Llama 3.1 native output into Jarvis JSON format.

        Llama 3.1 emits tool calls as <function=name>{"arg":"val"}</function>.
        This method:
        1. Extracts ALL <function=name>{...}</function> blocks and builds Jarvis JSON
        2. Wraps plain text responses as Jarvis JSON messages
        3. Returns None for content already in Jarvis JSON format (passthrough)

        Returns:
            Transformed Jarvis JSON string, or None if no transformation needed.
        """
        cleaned: str = raw_content.strip()

        # Extract ALL <function=name>{...}</function> blocks
        function_matches = _FUNCTION_CALL_RE.findall(cleaned)

        # Fallback: try unclosed <function=name>{...} (no </function> tag)
        if not function_matches:
            unclosed_match = _FUNCTION_CALL_UNCLOSED_RE.search(cleaned)
            if unclosed_match:
                function_matches = [unclosed_match.groups()]

        if function_matches:
            parsed_calls: list[Dict[str, Any]] = []
            for func_name, args_str in function_matches:
                try:
                    cleaned_args = args_str.strip().rstrip(";\"'")
                    arguments = json.loads(cleaned_args)
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse function args for %s: %s",
                        func_name,
                        args_str[:100],
                    )
                    continue
                # Unwrap {"param": {...}} nesting — the model sometimes wraps
                # real arguments inside a literal "param" key from the format
                # example. Unwrap only when "param" is the sole key and its
                # value is a dict (the actual arguments).
                if (
                    isinstance(arguments, dict)
                    and len(arguments) == 1
                    and "param" in arguments
                    and isinstance(arguments["param"], dict)
                ):
                    arguments = arguments["param"]
                # Normalize array parameters: wrap string values in a list
                if isinstance(arguments, dict):
                    for key in _ARRAY_PARAMS:
                        if key in arguments and isinstance(arguments[key], str):
                            arguments[key] = [arguments[key]]
                parsed_calls.append({
                    "name": func_name,
                    "arguments": arguments,
                })
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
            "model_family": "llama",
            "size_tier": "medium",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }

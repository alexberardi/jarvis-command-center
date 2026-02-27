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
    build_agent_context_summary,
    build_direct_answer_section,
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
        # function-calling format. chatml-function-calling causes extreme
        # latency (30s+ timeouts), so we use text-based tool calling.
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
        Build system prompt for Gemma 2 9B Instruct.

        Mirrors the Hermes prompt structure which achieves 91%+ accuracy:
        tools in <tools> XML tags, concise inline rules, <tool_call> output
        format. Gemma 2 has no function-calling fine-tuning, so the prompt
        includes a concrete example to demonstrate the expected format.
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

        # Build <tools> XML block (same format Hermes uses)
        tools_xml: str = Gemma2MediumUntrained._build_tools_xml(tools)

        system_prompt: str = f"""You are Jarvis, a function calling voice assistant.
Context: room={room}, user={user}, style={voice_mode}

You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You MUST call a function for any request that matches an available tool. Do not make assumptions about what values to plug into functions.

{tools_xml}

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{{"name": "<function-name>", "arguments": {{"<arg-name>": "<arg-value>"}}}}
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

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
    build_agent_context_summary,
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

        # Build <tools> block matching Qwen 2.5's chat template
        tools_block: str = Qwen25MediumUntrained._build_tools_block(tools)

        # Shared header (includes user memories if present)
        identity: str = build_identity_header(room, user, voice_mode, user_memories)

        # Shared rules (Qwen25: default param_names_rule, default terminology)
        rules: str = build_rules_block()

        # Shared fallback
        fallback: str = build_fallback_line()

        # Prompt is structured so stable content comes FIRST and tool-dependent
        # content comes LAST.  llama.cpp caches the KV state by prefix — the
        # longer the shared prefix between requests, the more tokens are reused.
        # When the tool router prunes tools for a high-confidence match, only
        # the suffix changes while the stable prefix (~700 tokens) stays cached.
        system_prompt: str = f"""{identity}

You are a function calling AI model. You may call one or more functions to assist with the user query. Always include all required parameters — use sensible defaults from context when the user does not state them explicitly. {ANTI_HALLUCINATION_MANDATE}

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": {{"<arg-name>": "<arg-value>"}}, "failure_message": "<brief spoken response if this call fails>"}}
</tool_call>

{rules}
{agent_context_section}
{fallback}

You are provided with function signatures within <tools></tools> XML tags:
{tools_block}
{direct_answer_section}
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

        # Check if content is already Jarvis JSON (passthrough) or a bare tool call
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                if "tool_calls" in parsed:
                    return None  # Already Jarvis JSON
                # Bare tool call: {"name": "...", "arguments": {...}}
                if "name" in parsed and "arguments" in parsed:
                    jarvis_json = {
                        "message": "",
                        "tool_calls": [parsed],
                        "error": None,
                    }
                    return json.dumps(jarvis_json)
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

    def build_training_system_prompt(self) -> str:
        """Training system prompt matching Qwen 2.5 inference structure.

        Includes the same structural cues (identity, rules, format instructions)
        the model sees at inference. Omits dynamic parts (tools block, agent
        context) that vary per request. This ensures the adapter learns
        attention patterns compatible with the full inference prompt.
        """
        rules: str = build_rules_block()
        fallback: str = build_fallback_line()
        return (
            "You are Jarvis, a function calling voice assistant.\n"
            "Context: room=unknown, user=default, style=brief\n\n"
            "You are a function calling AI model. You may call one or more "
            "functions to assist with the user query. Always include all "
            "required parameters — use sensible defaults from context when "
            f"the user does not state them explicitly. {ANTI_HALLUCINATION_MANDATE}\n\n"
            "For each function call, return a json object with function name "
            "and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": "<function-name>", "arguments": {"<arg-name>": "<arg-value>"}, "failure_message": "<brief spoken response if this call fails>"}\n'
            "</tool_call>\n\n"
            f"{rules}\n"
            f"{fallback}"
        )

    def build_training_completion(self, tool_call: Dict[str, Any]) -> str:
        """Format as <tool_call> XML tags matching Qwen 2.5's output."""
        return f" <tool_call>\n{json.dumps(tool_call)}\n</tool_call>"

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "qwen",
            "size_tier": "medium",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }

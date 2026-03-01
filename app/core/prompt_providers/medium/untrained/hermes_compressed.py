"""
HermesCompressed - Compressed prompt provider for Hermes 3 8B.

Applies the same compression technique proven in Qwen25Compressed:
replaces the verbose Tools: section (full param details + examples) with
a compact name + first-sentence listing. The <tools> JSON block is passed
through unchanged — param descriptions in JSON carry critical schema info.

Uses Hermes's own inline rules (not shared core_rules.py — that caused
93% → 58% regression). The only changes vs HermesMediumUntrained are:
  1. Compact Tools: listing instead of verbose format_tools_for_prompt
  2. DT_KEYS injection for date key vocabulary

Inherits parse_response, build_tools, _build_tools_xml, build_training_*,
get_response_format, supports_native_tools, and use_tool_classifier from parent.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from app.core.prompt_providers.medium.untrained.hermes_medium_untrained import (
    HermesMediumUntrained,
)
from app.core.prompt_providers.shared.context_builders import (
    build_agent_context_summary,
    build_direct_answer_section,
)

logger = logging.getLogger("uvicorn")


def _first_sentence(text: str) -> str:
    """Extract first sentence from text, keeping the period."""
    if not text:
        return ""
    sentence: str = text.split(".")[0].strip()
    return f"{sentence}." if sentence else ""


class HermesCompressed(HermesMediumUntrained):
    """
    Compressed prompt provider for Hermes 3 Llama 3.1 8B Instruct.

    Replaces the verbose Tools: section with a compact name + first-sentence
    listing. Hermes's inline rules are preserved (shared core_rules.py
    degrades Hermes accuracy).
    """

    @property
    def name(self) -> str:
        return "HermesCompressed"

    @staticmethod
    def _build_compact_tools_section(tools: List[Dict[str, Any]]) -> str:
        """Build compact Tools: listing with name + first-sentence description."""
        if not tools:
            return "No tools available."

        lines: list[str] = []
        for tool in tools:
            func: Dict[str, Any] = tool.get("function", {})
            name: str = func.get("name", "unknown")
            desc: str = _first_sentence(func.get("description", ""))
            lines.append(f"- {name}: {desc}")

        return "\n".join(lines)

    def build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        available_commands = available_commands or []
        node_context = node_context or {}

        room: str = node_context.get("room", "unknown")
        user: str = node_context.get("speaker_name") or node_context.get("user", "default")
        voice_mode: str = node_context.get("voice_mode", "brief")
        user_memories: str = node_context.get("user_memories", "")
        date_keys: list[str] = node_context.get("date_keys", [])

        # Shared sections
        direct_answer_section: str = build_direct_answer_section(available_commands)
        agent_context_section: str = build_agent_context_summary(node_context)

        # Hermes-specific: <tools> XML block
        tools_xml: str = HermesMediumUntrained._build_tools_xml(tools)

        # Compact tools summary: name + first-sentence description, no params
        compact_tools: str = self._build_compact_tools_section(tools)

        # Build memory block
        memory_block: str = ""
        if user_memories:
            memory_block = f"\nAbout {user}:\n{user_memories}\n"

        # Build DT_KEYS line from llm-proxy date keys
        dt_keys_line: str = ""
        if date_keys:
            dt_keys_line = (
                f"\nDT_KEYS: {'|'.join(date_keys)}\n"
                "Date params: ALWAYS include resolved_datetimes — use DT_KEYS only, NEVER ISO timestamps. "
                "If the user omits a date, you MUST still pass [\"today\"].\n"
            )

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
{{"name": "<function-name>", "arguments": {{"<arg-name>": "<arg-value>"}}}}
</tool_call>
{dt_keys_line}
{direct_answer_section}
{agent_context_section}
For final answers with no tool needed, respond with a brief spoken reply.

Tools:
{compact_tools}
"""

        logger.info(
            "Built HermesCompressed system prompt: %d chars, %d tools",
            len(system_prompt),
            len(tools),
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("System prompt (full):\n%s", system_prompt)

        return system_prompt

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "hermes",
            "size_tier": "medium",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }

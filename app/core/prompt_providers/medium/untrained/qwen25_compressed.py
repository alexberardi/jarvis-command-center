"""
Qwen25Compressed - Compressed prompt provider for Qwen 2.5 7B.

Compresses the system prompt by replacing the verbose Tools: section
(which lists every tool with full param details) with a compact
name:description listing. The <tools> JSON block is passed through
unchanged — param descriptions in JSON carry critical schema info.

The standard Qwen function-calling preamble is preserved exactly — Qwen 2.5
needs it verbatim to activate its fine-tuned tool-calling path.

Inherits parse_response, build_tools, and all other behavior from the parent.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from app.core.prompt_providers.medium.untrained.qwen25_medium_untrained import (
    Qwen25MediumUntrained,
)
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

logger = logging.getLogger("uvicorn")


def _first_sentence(text: str) -> str:
    """Extract first sentence from text, keeping the period."""
    if not text:
        return ""
    sentence: str = text.split(".")[0].strip()
    return f"{sentence}." if sentence else ""


class Qwen25Compressed(Qwen25MediumUntrained):
    """
    Compressed prompt provider for Qwen 2.5 7B Instruct.

    Replaces the verbose Tools: section with a compact name + first-sentence
    listing. The <tools> JSON block is passed through unchanged from the parent.
    """

    @property
    def name(self) -> str:
        return "Qwen25Compressed"

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

        # Reuse shared builders (identical to parent)
        identity: str = build_identity_header(room, user, voice_mode, user_memories)
        rules: str = build_rules_block()
        direct_answer_section: str = build_direct_answer_section(available_commands)
        agent_context_section: str = build_agent_context_summary(node_context)
        fallback: str = build_fallback_line()

        # Standard <tools> block from parent — no compression, param descriptions intact
        tools_block: str = Qwen25MediumUntrained._build_tools_block(tools)

        # Compact tools summary: name + first-sentence description, no params
        compact_tools: str = self._build_compact_tools_section(tools)

        # Build DT_KEYS line from llm-proxy date keys
        dt_keys_line: str = ""
        if date_keys:
            dt_keys_line = (
                f"\nDT_KEYS: {'|'.join(date_keys)}\n"
                "Date params: ALWAYS include resolved_datetimes — use DT_KEYS only, NEVER ISO timestamps. "
                "If the user omits a date, you MUST still pass [\"today\"].\n"
            )

        system_prompt: str = f"""{identity}

You are a function calling AI model. You may call one or more functions to assist with the user query. Always include all required parameters — use sensible defaults from context when the user does not state them explicitly. {ANTI_HALLUCINATION_MANDATE}

You are provided with function signatures within <tools></tools> XML tags:
{tools_block}

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": {{"<arg-name>": "<arg-value>"}}}}
</tool_call>

{rules}
{dt_keys_line}
{direct_answer_section}
{agent_context_section}
{fallback}

Tools:
{compact_tools}
"""

        logger.info(
            "Built Qwen25Compressed system prompt: %d chars, %d tools",
            len(system_prompt),
            len(tools),
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("System prompt (full):\n%s", system_prompt)

        return system_prompt

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "qwen",
            "size_tier": "medium",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }

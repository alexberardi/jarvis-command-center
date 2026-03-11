"""
Qwen25LargeUntrained - Prompt provider for Qwen 2.5 14B Instruct.

Based on Qwen25Compressed (the best-performing Qwen provider at 95.1% on 7B).
Uses the same compressed prompt strategy with compact tool listings and
standard Qwen function-calling preamble.

The 14B model has more capacity, so the compressed prompt avoids wasting
context window while the larger model handles harder tasks (compound time
math, entity resolution, multi-step reasoning) more reliably.

14B-specific tuning vs 7B:
- "Respond in user's spoken language" rule — the multilingual 14B model
  occasionally code-switches to Russian/Kazakh on ambiguous queries.
- Stronger DT_KEYS enforcement — the 14B model is smart enough to resolve
  dates itself, but the downstream system handles date resolution, so we
  need it to pass tokens instead of ISO timestamps.
- Best-effort tool call preference — larger models are more likely to ask
  for clarification instead of making a tool call.

Inherits parse_response, build_tools, and tool-calling format from
Qwen25MediumUntrained via Qwen25Compressed.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from app.core.prompt_providers.medium.untrained.qwen25_compressed import (
    Qwen25Compressed,
)
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


class Qwen25LargeUntrained(Qwen25Compressed):
    """
    Prompt provider for Qwen 2.5 14B Instruct (untrained).

    Same ChatML format and <tool_call> tags as the 7B variant, with
    additional rules to tame the 14B model's multilingual and
    date-resolution tendencies.
    """

    @property
    def name(self) -> str:
        return "Qwen25LargeUntrained"

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

        # Shared builders — with extra rules for the 14B model
        identity: str = build_identity_header(room, user, voice_mode)
        rules: str = build_rules_block(extra_rules=[
            "ALWAYS respond in the language of the voice command.",
            "Only ask for clarification when absolutely necessary — prefer making a best-effort tool call.",
            "NEVER return ISO dates or timestamps in resolved_datetimes — use ONLY date key strings like \"today\", \"tomorrow\", \"yesterday\", \"this_weekend\", \"next_week\", \"this_year\".",
        ])
        direct_answer_section: str = build_direct_answer_section(available_commands)
        agent_context_section: str = build_agent_context_summary(node_context)
        fallback: str = build_fallback_line()

        # Standard <tools> block — param descriptions intact
        tools_block: str = Qwen25MediumUntrained._build_tools_block(tools)

        # Compact tools summary: name + first-sentence description
        compact_tools: str = self._build_compact_tools_section(tools)

        # Stronger DT_KEYS enforcement for 14B (model tries to resolve dates itself)
        dt_keys_line: str = ""
        if date_keys:
            dt_keys_line = (
                f"\nDT_KEYS: {'|'.join(date_keys)}\n"
                "CRITICAL: resolved_datetimes MUST contain ONLY DT_KEY tokens from the list above. "
                "NEVER compute, resolve, or output ISO timestamps (e.g. 2026-02-28T05:00:00Z). "
                "The downstream system handles all date resolution — just pass the matching key.\n"
            )

        system_prompt: str = f"""{identity}

You are a function calling AI model. You may call one or more functions to assist with the user query. Always include all required parameters — use sensible defaults from context when the user does not state them explicitly. {ANTI_HALLUCINATION_MANDATE}

You are provided with function signatures within <tools></tools> XML tags:
{tools_block}

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": {{"<arg-name>": "<arg-value>"}}, "failure_message": "<brief spoken response if this call fails>"}}
</tool_call>

IMPORTANT: resolved_datetimes accepts ONLY literal date keys: "today", "tomorrow", "yesterday", "this_weekend", "last_weekend", "next_week", "this_year", "day_after_tomorrow". ISO timestamps (e.g. 2026-02-28T00:00:00Z) are INVALID and will cause a parsing error.

{rules}
{dt_keys_line}
{direct_answer_section}
{agent_context_section}
{fallback}

Tools:
{compact_tools}
"""

        logger.info(
            "Built Qwen25LargeUntrained system prompt: %d chars, %d tools",
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
            "size_tier": "large",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }

"""
MixtralLargeUntrained - Prompt provider for Mixtral 8x7B Instruct v0.1.

Large MoE model (~47B params). Compressed prompt — the model's capacity
means less instruction is needed. Follows the same lean pattern as the
Qwen 7B compressed provider which achieves 95%+ accuracy.

Inherits parse_response (handles both <tool_call> XML tags and Mistral's
native [TOOL_CALLS] format), build_tools, get_response_format, and
build_training_completion from Mistral7bMediumUntrained.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from app.core.prompt_providers.medium.untrained.mistral7b_medium_untrained import (
    Mistral7bMediumUntrained,
)
from app.core.prompt_providers.shared.context_builders import (
    build_agent_context_summary,
    build_direct_answer_section,
)
from app.core.prompt_providers.shared.core_rules import (
    ANTI_HALLUCINATION_MANDATE,
    RULE_BEST_MATCH_INTENT,
    RULE_EXTRACT_PARAMS,
    RULE_ONE_AT_A_TIME,
    RULE_POPULATE_REQUIRED,
    RULE_STT_AWARENESS,
    build_identity_header,
)
from app.core.tool_builder import ToolBuilder

logger = logging.getLogger("uvicorn")


class MixtralLargeUntrained(Mistral7bMediumUntrained):
    """
    Prompt provider for Mixtral 8x7B Instruct v0.1 (untrained).

    Compressed prompt — stripped param descriptions, one-per-line tool JSON,
    minimal rules. The larger model follows instructions with less guidance.
    """

    @property
    def name(self) -> str:
        return "MixtralLargeUntrained"

    @property
    def force_tool_calls(self) -> bool:
        """Re-prompt when the model returns a direct answer instead of a tool call."""
        return True

    @staticmethod
    def _build_compressed_tools_block(tools: List[Dict[str, Any]]) -> str:
        """Compact <tools> block — one tool per line, with param descriptions.

        Keeps param descriptions (Mixtral needs them for extraction) but strips
        format hints and excludes refinable params. Compact JSON separators.
        """
        clean_tools: List[Dict[str, Any]] = ToolBuilder.build(
            tools,
            include_param_descriptions=True,
            include_format_hints=False,
            exclude_refinable=True,
        )
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
        available_commands = available_commands or []
        node_context = node_context or {}

        room: str = node_context.get("room", "unknown")
        user: str = node_context.get("speaker_name") or node_context.get("user", "default")
        voice_mode: str = node_context.get("voice_mode", "brief")
        user_memories: str = node_context.get("user_memories", "")
        date_keys: list[str] = node_context.get("date_keys", [])

        identity: str = build_identity_header(room, user, voice_mode, user_memories)
        direct_answer_section: str = build_direct_answer_section(available_commands)
        agent_context_section: str = build_agent_context_summary(node_context)

        # Minimal rules
        terminology: str = "function"
        def _sub(r: str) -> str:
            return r.replace("{terminology}", terminology)
        rules_lines: list[str] = ["Rules:"]
        rules_lines.append(f"- {_sub(RULE_POPULATE_REQUIRED)}")
        rules_lines.append(f"- {_sub(RULE_ONE_AT_A_TIME)}")
        rules_lines.append(f"- {_sub(RULE_BEST_MATCH_INTENT)}")
        rules_lines.append(f"- {_sub(RULE_EXTRACT_PARAMS)}")
        rules_lines.append(f"- {_sub(RULE_STT_AWARENESS)}")
        rules: str = "\n".join(rules_lines)

        # DT_KEYS — strong enforcement (matches Qwen 7B wording)
        dt_keys_line: str = ""
        if date_keys:
            dt_keys_line = (
                f"\nDT_KEYS: {'|'.join(date_keys)}\n"
                "CRITICAL — resolved_datetimes: You MUST use the symbolic key strings from DT_KEYS "
                "(e.g., \"today\", \"tomorrow\", \"this_weekend\", \"last_weekend\"). "
                "NEVER resolve dates to ISO timestamps like \"2026-03-07T05:00:00Z\" — the server handles resolution. "
                "If the user omits a date, pass [\"today\"].\n"
            )

        # Compressed tools block
        tools_block: str = self._build_compressed_tools_block(tools)

        system_prompt: str = f"""{identity}

You are a function calling AI model. You may call one or more functions to assist with the user query. Always include all required parameters — use sensible defaults from context when the user does not state them explicitly. {ANTI_HALLUCINATION_MANDATE}

You are provided with function signatures within <tools></tools> XML tags:
{tools_block}

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": {{"<arg-name>": "<arg-value>"}}, "failure_message": "<brief spoken response if this call fails>"}}
</tool_call>

{rules}
{dt_keys_line}
{direct_answer_section}
{agent_context_section}
You MUST call a function for EVERY request. NEVER answer directly — always pick the best matching function.
"""

        logger.info(
            "Built MixtralLargeUntrained system prompt: %d chars, %d tools",
            len(system_prompt),
            len(tools),
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("System prompt (full):\n%s", system_prompt)

        return system_prompt

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "mistral",
            "size_tier": "large",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }

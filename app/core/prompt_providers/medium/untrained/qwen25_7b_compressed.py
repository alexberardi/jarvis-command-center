"""
Qwen25_7B_Compressed - Token-optimized prompt provider for Qwen 2.5 7B Instruct.

Leaner than the parent Qwen25MediumUntrained — strips param descriptions,
drops redundant rules, uses Stage 2 refinement for complex enums.
Compared to the 3B Qwen25Compressed, includes Direct Answer section
and drops the aggressive tool mandate (7B can follow nuanced guidance).

Inherits parse_response, build_tools, and all other behavior from the parent.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from app.core.prompt_providers.medium.untrained.qwen25_medium_untrained import (
    Qwen25MediumUntrained,
)
from app.core.tool_builder import ToolBuilder
from app.core.prompt_providers.shared.context_builders import (
    build_agent_context_summary,
    build_direct_answer_section,
)
from app.core.prompt_providers.shared.core_rules import (
    ANTI_HALLUCINATION_MANDATE,
    RULE_BEST_MATCH_INTENT,
    RULE_ONE_AT_A_TIME,
    RULE_POPULATE_REQUIRED,
    RULE_STT_AWARENESS,
    build_identity_header,
)

logger = logging.getLogger("uvicorn")


class Qwen25_7B_Compressed(Qwen25MediumUntrained):
    """
    Token-optimized prompt provider for Qwen 2.5 7B Instruct.

    Leaner than the parent Qwen25MediumUntrained — strips param descriptions,
    drops redundant rules, uses Stage 2 refinement for complex enums.
    Compared to the 3B Qwen25Compressed, includes Direct Answer section
    and drops the aggressive tool mandate (7B can follow nuanced guidance).
    """

    @property
    def name(self) -> str:
        return "Qwen25_7B_Compressed"

    @property
    def force_tool_calls(self) -> bool:
        """Signal that every request must produce a tool call.

        The tool execution engine checks this flag and retries when
        the model returns finish_reason='stop' without calling a tool.
        """
        return True

    @staticmethod
    def _build_compressed_tools_block(tools: List[Dict[str, Any]]) -> str:
        """Build <tools> block with param descriptions and format hints stripped.

        Same Qwen-style one-per-line JSON format, but without "description"
        on individual parameter properties and without "format":"date-time"
        keys that conflict with datekey instructions. Refinable params are
        excluded (resolved in Stage 2). Full tool descriptions are preserved
        (they carry routing hints).
        """
        clean_tools: List[Dict[str, Any]] = ToolBuilder.build(
            tools,
            include_param_descriptions=False,
            include_format_hints=False,
            exclude_refinable=True,
        )
        if not clean_tools:
            return "<tools>\n</tools>"
        lines: List[str] = [json.dumps(t, separators=(",", ":")) for t in clean_tools]
        return "<tools>\n" + "\n".join(lines) + "\n</tools>"

    @staticmethod
    def _build_rules_block() -> str:
        """Build minimal Rules: block for 7B.

        Drops RULE_USE_ACTUAL_PARAM_NAMES, RULE_EXTRACT_PARAMS, and
        RULE_DATE_PARAMS — 7B naturally uses schema param names, extracts
        params from context, and the DT_KEYS section covers date handling
        more specifically.
        """
        terminology: str = "function"

        def _sub(rule: str) -> str:
            return rule.replace("{terminology}", terminology)

        rules: list[str] = [
            _sub(RULE_POPULATE_REQUIRED),
            _sub(RULE_ONE_AT_A_TIME),
            _sub(RULE_BEST_MATCH_INTENT),
            _sub(RULE_STT_AWARENESS),
        ]

        lines: list[str] = ["Rules:"]
        for rule in rules:
            lines.append(f"- {rule}")
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

        # Shared builders
        identity: str = build_identity_header(room, user, voice_mode, user_memories)
        rules: str = self._build_rules_block()
        direct_answer_section: str = build_direct_answer_section(available_commands)
        agent_context_section: str = build_agent_context_summary(node_context)

        # Compressed <tools> block — last for KV cache optimization
        tools_block: str = self._build_compressed_tools_block(tools)

        # Build DT_KEYS line from llm-proxy date keys
        dt_keys_line: str = ""
        if date_keys:
            dt_keys_line = (
                f"\nDT_KEYS: {'|'.join(date_keys)}\n"
                "CRITICAL — resolved_datetimes: You MUST use the symbolic key strings from DT_KEYS "
                "(e.g., \"today\", \"tomorrow\", \"this_weekend\", \"last_weekend\"). "
                "NEVER resolve dates to ISO timestamps like \"2026-03-07T05:00:00Z\" — the server handles resolution. "
                "If the user omits a date, pass [\"today\"].\n"
            )

        # Stable content first, variable <tools> block last for KV cache
        system_prompt: str = f"""{identity}

You are a function calling AI model. You may call one or more functions to assist with the user query. Always include all required parameters — use sensible defaults from context when the user does not state them explicitly. {ANTI_HALLUCINATION_MANDATE}

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": {{"<arg-name>": "<arg-value>"}}}}
</tool_call>

{rules}
{dt_keys_line}
{direct_answer_section}
{agent_context_section}
You are provided with function signatures within <tools></tools> XML tags:
{tools_block}
"""

        logger.info(
            "Built Qwen25_7B_Compressed system prompt: %d chars, %d tools",
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

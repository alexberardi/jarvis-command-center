"""
Qwen25Compressed - Token-optimized prompt provider for Qwen 2.5 3B/7B.

DEPRECATED for 7B use — use Qwen25_7B_Compressed instead (fewer rules,
Direct Answer section, KV-cache-optimized layout). This class is kept as the
base for Qwen25_3B_Compressed.

Minimizes system prompt tokens to reduce prefill latency while maintaining
command parsing accuracy. The <tools> JSON block carries all schema info;
redundant sections are removed or compressed.

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
)
from app.core.prompt_providers.shared.core_rules import (
    ANTI_HALLUCINATION_MANDATE,
    RULE_BEST_MATCH_INTENT,
    RULE_EXTRACT_PARAMS,
    RULE_ONE_AT_A_TIME,
    RULE_POPULATE_REQUIRED,
    RULE_STT_AWARENESS,
    RULE_USE_ACTUAL_PARAM_NAMES,
)

logger = logging.getLogger("uvicorn")


class Qwen25Compressed(Qwen25MediumUntrained):
    """
    Token-optimized prompt provider for Qwen 2.5 3B/7B Instruct.

    The <tools> JSON block carries all tool names, descriptions, and parameter
    schemas. No redundant Tools: listing or antipatterns section is needed —
    routing hints are embedded directly in tool descriptions (positive framing).
    """

    @property
    def name(self) -> str:
        return "Qwen25Compressed"

    @staticmethod
    def _build_compressed_tools_block(tools: List[Dict[str, Any]]) -> str:
        """Build <tools> block with param descriptions and format hints stripped.

        Same Qwen-style one-per-line JSON format, but without "description"
        on individual parameter properties and without "format":"date-time"
        keys that conflict with datekey instructions. Full tool descriptions
        are preserved (they carry routing hints).
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
    def _build_rules_block_no_date_params() -> str:
        """Build Rules: block excluding RULE_DATE_PARAMS.

        The DT_KEYS section already covers date parameter instructions with
        more keys and the critical 'must pass today' instruction, so
        RULE_DATE_PARAMS is redundant here.
        """
        terminology: str = "function"

        def _sub(rule: str) -> str:
            return rule.replace("{terminology}", terminology)

        rules: list[str] = [
            _sub(RULE_POPULATE_REQUIRED),
            _sub(RULE_ONE_AT_A_TIME),
            _sub(RULE_USE_ACTUAL_PARAM_NAMES),
            _sub(RULE_BEST_MATCH_INTENT),
            _sub(RULE_EXTRACT_PARAMS),
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

        date_keys: list[str] = node_context.get("date_keys", [])

        # Shared builders
        identity: str = self.build_context_header(node_context)
        rules: str = self._build_rules_block_no_date_params()
        agent_context_section: str = build_agent_context_summary(node_context)

        # Strict tool mandate — the 3B model ignores nuanced "only X can be direct"
        # instructions, so we enforce unconditionally and let the server-side
        # force_tool_calls guard handle retries.
        tool_mandate: str = (
            "You MUST call a tool for EVERY request. "
            "NEVER answer directly — always pick the best matching function."
        )

        # Compressed <tools> block — param descriptions + format hints stripped
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
{tool_mandate}
{agent_context_section}
"""

        logger.info(
            "Built Qwen25Compressed system prompt: %d chars, %d tools",
            len(system_prompt),
            len(tools),
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("System prompt (full):\n%s", system_prompt)

        return system_prompt

    @property
    def force_tool_calls(self) -> bool:
        """Signal that every request must produce a tool call.

        The tool execution engine checks this flag and retries when
        the model returns finish_reason='stop' without calling a tool.
        """
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "qwen",
            "size_tier": "medium",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }

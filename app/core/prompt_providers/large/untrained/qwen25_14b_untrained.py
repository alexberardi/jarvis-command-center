"""
Qwen25_14B_Untrained - Prompt provider for Qwen 2.5 14B Instruct (safetensors).

Based on the Qwen25_7B_Compressed pattern (95%+ accuracy) — compressed tools
block, minimal rules, DT_KEYS enforcement, force_tool_calls.

14B-specific tuning vs 7B compressed:
- RULE_EXTRACT_PARAMS added back — 14B is more eager to ask clarifying
  questions instead of making best-effort tool calls.
- Language rule — the multilingual 14B model occasionally code-switches
  to non-English on ambiguous queries.
- User memories in identity header — 14B has enough capacity to use them.
- Full tool descriptions preserved (compressed block already does this).

Inherits parse_response, build_tools, get_response_format, and
build_training_completion from Qwen25MediumUntrained via Qwen25_7B_Compressed.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from app.core.prompt_providers.medium.untrained.qwen25_7b_compressed import (
    Qwen25_7B_Compressed,
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

logger = logging.getLogger("uvicorn")


class Qwen25_14B_Untrained(Qwen25_7B_Compressed):
    """
    Prompt provider for Qwen 2.5 14B Instruct (untrained, safetensors).

    Compressed prompt matching the 7B Compressed pattern with additional
    rules to tame the 14B model's tendencies to over-clarify and
    code-switch languages.
    """

    @property
    def name(self) -> str:
        return "Qwen25_14B_Untrained"

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

        # 6 rules — 7B compressed base (4) + extract_params + language rule
        terminology: str = "function"

        def _sub(r: str) -> str:
            return r.replace("{terminology}", terminology)

        rules_lines: list[str] = ["Rules:"]
        rules_lines.append(f"- {_sub(RULE_POPULATE_REQUIRED)}")
        rules_lines.append(f"- {_sub(RULE_ONE_AT_A_TIME)}")
        rules_lines.append(f"- {_sub(RULE_BEST_MATCH_INTENT)}")
        rules_lines.append(f"- {_sub(RULE_EXTRACT_PARAMS)}")
        rules_lines.append(f"- {_sub(RULE_STT_AWARENESS)}")
        rules_lines.append("- ALWAYS respond in the language the user spoke. Prefer making a best-effort tool call over asking for clarification.")
        rules: str = "\n".join(rules_lines)

        # DT_KEYS — strong enforcement for 14B (model tries to resolve dates itself)
        dt_keys_line: str = ""
        if date_keys:
            dt_keys_line = (
                f"\nDT_KEYS: {'|'.join(date_keys)}\n"
                "CRITICAL — resolved_datetimes: You MUST use the symbolic key strings from DT_KEYS "
                "(e.g., \"today\", \"tomorrow\", \"this_weekend\", \"last_weekend\"). "
                "NEVER resolve dates to ISO timestamps like \"2026-03-07T05:00:00Z\" — the server handles resolution. "
                "If the user omits a date, pass [\"today\"].\n"
            )

        # Compressed tools block (param descriptions stripped, format hints stripped)
        tools_block: str = self._build_compressed_tools_block(tools)

        system_prompt: str = f"""{identity}

You are a function calling AI model. You may call one or more functions to assist with the user query. Always include all required parameters — use sensible defaults from context when the user does not state them explicitly. {ANTI_HALLUCINATION_MANDATE}

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": {{"<arg-name>": "<arg-value>"}}, "failure_message": "<brief spoken response if this call fails>"}}
</tool_call>

{rules}
{dt_keys_line}
{direct_answer_section}
{agent_context_section}
You are provided with function signatures within <tools></tools> XML tags:
{tools_block}
"""

        logger.info(
            "Built Qwen25_14B_Untrained system prompt: %d chars, %d tools",
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

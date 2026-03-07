"""
Qwen3LargeUntrained - Prompt provider for Qwen3-30B-A3B (MoE).

30B total params, 3.3B active per token. Uses the same ChatML format and
<tool_call> tags as Qwen 2.5 but strips Qwen3's <think> blocks to avoid
latency from chain-of-thought reasoning on simple voice commands.

Based on Qwen25_7B_Compressed (95%+ accuracy) — compressed tools block,
5 rules, CRITICAL DT_KEYS, force_tool_calls.

Inherits parse_response, build_tools, get_response_format, and
build_training_completion from Qwen25MediumUntrained via Qwen25_7B_Compressed.
"""

import logging
import os
import re
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

# Strip <think>...</think> blocks (Qwen3 thinking mode output)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class Qwen3LargeUntrained(Qwen25_7B_Compressed):
    """
    Prompt provider for Qwen3-30B-A3B (MoE, untrained).

    Compressed prompt matching the Qwen 7B Compressed pattern. Strips
    <think> blocks from Qwen3's output and injects /nothink in user
    messages to disable thinking mode.
    """

    @property
    def name(self) -> str:
        return "Qwen3LargeUntrained"

    @property
    def user_message_suffix(self) -> str:
        """Append /nothink to user messages to disable Qwen3 thinking mode.

        Qwen3 ignores /no_think in the system prompt — the model was
        trained to recognize this token in user turns only.
        """
        return "/nothink"

    def parse_response(self, raw_content: str) -> Optional[str]:
        """Strip <think> blocks, then delegate to Qwen 2.5 parser."""
        cleaned: str = _THINK_BLOCK_RE.sub("", raw_content)
        return super().parse_response(cleaned)

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

        # 5 rules (same as Mixtral large — proven at 94.9%)
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

        # DT_KEYS — strong enforcement
        dt_keys_line: str = ""
        if date_keys:
            dt_keys_line = (
                f"\nDT_KEYS: {'|'.join(date_keys)}\n"
                "CRITICAL — resolved_datetimes: You MUST use the symbolic key strings from DT_KEYS "
                "(e.g., \"today\", \"tomorrow\", \"this_weekend\", \"last_weekend\"). "
                "NEVER resolve dates to ISO timestamps like \"2026-03-07T05:00:00Z\" — the server handles resolution. "
                "If the user omits a date, pass [\"today\"].\n"
            )

        # Compressed tools block (with param descriptions, no format hints)
        tools_block: str = self._build_compressed_tools_block(tools)

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
"""

        logger.info(
            "Built Qwen3LargeUntrained system prompt: %d chars, %d tools",
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

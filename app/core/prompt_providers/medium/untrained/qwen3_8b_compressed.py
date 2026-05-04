"""
Qwen3_8B_Compressed - Prompt provider for Qwen3-8B Instruct (Q4_K_M GGUF).

8B dense model using ChatML format and <tool_call> tags like Qwen 2.5.
Strips Qwen3's <think> blocks and injects /nothink in user messages to
disable chain-of-thought reasoning on simple voice commands.

Based on Qwen25_7B_Compressed (95%+ accuracy) — compressed tools block,
4 rules, CRITICAL DT_KEYS, force_tool_calls. Adds Qwen3-specific thinking
mode handling from Qwen3LargeUntrained.

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
    build_tool_guidance_section,
)
from app.core.prompt_providers.shared.core_rules import (
    ANTI_HALLUCINATION_MANDATE,
)

logger = logging.getLogger("uvicorn")

# Strip <think>...</think> blocks (Qwen3 thinking mode output)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# Strip unclosed <think> blocks (truncated by max_tokens)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL)


class Qwen3_8B_Compressed(Qwen25_7B_Compressed):
    """
    Prompt provider for Qwen3-8B Instruct (Q4_K_M, untrained).

    Same compressed prompt as Qwen 2.5 7B Compressed (4 rules, stripped
    param descriptions, Direct Answer section). Adds /nothink suffix
    and <think> block stripping for Qwen3's thinking mode.
    """

    @property
    def name(self) -> str:
        return "Qwen3_8B_Compressed"

    @property
    def user_message_suffix(self) -> str:
        """Return /think or /no_think based on the advanced_thinking setting.

        Qwen3's documented control tokens are /think and /no_think
        (underscore-separated). /nothink (without the underscore) is
        unrecognized and silently ignored — the model emits full thinking
        blocks, adding ~500 tokens and ~8-10s of decode per voice response.

        /no_think must land in a USER turn; /system prompt placement is
        ignored by Qwen3's training.

        When model.advanced_thinking is enabled, /think allows chain-of-thought
        reasoning (~2s extra latency, better for complex queries).
        """
        return "/think" if self.advanced_thinking else "/no_think"

    def parse_response(self, raw_content: str) -> Optional[str]:
        """Strip <think> blocks, then delegate to Qwen 2.5 parser."""
        cleaned: str = _THINK_BLOCK_RE.sub("", raw_content)
        cleaned = _THINK_UNCLOSED_RE.sub("", cleaned)
        return super().parse_response(cleaned)

    def sanitize_text(self, text: str) -> str:
        """Strip Qwen3 think blocks from user-facing text.

        See Qwen3LargeUntrained.sanitize_text for the full rationale —
        same defense-in-depth for wake/chat/direct-answer paths.
        """
        cleaned = _THINK_BLOCK_RE.sub("", text)
        cleaned = _THINK_UNCLOSED_RE.sub("", cleaned)
        return cleaned.strip()

    def build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Same as Qwen25_7B_Compressed but injects a Tool Guidance section
        aggregated from per-tool ``included_system_prompt_text`` hints."""
        available_commands = available_commands or []
        node_context = node_context or {}

        date_keys: list[str] = node_context.get("date_keys", [])

        identity: str = self.build_context_header(node_context)
        rules: str = self._build_rules_block()
        direct_answer_section: str = build_direct_answer_section(available_commands)
        agent_context_section: str = build_agent_context_summary(node_context)
        tool_guidance_section: str = build_tool_guidance_section(tools)
        tools_block: str = self._build_compressed_tools_block(tools)

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

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": {{"<arg-name>": "<arg-value>"}}, "failure_message": "<brief spoken response if this call fails>"}}
</tool_call>

{rules}
{dt_keys_line}{tool_guidance_section}{direct_answer_section}
{agent_context_section}
You are provided with function signatures within <tools></tools> XML tags:
{tools_block}
"""

        logger.info(
            "Built Qwen3_8B_Compressed system prompt: %d chars, %d tools",
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

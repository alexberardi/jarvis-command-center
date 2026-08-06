"""
Qwen3_5_9B_Compressed - Prompt provider for Qwen3.5-9B (Q4_K_M GGUF, MTP).

Qwen3.5-9B is the same ChatML + ``<think>`` reasoning family as Qwen3-8B, so it
reuses ``Qwen3_8B_Compressed``'s ``/think`` vs ``/no_think`` control (via
``user_message_suffix``), ``<think>``-block stripping (``parse_response`` /
``sanitize_text``), and ``build_tools`` (OpenAI-format via ``ToolBuilder``).

TOOL CALLING — NATIVE (differs from the 8B parent). The 8B provider INSTRUCTS a
JSON-in-``<tool_call>`` text format and parses that. Qwen3.5's own chat_template
emits a DIFFERENT native tool syntax, so that instruction fought the model and its
calls were silently dropped to a direct answer (the exact risk the previous version
of this file flagged, and the live symptom: ``run_errand`` never firing). Fixed by
going fully native: ``supports_native_tools = True`` → the engine passes tools via the
API (``tools=``), llama.cpp injects them through the model's native template, and the
proxy returns structured OpenAI ``tool_calls`` (verified live: a get_weather prompt
came back ``finish_reason=tool_calls`` with a parsed function call). Because the tools
now reach the model natively, ``build_system_prompt`` OMITS the ``<tools>`` block and
the "return JSON in ``<tool_call>``" instruction — re-declaring a text format there
only re-creates the conflict. It keeps the routing GUIDANCE (identity, rules, DT_KEYS,
per-tool hints, Direct Answer, agent context).

Serving notes (llama.cpp side, independent of this prompt): arch ``qwen35``; 256K
native context (``qwen35.context_length = 262144``); 1 MTP ``nextn`` layer (speculative
decode only on an MTP-merged build, else a standard 9B).
"""

import logging
import os
from typing import Any, Dict, List, Optional

from app.core.prompt_providers.medium.untrained.qwen3_8b_compressed import (
    Qwen3_8B_Compressed,
)
from app.core.prompt_providers.shared.context_builders import (
    build_agent_context_summary,
    build_direct_answer_section,
    build_tool_guidance_section,
)
from app.core.prompt_providers.shared.core_rules import ANTI_HALLUCINATION_MANDATE

logger = logging.getLogger("uvicorn")


class Qwen3_5_9B_Compressed(Qwen3_8B_Compressed):
    """Qwen3.5-9B (MTP GGUF) — native tool calling; Qwen3 /think + parsing inherited."""

    @property
    def name(self) -> str:
        return "Qwen3_5_9B_Compressed"

    @property
    def supports_native_tools(self) -> bool:
        # Native path: pass tools via the API, get structured tool_calls back. Avoids
        # the JSON-in-<tool_call> instruction fighting Qwen3.5's native tool syntax
        # (which dropped tool calls to direct answers — the run_errand miss).
        return True

    def build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """NATIVE-tools system prompt: the 8B guidance MINUS the ``<tools>`` block and
        the JSON ``<tool_call>`` format instruction. The tools reach the model through
        the API's native tool mechanism, so re-declaring a text format here would only
        conflict with it. Keeps identity, rules, DT_KEYS, per-tool guidance, the Direct
        Answer section, and agent context."""
        available_commands = available_commands or []
        node_context = node_context or {}

        date_keys: list[str] = node_context.get("date_keys", [])

        identity: str = self.build_context_header(node_context)
        rules: str = self._build_rules_block()
        direct_answer_section: str = build_direct_answer_section(available_commands)
        agent_context_section: str = build_agent_context_summary(node_context)
        tool_guidance_section: str = build_tool_guidance_section(tools)

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

You are a function calling AI model. You may call one or more of the available functions to assist with the user query. Always include all required parameters — use sensible defaults from context when the user does not state them explicitly. {ANTI_HALLUCINATION_MANDATE}

{rules}
{dt_keys_line}{tool_guidance_section}{direct_answer_section}
{agent_context_section}
"""

        logger.info(
            "Built Qwen3_5_9B_Compressed NATIVE system prompt: %d chars, %d tools",
            len(system_prompt),
            len(tools),
        )
        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("System prompt (full):\n%s", system_prompt)
        return system_prompt

    def get_capabilities(self) -> Dict[str, Any]:
        caps = super().get_capabilities()
        caps["provider_name"] = self.name
        caps["model_family"] = "qwen3.5"
        return caps

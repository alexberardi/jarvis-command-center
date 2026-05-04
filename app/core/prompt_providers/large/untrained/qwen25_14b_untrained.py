"""
Qwen25_14B_Untrained - Prompt provider for Qwen 2.5 14B Instruct (safetensors).

Uses lazy tool loading: system prompt contains a compact capability list
and a single ``get_tools`` meta-tool instead of full tool schemas. The model
answers directly from injected context (weather, calendar, news) when
possible, and calls ``get_tools`` only when it needs to perform an action.

14B-specific tuning:
- RULE_EXTRACT_PARAMS — 14B is more eager to ask clarifying questions.
- Language rule — the multilingual 14B model occasionally code-switches.
- User memories in identity header — 14B has enough capacity to use them.
- force_tool_calls disabled — model answers from context when appropriate.
- lazy_tool_loading enabled — full tool schemas loaded on demand.

Inherits parse_response, build_tools, get_response_format, and
build_training_completion from Qwen25MediumUntrained via Qwen25_7B_Compressed.
"""

import json
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
)

logger = logging.getLogger("uvicorn")

# get_tools definition — the only tool in the <tools> block
_GET_TOOLS_DEFINITION: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_tools",
        "description": (
            "Retrieve full definitions of available tools with parameter details. "
            "Call this when you need to perform an action (set timers, control devices, "
            "search the web, check sports scores, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


class Qwen25_14B_Untrained(Qwen25_7B_Compressed):
    """
    Prompt provider for Qwen 2.5 14B Instruct (untrained, safetensors).

    Uses lazy tool loading to minimize prompt tokens. Full tool schemas
    are only provided when the model calls ``get_tools``.
    """

    @property
    def name(self) -> str:
        return "Qwen25_14B_Untrained"

    @property
    def force_tool_calls(self) -> bool:
        return False

    @property
    def lazy_tool_loading(self) -> bool:
        return True

    @staticmethod
    def _build_capability_summary(tools: List[Dict[str, Any]]) -> str:
        """Build a compact one-line-per-tool capability list.

        Extracts tool name + first sentence of description from each tool
        definition. This gives the model enough context to decide whether
        it needs tools without the full parameter schemas.
        """
        lines: list[str] = []
        for tool in tools:
            func = tool.get("function", {})
            if not isinstance(func, dict):
                continue
            name = func.get("name", "")
            desc = func.get("description", "")
            # Take first sentence only
            if ". " in desc:
                desc = desc[: desc.index(". ") + 1]
            elif desc and not desc.endswith("."):
                desc = desc + "."
            if name and name != "get_tools":
                lines.append(f"  {name}: {desc}")
        if not lines:
            return ""
        return "Available capabilities (call get_tools for full details):\n" + "\n".join(lines)

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

        identity: str = self.build_context_header(node_context)
        direct_answer_section: str = build_direct_answer_section(available_commands)
        agent_context_section: str = build_agent_context_summary(node_context)

        # 6 rules — same as before
        terminology: str = "function"

        def _sub(r: str) -> str:
            return r.replace("{terminology}", terminology)

        rules_lines: list[str] = ["Rules:"]
        rules_lines.append(f"- {_sub(RULE_POPULATE_REQUIRED)}")
        rules_lines.append(f"- {_sub(RULE_ONE_AT_A_TIME)}")
        rules_lines.append(f"- {_sub(RULE_BEST_MATCH_INTENT)}")
        rules_lines.append(f"- {_sub(RULE_EXTRACT_PARAMS)}")
        rules_lines.append(f"- {_sub(RULE_STT_AWARENESS)}")
        rules_lines.append(
            "- ALWAYS respond in the language the user spoke. "
            "Prefer making a best-effort tool call over asking for clarification."
        )
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

        # Only get_tools in the <tools> block — no capability summary
        get_tools_json: str = json.dumps(_GET_TOOLS_DEFINITION, separators=(",", ":"))
        tools_block: str = f"<tools>\n{get_tools_json}\n</tools>"

        system_prompt: str = f"""{identity}

You are a voice assistant that can answer questions and perform actions. {ANTI_HALLUCINATION_MANDATE}

When context (weather, calendar, news, etc.) is provided after the user's message, answer directly — do NOT call a tool for information you already have.
When you need to perform an action or access information not in context, you MUST call get_tools first — it is the ONLY tool available to you right now. After calling get_tools you will receive the full list of tools you can use. Do NOT call any other tool name until you have called get_tools.

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
            "Built Qwen25_14B_Untrained (lazy) system prompt: %d chars, %d tools available",
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
            "lazy_tool_loading": self.lazy_tool_loading,
        }

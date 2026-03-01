"""
Llama32_3B_Compressed - Minimal prompt provider for Llama 3.2 3B Instruct.

Text-based tool calling (supports_native_tools=False) with compact prompts.
The model outputs <function=name>{args}</function> tags naturally from its
Llama 3 fine-tuning, and parse_response() (inherited from Llama31MediumUntrained)
extracts them into structured tool calls.

System prompt contains only:
- Identity + context in one line
- Function calling format + examples
- Compact tool listing with param names: name(p1, p2): description
- Rules (call tools, one per response)
- DT_KEYS vocabulary for date params
- Direct answer policy
- Agent context (HA devices)
- Memory block (if available)

No JSON schemas — compact format keeps prompt under ~3K chars total.

Inherits parse_response, build_tools, build_training_completion,
get_response_format, and supports_native_tools from Llama31MediumUntrained.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from app.core.prompt_providers.medium.untrained.llama31_medium_untrained import (
    Llama31MediumUntrained,
)
from app.core.prompt_providers.shared.context_builders import (
    build_agent_context_summary,
    build_direct_answer_section,
)

logger = logging.getLogger("uvicorn")


class Llama32_3B_Compressed(Llama31MediumUntrained):
    """
    Minimal prompt provider for Meta Llama 3.2 3B Instruct.

    Keeps the system prompt as short as possible to avoid overwhelming
    the model's limited capacity. Reuses Llama 3.1's parse_response
    for <function=name>{args}</function> tag extraction.
    """

    @property
    def name(self) -> str:
        return "Llama32_3B_Compressed"

    @staticmethod
    def _build_compact_tools_with_params(tools: List[Dict[str, Any]]) -> str:
        """Build compact tool listing with param names, types, and enums."""
        lines: list[str] = []
        for tool in tools:
            func: Dict[str, Any] = tool.get("function", tool)
            name: str = func.get("name", "")
            desc: str = func.get("description", "")
            # First sentence only
            dot: int = desc.find(".")
            short_desc: str = desc[: dot + 1] if dot != -1 else desc
            # Param signatures with types and enums
            props: Dict[str, Any] = func.get("parameters", {}).get("properties", {})
            param_parts: list[str] = []
            for pname, pinfo in props.items():
                ptype: str = pinfo.get("type", "string")
                enum: list | None = pinfo.get("enum")
                if enum:
                    param_parts.append(f"{pname}:{'/'.join(str(e) for e in enum)}")
                else:
                    param_parts.append(f"{pname}:{ptype}")
            params_str: str = ", ".join(param_parts)
            lines.append(f"- {name}({params_str}): {short_desc}")
        return "\n".join(lines)

    def build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],  # noqa: ARG002
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

        # Compact tools with param names
        compact_tools: str = self._build_compact_tools_with_params(tools)

        # Direct answer policy
        direct_answer_section: str = build_direct_answer_section(available_commands)

        # HA summary (compact ~50 tokens)
        agent_context: str = build_agent_context_summary(node_context)

        # Memory block
        memory_block: str = ""
        if user_memories:
            memory_block = f"\nAbout {user}:\n{user_memories}\n"

        # DT_KEYS
        dt_keys_line: str = ""
        if date_keys:
            dt_keys_line = (
                f"\nFor resolved_datetimes, ONLY use these words: {', '.join(date_keys)}\n"
                "NEVER use ISO dates like 2026-03-01. Always pass as a list: [\"today\"], [\"tomorrow\"].\n"
                "If no date is mentioned, use [\"today\"].\n"
            )

        system_prompt: str = f"""You are Jarvis, a voice assistant. room={room}, user={user}, style={voice_mode}
{memory_block}
You MUST call functions using EXACTLY this format — no variations:
<function=FUNCTION_NAME>{{"param": "value"}}</function>

Examples:
<function=get_weather>{{"city": "Miami"}}</function>
<function=set_timer>{{"duration_seconds": 300}}</function>
<function=get_calendar_events>{{"resolved_datetimes": ["tomorrow"]}}</function>

Rules:
- ALWAYS call a tool when one matches the request. NEVER answer from memory when a tool can handle it.
- Call ONE tool per response.
- Read tool descriptions carefully to pick the best match.
{direct_answer_section}{dt_keys_line}{agent_context}
Available functions:
{compact_tools}
"""

        logger.info(
            "Built Llama32_3B_Compressed system prompt: %d chars, %d tools",
            len(system_prompt),
            len(tools),
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("System prompt (full):\n%s", system_prompt)

        return system_prompt

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "llama",
            "size_tier": "small",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }

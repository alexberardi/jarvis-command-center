"""
Gemma2Compressed - Compressed prompt provider for Gemma 2 9B Instruct.

Applies the same compression technique proven in HermesCompressed and
Qwen25Compressed: replaces the verbose Tools: section (full param details +
examples) with a compact name + full-description listing. The <tools> JSON
block is passed through unchanged — param descriptions in JSON carry critical
schema info.

Uses Gemma 2's own inline rules (not shared core_rules.py — that degraded
Hermes from 93% to 58%, so we follow the same inline approach). The only
changes vs Gemma2MediumUntrained are:
  1. Compact Tools: listing instead of verbose format_tools_for_prompt
  2. DT_KEYS injection for date key vocabulary
  3. Memory block for user-specific context

Inherits parse_response, build_tools, _build_tools_xml, build_training_completion,
get_response_format, supports_native_tools, and use_tool_classifier from parent.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from app.core.prompt_providers.medium.untrained.gemma2_medium_untrained import (
    Gemma2MediumUntrained,
)
from app.core.prompt_providers.shared.context_builders import (
    build_agent_context_summary,
    build_direct_answer_section,
)

logger = logging.getLogger("uvicorn")


class Gemma2Compressed(Gemma2MediumUntrained):
    """
    Compressed prompt provider for Google Gemma 2 9B Instruct.

    Replaces the verbose Tools: section with a compact name + full-description
    listing (no param details or examples). Gemma 2's inline rules are
    preserved (shared core_rules.py degrades Hermes accuracy, so we follow
    the same inline approach here).
    """

    @property
    def name(self) -> str:
        return "Gemma2Compressed"

    @staticmethod
    def _build_compact_tools_section(tools: List[Dict[str, Any]]) -> str:
        """Build compact Tools: listing with name + first-sentence description."""
        if not tools:
            return "No tools available."

        lines: list[str] = []
        for tool in tools:
            func: Dict[str, Any] = tool.get("function", {})
            name: str = func.get("name", "unknown")
            desc: str = func.get("description", "").strip()
            lines.append(f"- {name}: {desc}")

            # Render antipatterns as compact NOT lines
            for ap in tool.get("antipatterns", []):
                ap_cmd: str = ap.get("command_name", "")
                ap_desc: str = ap.get("description", "")
                if ap_cmd and ap_desc:
                    lines.append(f"  NOT {name} → use {ap_cmd}: {ap_desc}")

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

        # Shared sections
        direct_answer_section: str = build_direct_answer_section(available_commands)
        agent_context_section: str = build_agent_context_summary(node_context)

        # Gemma 2: <tools> XML block (inherited from parent)
        tools_xml: str = Gemma2MediumUntrained._build_tools_xml(tools)

        # Compact tools summary: name + first-sentence description, no params
        compact_tools: str = self._build_compact_tools_section(tools)

        # Build memory block
        memory_block: str = ""
        if user_memories:
            memory_block = f"\nAbout {user}:\n{user_memories}\n"

        # Build DT_KEYS line from llm-proxy date keys
        dt_keys_line: str = ""
        if date_keys:
            dt_keys_line = (
                f"\nDT_KEYS: {'|'.join(date_keys)}\n"
                "Date params: ALWAYS include resolved_datetimes — use DT_KEYS only, NEVER ISO timestamps. "
                "If the user omits a date, you MUST still pass [\"today\"].\n"
            )

        system_prompt: str = f"""You are Jarvis, a function calling voice assistant.
Context: room={room}, user={user}, style={voice_mode}
{memory_block}
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You MUST call a function for any request that matches an available tool. Do not make assumptions about what values to plug into functions.

{tools_xml}

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{{"name": "<function-name>", "arguments": {{"<arg-name>": "<arg-value>"}}, "failure_message": "<brief spoken response if this call fails>"}}
</tool_call>

Example — if the user says "What's the weather in Miami?", respond ONLY with:
<tool_call>
{{"name": "get_weather", "arguments": {{"city": "Miami", "resolved_datetimes": ["today"]}}}}
</tool_call>

Rules:
- You MUST call a tool for weather, sports, calendar, timers, music, device control, web search, and all other tool-covered domains. NEVER answer these from memory.
- Call ONE tool at a time to fulfill requests.
- Pick the tool that best matches intent; use get_command_utterance_examples if unsure.
- Extract parameters from the user's words; only request validation if required params are truly missing/ambiguous.
- For date parameters like resolved_datetimes, use natural words: "today", "tomorrow", "day_after_tomorrow", "this_weekend", "this_year". NEVER convert to ISO dates or timestamps.
- Always populate required tool parameters from the user's request.
{dt_keys_line}
{direct_answer_section}
{agent_context_section}
For final answers with no tool needed, respond with a brief spoken reply.

Tools:
{compact_tools}
"""

        logger.info(
            "Built Gemma2Compressed system prompt: %d chars, %d tools",
            len(system_prompt),
            len(tools),
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("System prompt (full):\n%s", system_prompt)

        return system_prompt

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "gemma",
            "size_tier": "medium",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }

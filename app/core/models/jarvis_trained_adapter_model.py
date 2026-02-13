"""
Jarvis Trained Adapter Model - Minimal prompt for LoRA-trained adapters.

The adapter already learned tool schemas, parameter extraction, and routing
during training. This prompt includes only runtime-variable context the
adapter couldn't have learned: room, user, HA device state, tool names,
and response format.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.models.jarvis_adapter_model import JarvisAdapterModel

logger = logging.getLogger("uvicorn")


class JarvisTrainedAdapterModel(JarvisAdapterModel):
    """
    Minimal-prompt model for LoRA adapters that were trained on
    full tool descriptions and examples.

    Inherits tool execution loop, classifier-disabled behavior,
    and HA context builder from JarvisAdapterModel. Only overrides
    name and _build_system_prompt to produce a ~400-500 token prompt.
    """

    @property
    def name(self) -> str:
        return "JarvisTrainedAdapterModel"

    def _build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build minimal system prompt for a trained adapter.

        Includes only runtime-variable context:
        1. Identity + context (room, user, style)
        2. HA device context (dynamic entity_ids)
        3. Response format (JSON structure)
        4. Available tool names (no descriptions/params)
        5. Direct answer policy
        """
        available_commands = available_commands or []
        node_context = node_context or {}

        room = node_context.get("room", "unknown")
        user = node_context.get("user", "default")
        voice_mode = node_context.get("voice_mode", "brief")

        # Reuse parent's HA context builder
        agent_context_section = self._build_agent_context_section(node_context)

        # Tool names only — adapter already knows schemas
        tool_names = [t.get("name", "") for t in tools if t.get("name")]
        tool_names_line = ", ".join(tool_names)

        # Build direct answer policy section
        must_call_tools: list[str] = []
        direct_answer_allowed: list[str] = []
        for cmd in available_commands:
            cmd_name = cmd.get("command_name")
            if not cmd_name:
                continue
            allow_direct = cmd.get("allow_direct_answer")
            if allow_direct is False:
                must_call_tools.append(cmd_name)
            elif allow_direct is True:
                direct_answer_allowed.append(cmd_name)

        direct_answer_section = ""
        if must_call_tools or direct_answer_allowed:
            direct_answer_section = "\nDirect Answer Policy:\n"
            if must_call_tools:
                direct_answer_section += f"- MUST call tools for: {', '.join(sorted(set(must_call_tools)))}\n"
            if direct_answer_allowed:
                direct_answer_section += f"- Direct answers allowed for: {', '.join(sorted(set(direct_answer_allowed)))}\n"

        system_prompt = f"""You are Jarvis, a voice assistant that calls tools to fulfill requests.
Context: room={room}, user={user}, style={voice_mode}
{agent_context_section}
Rules:
- Call ONE tool at a time to fulfill the user's request.
- Extract parameters directly from the user's utterance.
- If no tool is a clear match, call get_command_utterance_examples.
{direct_answer_section}
Response format: JSON ONLY (no extra text).
- Tool call: {{"message":"brief ack","tool_calls":[{{"name":"<tool>","arguments":{{...}}}}],"error":null}}
- Final: {{"message":"<concise spoken reply>","tool_calls":[],"error":null}}

Available tools: {tool_names_line}
"""

        logger.info(
            "📝 Built trained adapter system prompt: %d characters, %d tools available (names only)",
            len(system_prompt),
            len(tools)
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("📝 System prompt (full):\n%s", system_prompt)

        # Write system prompt to file for inspection
        repo_root = Path(__file__).resolve().parents[3]
        prompt_path = repo_root / "temp" / "trained_adapter_system_prompt.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(system_prompt, encoding="utf-8")
        logger.info("📝 System prompt written to %s", prompt_path)

        return system_prompt

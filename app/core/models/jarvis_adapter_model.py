"""
Jarvis Adapter Model - Adapter-Tuned Model for Jarvis

This model uses the same tool execution loop as JarvisToolModel, with a
system prompt optimized for LoRA adapter usage:
- Full descriptions, antipatterns, and parameter info (rich context)
- Only is_primary examples (adapter trained on full example set)
- No fastText classifier (let LLM + adapter handle routing)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.models.jarvis_tool_model import JarvisToolModel
from app.core.prompt_providers.shared.context_builders import (
    build_agent_context_summary,
    device_agent_data,
)
from app.core.tool_call_parser import tool_call_parser

logger = logging.getLogger("uvicorn")


class JarvisAdapterModel(JarvisToolModel):
    """
    Adapter-tuned model implementation for Jarvis.

    Uses rich prompt context combined with LoRA adapters trained on
    command schemas and examples for improved accuracy.
    """

    @property
    def name(self) -> str:
        return "JarvisAdapterModel"

    @property
    def use_tool_classifier(self) -> bool:
        """Disable fastText classifier - let LLM + adapter handle routing."""
        return False

    def _build_agent_context_section(self, node_context: Dict[str, Any]) -> str:
        """Build agent context section (e.g., Home Assistant devices) from node context."""
        agent_context_section = ""
        agents_data = node_context.get("agents", {})
        if not agents_data:
            return agent_context_section

        ha_data = device_agent_data(agents_data)
        if not ha_data:
            return agent_context_section

        # Include light controls for room-based light control
        light_controls = ha_data.get("light_controls", {})
        if light_controls:
            agent_context_section += "\nAvailable Light Controls (room groups):\n"
            for name, info in light_controls.items():
                entity_id = info.get("entity_id", "")
                state = info.get("state", "unknown")
                agent_context_section += f"- {name}: {entity_id} (currently {state})\n"

        # Include device controls for other device types
        device_controls = ha_data.get("device_controls", {})
        if device_controls:
            # Include individual lights (not room groups)
            lights = device_controls.get("light", [])
            room_group_ids = {info.get("entity_id") for info in light_controls.values()} if light_controls else set()
            individual_lights = [l for l in lights if l.get("entity_id") not in room_group_ids]
            if individual_lights:
                agent_context_section += "\nAvailable Individual Lights:\n"
                for dev in individual_lights[:15]:  # Limit to 15
                    entity_id = dev.get("entity_id", "")
                    name = dev.get("name", "")
                    state = dev.get("state", "unknown")
                    agent_context_section += f"- {name}: {entity_id} (currently {state})\n"

            # Include switches (HA devices - use control_device, NOT set_timer/check_timers)
            switches = device_controls.get("switch", [])
            if switches:
                agent_context_section += "\nAvailable Switches (HA devices - use control_device/get_device_status):\n"
                for dev in switches[:10]:  # Limit to 10
                    entity_id = dev.get("entity_id", "")
                    name = dev.get("name", "")
                    state = dev.get("state", "unknown")
                    agent_context_section += f"- {name}: {entity_id} (currently {state})\n"

            # Include scenes
            scenes = device_controls.get("scene", [])
            if scenes:
                agent_context_section += "\nAvailable Scenes:\n"
                for dev in scenes[:20]:  # Limit to 20
                    entity_id = dev.get("entity_id", "")
                    name = dev.get("name", "")
                    agent_context_section += f"- {name}: {entity_id}\n"

        return agent_context_section

    def _build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build system prompt with full context but only primary examples.

        The adapter is trained on the full example set, so we only need
        primary examples in the prompt for disambiguation hints.
        """
        available_commands = available_commands or []
        node_context = node_context or {}

        room = node_context.get("room", "unknown")
        user = node_context.get("user", "default")
        voice_mode = node_context.get("voice_mode", "brief")

        # Build direct answer policy section
        must_call_tools = []
        direct_answer_allowed = []
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

        # Format tools with PRIMARY EXAMPLES ONLY
        tools_section = tool_call_parser.format_tools_for_prompt(
            tools,
            available_commands,
            primary_examples_only=True
        )

        agent_context_section = build_agent_context_summary(node_context)

        system_prompt = f"""You are Jarvis, a voice assistant that uses tools.
Context: room={room}, user={user}, style={voice_mode}
{agent_context_section}
Rules:
- Use tools to fulfill requests. Call ONE tool at a time.
- Select the tool that best matches the user's intent, keywords, or examples; if no tool is a clear match, use get_command_utterance_examples.
- Extract parameters directly from the user's utterance; only use request_validation if required parameters are missing or ambiguous.
{direct_answer_section}

Response format: JSON ONLY (no extra text).
- Tool call: {{"message":"brief ack","tool_calls":[{{"name":"<tool>","arguments":{{...}}}}],"error":null}}
- Final: {{"message":"<concise spoken reply>","tool_calls":[],"error":null}}

Tools:
{tools_section}
"""

        logger.info(
            "📝 Built adapter system prompt: %d characters, %d tools available (primary examples only)",
            len(system_prompt),
            len(tools)
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("📝 System prompt (full):\n%s", system_prompt)

        # Write system prompt to file for inspection
        repo_root = Path(__file__).resolve().parents[3]
        prompt_path = repo_root / "temp" / "adapter_system_prompt.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(system_prompt, encoding="utf-8")
        logger.info("📝 System prompt written to %s", prompt_path)

        return system_prompt

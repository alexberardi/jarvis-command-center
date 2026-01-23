"""
Jarvis Adapter Model - Slim Prompt Model for LoRA Adapter Usage

This model uses the same tool execution loop as JarvisToolModel, but swaps
the system prompt for a slimmer, adapter-friendly version.
"""

import logging
from typing import Dict, Any, List, Optional

from app.core.models.jarvis_tool_model import JarvisToolModel

logger = logging.getLogger("uvicorn")


class JarvisAdapterModel(JarvisToolModel):
    """
    Adapter-tuned model implementation for Jarvis.

    Uses a compact system prompt intended for use with LoRA adapters trained
    on command schemas and examples.
    """

    @property
    def name(self) -> str:
        return "JarvisAdapterModel"

    def _build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build a slim system prompt intended for adapter-tuned models.
        """
        available_commands = available_commands or []
        node_context = node_context or {}

        room = node_context.get("room", "unknown")
        user = node_context.get("user", "default")
        voice_mode = node_context.get("voice_mode", "brief")

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
            parts = []
            if must_call_tools:
                parts.append(f"must_call={', '.join(sorted(set(must_call_tools)))}")
            if direct_answer_allowed:
                parts.append(f"direct_ok={', '.join(sorted(set(direct_answer_allowed)))}")
            direct_answer_section = f"DirectAnswer: {' | '.join(parts)}\n"

        tools_section = self._format_tools_compact(tools)

        system_prompt = (
            "You are Jarvis. Use tools to complete requests.\n"
            f"Context: room={room}, user={user}, style={voice_mode}\n"
            "Rules:\n"
            "- JSON only.\n"
            "- Call tools as needed; one tool call per response.\n"
            "- If required params are missing/ambiguous, call request_validation.\n"
            f"{direct_answer_section}"
            "Response JSON:\n"
            '{"message":"...", "tool_calls":[{"name":"tool","arguments":{...}}], "error":null}\n'
            '{"message":"...", "tool_calls":[], "error":null}\n'
            "Tools:\n"
            f"{tools_section}\n"
        )

        logger.info(
            "📝 Built adapter system prompt: %d characters, %d tools available",
            len(system_prompt),
            len(tools)
        )

        return system_prompt

    def _format_tools_compact(self, tools: List[Dict[str, Any]]) -> str:
        if not tools:
            return "No tools available."

        lines = []
        for tool in tools:
            func = tool.get("function", {}) if isinstance(tool.get("function"), dict) else {}
            name = func.get("name") or tool.get("name") or "unknown"
            description = func.get("description") or tool.get("description") or ""
            description = self._trim_description(description)

            parameters = func.get("parameters", {}) if isinstance(func.get("parameters"), dict) else {}
            props = parameters.get("properties", {}) if isinstance(parameters.get("properties"), dict) else {}
            required = set(parameters.get("required", []) or [])

            if props:
                param_parts = []
                for param_name, param_info in props.items():
                    param_type = self._format_param_type(param_info or {})
                    suffix = "!" if param_name in required else ""
                    param_parts.append(f"{param_name}:{param_type}{suffix}")
                params_str = ", ".join(param_parts)
            else:
                params_str = "none"

            if description:
                lines.append(f"- {name}: {description} params: {params_str}")
            else:
                lines.append(f"- {name} params: {params_str}")

        return "\n".join(lines)

    def _trim_description(self, description: str, max_chars: int = 160) -> str:
        if not isinstance(description, str):
            return ""
        text = description.strip()
        if not text:
            return ""
        first_sentence = text.split(".")[0].strip()
        if first_sentence:
            text = first_sentence + "."
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        return text

    def _format_param_type(self, schema: Dict[str, Any]) -> str:
        if not isinstance(schema, dict):
            return "any"
        ptype = schema.get("type", "any")
        if isinstance(ptype, list):
            ptype = ptype[0] if ptype else "any"

        if ptype == "array":
            items = schema.get("items", {}) or {}
            return f"array<{self._format_param_type(items)}>"
        if ptype == "string":
            fmt = schema.get("format")
            if fmt == "date-time":
                return "datetime"
            if fmt == "date":
                return "date"
        return str(ptype)

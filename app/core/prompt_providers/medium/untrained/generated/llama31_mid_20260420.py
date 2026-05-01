"""
Llama31Mid_20260420 - Phase 6 middle ablation variant.

Keeps the routing-critical signal (1-line tool description + enum values)
but drops the bulky parts the adapter has memorized (per-param descriptions,
examples, critical_rules, antipatterns, duplicate text-mode tools section).

Target: ~50-60% prompt reduction vs full, retain most of the adapter-full
accuracy gains on routing-sensitive commands (control_device, answer_question).
"""

import json
from typing import Any, Dict, List, Optional

from app.core.prompt_providers.medium.untrained.llama31_medium_untrained import (
    Llama31MediumUntrained,
)
from app.core.prompt_providers.shared.context_builders import (
    build_agent_context_summary,
    build_direct_answer_section,
)


def _mid_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Keep name, one-line description, param names, enums, required.

    Drops: per-parameter description text, examples, critical_rules.
    """
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    name = fn.get("name")
    description = fn.get("description") or ""
    # Trim description to first sentence (cap at 160 chars)
    if description:
        first_stop = description.find(". ")
        if first_stop > 0:
            description = description[: first_stop + 1]
        description = description[:160]

    params = fn.get("parameters", {}) or {}
    props = params.get("properties", {}) or {}
    required = params.get("required", []) or []

    mid_params: Dict[str, Any] = {}
    for pname, pschema in props.items():
        entry: Dict[str, Any] = {"type": pschema.get("type", "string")}
        # Preserve enum values — critical for operations/actions/modes
        if "enum" in pschema:
            entry["enum"] = pschema["enum"]
        mid_params[pname] = entry

    result: Dict[str, Any] = {
        "name": name,
        "description": description,
        "params": mid_params,
    }
    if required:
        result["required"] = required
    return result


class Llama31Mid_20260420(Llama31MediumUntrained):
    """Middle ablation: descriptions + enums kept, everything else stripped."""

    @property
    def name(self) -> str:
        return "Llama31Mid_20260420"

    def build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        available_commands = available_commands or []
        node_context = node_context or {}

        _tool_flags = []
        for t in tools:
            fn = t.get("function") if isinstance(t.get("function"), dict) else None
            nm = (fn.get("name") if fn else None) or t.get("name")
            if not nm:
                continue
            _tool_flags.append({
                "command_name": nm,
                "allow_direct_answer": t.get("allow_direct_answer"),
            })
        direct_answer_section: str = build_direct_answer_section(_tool_flags)
        agent_context_section: str = build_agent_context_summary(node_context)

        mid_tools = [_mid_tool(t) for t in tools]
        tool_json: str = json.dumps(mid_tools, indent=2) if mid_tools else "[]"

        identity: str = self.build_context_header(node_context)

        system_prompt: str = f"""{identity}

You are a function calling AI model. Call a function for any request that matches an available tool. Do NOT fabricate data.

To call a function, respond with:
<function=function_name>{{"arg_name": "value"}}</function>

Use ONLY the exact function names from the list below. Do NOT invent names or use dotted paths (e.g., "light.turn_off"). Use actual parameter names from the schema. For params with "enum", use ONLY listed values.

Available functions:
{tool_json}
{direct_answer_section}
{agent_context_section}
"""

        return system_prompt

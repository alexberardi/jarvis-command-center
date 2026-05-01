"""
Llama31Stripped_20260420 - Phase 6 ablation variant of Llama31MediumUntrained.

Adapter-aware minimal prompt: strips tool descriptions, examples, enum
annotations and the duplicate text-mode tool section. Keeps only tool
names + required/optional param names. Target ~5-8k chars vs base 29k.

The adapter has internalized descriptions + routing; this variant tests
whether tool schemas can be compressed without accuracy loss.
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


def _minify_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a tool schema to name + param names + required list.

    Drops descriptions, enum hints, examples — adapter knows these.
    """
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    name = fn.get("name")
    params = fn.get("parameters", {}) or {}
    props = params.get("properties", {}) or {}
    required = params.get("required", []) or []

    param_names = sorted(props.keys())
    minified: Dict[str, Any] = {"name": name, "params": param_names}
    if required:
        minified["required"] = required
    return minified


class Llama31Stripped_20260420(Llama31MediumUntrained):
    """Ablated Llama31 provider: minimal tool schemas, adapter-relied."""

    @property
    def name(self) -> str:
        return "Llama31Stripped_20260420"

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

        minified_tools = [_minify_tool(t) for t in tools]
        tool_json: str = json.dumps(minified_tools, indent=2) if minified_tools else "[]"

        identity: str = self.build_context_header(node_context)

        system_prompt: str = f"""{identity}

You are a function calling AI model. Call a function for any request that matches an available tool. Do NOT fabricate data.

To call a function, respond with:
<function=function_name>{{"arg_name": "value"}}</function>

Use ONLY the exact function names from the list below. Do NOT invent names or use dotted paths (e.g., "light.turn_off"). Use actual parameter names from the schema.

Available functions:
{tool_json}
{direct_answer_section}
{agent_context_section}
"""

        return system_prompt

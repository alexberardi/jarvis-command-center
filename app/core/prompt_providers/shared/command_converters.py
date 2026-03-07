"""
Command-to-tool conversion utilities for prompt providers.

Extracted from JarvisToolModel._convert_commands_to_tools to allow
reuse across prompt providers without inheriting the full model class.
"""

from typing import Any, Dict, List, Optional

from app.request_models.voice_command_request import CommandDefinition


def _normalize_param_type(param_type: Optional[str]) -> Dict[str, Any]:
    """
    Normalize a parameter type string to a JSON Schema fragment.

    Handles array types (array<T>, array[T], T[]), primitives
    (int, float, bool), and date/datetime format hints.

    Args:
        param_type: Raw type string from CommandParameter, or None.

    Returns:
        JSON Schema dict (e.g. {"type": "string"}).
    """
    if not param_type:
        return {"type": "string"}

    raw = param_type.strip().lower()

    # Array variants: array<inner>, array[inner], inner[]
    if raw.startswith("array<") and raw.endswith(">"):
        inner = raw[len("array<"):-1].strip()
        return {"type": "array", "items": _normalize_param_type(inner)}
    if raw.startswith("array[") and raw.endswith("]"):
        inner = raw[len("array["):-1].strip()
        return {"type": "array", "items": _normalize_param_type(inner)}
    if raw.endswith("[]"):
        inner = raw[:-2].strip()
        return {"type": "array", "items": _normalize_param_type(inner)}

    # Numeric
    if raw in {"int", "integer"}:
        return {"type": "integer"}
    if raw in {"float", "number", "double"}:
        return {"type": "number"}

    # Boolean
    if raw in {"bool", "boolean"}:
        return {"type": "boolean"}

    # Date/time with format hints
    if raw == "datetime":
        return {"type": "string", "format": "date-time"}
    if raw == "date":
        return {"type": "string", "format": "date"}

    return {"type": "string"}


# Commands where specific params are always required regardless of schema
_FORCE_REQUIRED_COMMANDS = {"get_calendar_events", "get_sports_scores"}
_FORCE_REQUIRED_PARAMS = {"resolved_datetimes"}


def convert_commands_to_tools(
    commands: List[CommandDefinition],
) -> List[Dict[str, Any]]:
    """
    Convert CommandDefinition objects to OpenAI function-calling tool format.

    This is the canonical conversion used when nodes send CommandDefinition
    lists during warmup. The resulting tool dicts are passed to the LLM
    alongside server-registered tools.

    Args:
        commands: List of CommandDefinition Pydantic models.

    Returns:
        List of tool dicts in OpenAI function format.
    """
    tools: list[Dict[str, Any]] = []

    for cmd in commands:
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in cmd.parameters:
            schema = _normalize_param_type(param.type)
            properties[param.name] = {
                **schema,
                "description": param.description or f"Parameter {param.name}",
            }

            if param.enum_values:
                if properties[param.name].get("type") == "array":
                    properties[param.name].setdefault("items", {})["enum"] = param.enum_values
                else:
                    properties[param.name]["enum"] = param.enum_values

            if param.refinable:
                properties[param.name]["_refinable"] = True

            is_forced = (
                cmd.command_name in _FORCE_REQUIRED_COMMANDS
                and param.name in _FORCE_REQUIRED_PARAMS
            )
            if param.required or is_forced:
                required.append(param.name)

        tool: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": cmd.command_name,
                "description": cmd.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
            "allow_direct_answer": cmd.allow_direct_answer,
            "keywords": cmd.keywords,
        }
        if cmd.examples:
            tool["examples"] = [ex.model_dump() for ex in cmd.examples]
        if cmd.antipatterns:
            tool["antipatterns"] = [ap.model_dump() for ap in cmd.antipatterns]

        tools.append(tool)

    return tools

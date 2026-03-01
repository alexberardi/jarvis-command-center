"""
Parameter validation helpers for Jarvis voice assistant.

This module provides utilities for validating tool call parameters
against expected types.
"""

import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def normalize_param_type(param_type: Optional[str]) -> Tuple[Optional[str], bool]:
    """
    Normalize a parameter type string and detect if it's an array.

    Supports formats:
    - "string", "integer", etc.
    - "array<string>", "array[datetime]", "string[]"

    Args:
        param_type: The parameter type string

    Returns:
        Tuple of (base_type, is_array)
    """
    if not param_type:
        return None, False

    raw = param_type.strip().lower()

    # Handle array<type> format
    if raw.startswith("array<") and raw.endswith(">"):
        return raw[len("array<"):-1].strip(), True

    # Handle array[type] format
    if raw.startswith("array[") and raw.endswith("]"):
        return raw[len("array["):-1].strip(), True

    # Handle type[] format
    if raw.endswith("[]"):
        return raw[:-2].strip(), True

    return raw, False


def is_iso_date(value: str) -> bool:
    """
    Check if a string is a valid ISO date (YYYY-MM-DD).

    Args:
        value: The string to check

    Returns:
        True if valid ISO date format
    """
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", value))


def is_iso_datetime(value: str) -> bool:
    """
    Check if a string is a valid ISO datetime with timezone.

    Args:
        value: The string to check

    Returns:
        True if valid ISO datetime with timezone
    """
    if "T" not in value:
        return False

    normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False

    return parsed.tzinfo is not None


def validate_scalar(value: Any, base_type: str) -> bool:
    """
    Validate a scalar value against an expected type.

    Args:
        value: The value to validate
        base_type: Expected type (string, integer, float, bool, date, datetime)

    Returns:
        True if value matches the expected type
    """
    if base_type in {"string"}:
        return isinstance(value, str)

    if base_type in {"int", "integer"}:
        return isinstance(value, int) and not isinstance(value, bool)

    if base_type in {"float", "number", "double"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if base_type in {"bool", "boolean"}:
        return isinstance(value, bool)

    if base_type == "date":
        return isinstance(value, str) and is_iso_date(value)

    if base_type == "datetime":
        return isinstance(value, str) and is_iso_datetime(value)

    # Unknown types pass validation (permissive)
    return True


def validate_value(value: Any, base_type: str, is_array: bool) -> bool:
    """
    Validate a value against a type (scalar or array).

    Args:
        value: The value to validate
        base_type: Expected base type
        is_array: Whether the value should be an array

    Returns:
        True if value matches the expected type
    """
    if is_array:
        if not isinstance(value, list):
            return False
        return all(validate_scalar(item, base_type) for item in value)

    return validate_scalar(value, base_type)


def find_invalid_params(
    tool_calls: List[Dict[str, Any]],
    param_types: Dict[str, Dict[str, str]],
    param_enums: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> List[str]:
    """
    Find invalid parameters in tool calls.

    Args:
        tool_calls: List of tool calls with function.name and function.arguments
        param_types: Mapping of tool_name -> {param_name: param_type}
        param_enums: Optional mapping of tool_name -> {param_name: [allowed_values]}

    Returns:
        List of invalid parameter descriptions (e.g., "tool.param expected type")
    """
    invalid: List[str] = []

    if not param_types and not param_enums:
        return invalid

    for call in tool_calls:
        tool_name = call.get("function", {}).get("name")
        if not tool_name:
            continue

        args_raw = call.get("function", {}).get("arguments", "{}")
        try:
            args_obj = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            args_obj = {}

        if not isinstance(args_obj, dict):
            continue

        # Type validation
        if param_types and tool_name in param_types:
            for param_name, param_type in param_types[tool_name].items():
                if param_name not in args_obj:
                    continue

                base_type, is_array = normalize_param_type(param_type)
                if not base_type:
                    continue

                if not validate_value(args_obj.get(param_name), base_type, is_array):
                    invalid.append(f"{tool_name}.{param_name} expected {param_type}")

        # Enum validation
        if param_enums and tool_name in param_enums:
            for param_name, allowed in param_enums[tool_name].items():
                value = args_obj.get(param_name)
                if value is None:
                    continue
                if str(value) not in allowed:
                    invalid.append(
                        f"{tool_name}.{param_name} must be one of: {', '.join(allowed)}"
                    )

    return invalid

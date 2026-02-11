"""
Date resolution helpers for Jarvis voice assistant.

This module provides utilities for normalizing date keys, flattening date context,
and resolving relative date expressions to ISO datetime strings.
"""

import re
from datetime import datetime, timedelta, timezone as tz
from typing import Any, Dict, List, Optional, Tuple


def normalize_date_key(raw: str) -> str:
    """
    Normalize a date key string for lookup.

    Converts to lowercase, replaces spaces and colons with underscores,
    and collapses multiple whitespace.

    Args:
        raw: The raw date key string

    Returns:
        Normalized date key
    """
    text = raw.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = text.replace(":", "_")
    return text


RELATIVE_TIME_PATTERN = re.compile(
    r"^in_(\d+)_(minutes|hours|days)(?:_(\d+)_(minutes))?$"
)


def resolve_relative_time(key: str, date_context: Dict[str, Any]) -> Optional[str]:
    """
    Resolve a relative time key like 'in_30_minutes' to an ISO datetime string.

    Args:
        key: Normalized relative time key (e.g., 'in_30_minutes', 'in_2_hours')
        date_context: The nested date context object (needs current.datetime)

    Returns:
        ISO datetime string or None if key doesn't match pattern
    """
    match = RELATIVE_TIME_PATTERN.match(key)
    if not match:
        return None

    now_str = (
        date_context.get("current", {}).get("datetime")
        or date_context.get("current", {}).get("utc_start_of_day")
    )
    if not now_str:
        return None

    try:
        normalized = now_str.replace("Z", "+00:00") if now_str.endswith("Z") else now_str
        now = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    amount = int(match.group(1))
    unit = match.group(2)

    offset = timedelta()
    if unit == "minutes":
        offset = timedelta(minutes=amount)
    elif unit == "hours":
        offset = timedelta(hours=amount)
    elif unit == "days":
        offset = timedelta(days=amount)

    # Handle compound: in_1_hours_30_minutes
    if match.group(3) and match.group(4):
        offset += timedelta(minutes=int(match.group(3)))

    result = now + offset
    return result.astimezone(tz.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def flatten_date_context(nested_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a nested date context object into a simple key-value map.

    Handles the complex date context structure and extracts dates from:
    - current (today)
    - relative_dates (tomorrow, yesterday, etc.)
    - weekdays (next_monday, next_friday, etc.)
    - weeks.this_week (this_monday, this_tuesday, etc.)
    - time_expressions (at 3pm, at noon, etc.)
    - bucket lists (weekend, weeks, months, years)

    Args:
        nested_context: The nested date context from generate_date_context_object

    Returns:
        Flattened dict mapping normalized keys to datetime strings or lists
    """
    flat: Dict[str, Any] = {}

    if not isinstance(nested_context, dict):
        return flat

    # Extract today from current
    current = nested_context.get("current", {})
    if isinstance(current, dict) and isinstance(current.get("utc_start_of_day"), str):
        flat["today"] = current["utc_start_of_day"]

    # Extract relative dates
    relative = nested_context.get("relative_dates", {})
    if isinstance(relative, dict):
        for key, value in relative.items():
            if not isinstance(value, dict):
                continue
            if isinstance(value.get("utc_start_of_day"), str):
                flat[key] = value["utc_start_of_day"]
            elif isinstance(value.get("datetime"), str):
                flat[key] = value["datetime"]

    # Extract bucket lists (weekend, weeks, months, years)
    for bucket_name in ("weekend", "weeks", "months", "years"):
        bucket = nested_context.get(bucket_name, {})
        if not isinstance(bucket, dict):
            continue
        for key, value in bucket.items():
            if not isinstance(value, list):
                continue
            dates = [
                item.get("utc_start_of_day")
                for item in value
                if isinstance(item, dict) and isinstance(item.get("utc_start_of_day"), str)
            ]
            if dates:
                flat[key] = dates

    # Extract weekdays
    weekdays = nested_context.get("weekdays", {})
    if isinstance(weekdays, dict):
        for key, value in weekdays.items():
            if isinstance(value, dict) and isinstance(value.get("utc_start_of_day"), str):
                flat[key] = value["utc_start_of_day"]

    # Extract this_week entries (this_monday, this_tuesday, etc.)
    this_week = nested_context.get("weeks", {}).get("this_week", [])
    if isinstance(this_week, list):
        for entry in this_week:
            if not isinstance(entry, dict):
                continue
            day = entry.get("day")
            if isinstance(day, str) and isinstance(entry.get("utc_start_of_day"), str):
                flat[f"this_{day.strip().lower()}"] = entry["utc_start_of_day"]

    # Extract time expressions
    time_expressions = nested_context.get("time_expressions", {})
    if isinstance(time_expressions, dict):
        for key, value in time_expressions.items():
            if isinstance(value, str):
                flat[normalize_date_key(key)] = value

    return flat


def parse_time_string(time_str: str) -> Tuple[int, int]:
    """
    Parse a time string like "9am", "3pm", "9_30am", "3_45pm".

    Args:
        time_str: The time string to parse

    Returns:
        Tuple of (hour, minute) in 24-hour format
    """
    # Try format with minutes: 9_30am, 3_45pm
    match = re.match(r"(\d+)_(\d+)(am|pm)", time_str)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if match.group(3) == "pm" and hour != 12:
            hour += 12
        elif match.group(3) == "am" and hour == 12:
            hour = 0
        return hour, minute

    # Try format without minutes: 9am, 3pm
    match = re.match(r"(\d+)(am|pm)", time_str)
    if match:
        hour = int(match.group(1))
        if match.group(2) == "pm" and hour != 12:
            hour += 12
        elif match.group(2) == "am" and hour == 12:
            hour = 0
        return hour, 0

    return 0, 0


def apply_time_modifier(base_datetime: str, modifier: str) -> Optional[str]:
    """
    Apply a time modifier to a base datetime string.

    Args:
        base_datetime: ISO datetime string (e.g., "2025-01-15T00:00:00Z")
        modifier: Time modifier (e.g., "morning", "afternoon", "at_3pm")

    Returns:
        Modified ISO datetime string or None if base is invalid
    """
    try:
        normalized = base_datetime.replace("Z", "+00:00") if base_datetime.endswith("Z") else base_datetime
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    time_map = {
        "morning": 7,
        "afternoon": 13,
        "evening": 18,
        "night": 21,
        "noon": 12,
        "midnight": 0,
    }

    if modifier in time_map:
        dt = dt.replace(hour=time_map[modifier], minute=0, second=0, microsecond=0)
    elif modifier.startswith("at_"):
        hour, minute = parse_time_string(modifier[3:])
        dt = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)

    return dt.astimezone(tz.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def is_datetime_param(param_schema: Dict[str, Any]) -> bool:
    """
    Check if a parameter schema is a datetime parameter.

    Args:
        param_schema: JSON schema for the parameter

    Returns:
        True if this is a datetime parameter
    """
    if not isinstance(param_schema, dict):
        return False
    if param_schema.get("format") == "date-time":
        return True
    if param_schema.get("type") == "array":
        items = param_schema.get("items", {})
        return isinstance(items, dict) and items.get("format") == "date-time"
    return False


def is_datetime_array(param_schema: Dict[str, Any]) -> bool:
    """
    Check if a parameter schema is an array of datetimes.

    Args:
        param_schema: JSON schema for the parameter

    Returns:
        True if this is an array of datetime parameters
    """
    if not isinstance(param_schema, dict):
        return False
    if param_schema.get("type") != "array":
        return False
    items = param_schema.get("items", {})
    return isinstance(items, dict) and items.get("format") == "date-time"


def resolve_date_keys(
    date_keys: List[str],
    date_context: Dict[str, Any]
) -> Tuple[List[str], List[str]]:
    """
    Resolve date keys to datetime strings.

    Args:
        date_keys: List of date key strings (e.g., ["tomorrow", "morning"])
        date_context: The nested date context object

    Returns:
        Tuple of (resolved_dates, unresolved_keys)
    """
    if not date_keys:
        return [], []

    normalized_keys = [normalize_date_key(key) for key in date_keys if isinstance(key, str)]
    flat_context = flatten_date_context(date_context)

    resolved: List[str] = []
    unresolved: List[str] = []

    for key in normalized_keys:
        # Try relative time resolution first (e.g., in_30_minutes, in_2_hours)
        relative_result = resolve_relative_time(key, date_context)
        if relative_result:
            resolved.append(relative_result)
            continue

        value = flat_context.get(key)
        if isinstance(value, list):
            resolved.extend([v for v in value if isinstance(v, str)])
        elif isinstance(value, str):
            resolved.append(value)
        else:
            # Key not found - track for potential LLM fallback
            # But skip time modifiers (they combine with date keys)
            if key not in {"morning", "afternoon", "evening", "night", "noon", "midnight"} and not key.startswith("at_"):
                unresolved.append(key)

    # Handle date + time modifier combination
    date_key = None
    for key in normalized_keys:
        value = flat_context.get(key)
        if isinstance(value, str):
            date_key = key
            break

    time_key = None
    for key in normalized_keys:
        if key in {"morning", "afternoon", "evening", "night", "noon", "midnight"} or key.startswith("at_"):
            time_key = key
            break

    if date_key and time_key:
        base = flat_context.get(date_key)
        if isinstance(base, str):
            combined = apply_time_modifier(base, time_key)
            if combined:
                resolved.append(combined)
        elif isinstance(base, list):
            for entry in base:
                if not isinstance(entry, str):
                    continue
                combined = apply_time_modifier(entry, time_key)
                if combined:
                    resolved.append(combined)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for entry in resolved:
        if entry in seen:
            continue
        seen.add(entry)
        unique.append(entry)

    return unique, unresolved

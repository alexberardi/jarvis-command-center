# Date Key Integration - Command Center API

## Overview

This PRD describes the changes needed in the Jarvis Command Center (jcc) API to integrate with the new date key extraction system in jarvis-llm-proxy-api.

## Background

The LLM proxy now supports extracting semantic date keys from natural language instead of absolute timestamps. This allows:
- Timezone-correct date handling in the application layer
- Domain-specific date interpretation
- No dependency on the LLM "knowing" the current date

## Changes Required

### 1. Request Model Update

Add `include_date_context` to requests sent to the LLM proxy:

```python
# When calling jarvis-llm-proxy-api
request = {
    "model": "jarvis",
    "messages": [...],
    "include_date_context": True  # NEW: Enable date key extraction
}
```

**Always set this to `True`** for now. The date extraction adapter is lightweight and false negatives (missing dates) are worse than false positives (empty `[]`). We can add conditional logic later if latency becomes an issue.

### 2. Response Handling

The LLM proxy response will now include a `date_keys` field when `include_date_context=True`:

```json
{
  "id": "chatcmpl-xxx",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "{\"tool_calls\": [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Miami\"}}]}"
    }
  }],
  "date_keys": ["tomorrow", "morning"]
}
```

### 3. Tool Parameter Resolution

After receiving the response, jcc must:

1. **Parse tool calls** from the response content
2. **Identify datetime parameters** in the called tool's schema
3. **Resolve date keys** to actual datetimes using the local date context dictionary
4. **Inject resolved values** into the tool call arguments

#### Implementation

```python
def is_datetime_param(param_schema: dict) -> bool:
    """
    Check if a parameter is a datetime type using JSON Schema conventions.
    
    Handles:
    - { "type": "string", "format": "date-time" }
    - { "type": "array", "items": { "type": "string", "format": "date-time" } }
    """
    # Single datetime
    if param_schema.get("format") == "date-time":
        return True
    
    # Array of datetimes
    if param_schema.get("type") == "array":
        items = param_schema.get("items", {})
        if items.get("format") == "date-time":
            return True
    
    return False


def is_datetime_array(param_schema: dict) -> bool:
    """Check if parameter is an array of datetimes."""
    if param_schema.get("type") == "array":
        items = param_schema.get("items", {})
        return items.get("format") == "date-time"
    return False


def resolve_date_keys_in_tool_calls(
    tool_calls: list, 
    date_keys: list, 
    date_context: dict,
    tools: list
) -> list:
    """
    Resolve date keys into actual datetime values for tool call parameters.
    
    Args:
        tool_calls: List of tool calls from LLM response
        date_keys: Extracted date keys from response (e.g., ["tomorrow", "morning"])
        date_context: Dictionary mapping keys to datetime values
        tools: List of tool definitions (OpenAI format)
    
    Returns:
        Tool calls with datetime parameters populated
    """
    # Build tool lookup
    tool_map = {t["function"]["name"]: t["function"] for t in tools}
    
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_def = tool_map.get(tool_name)
        
        if not tool_def:
            continue
        
        # Get parameter schemas - nested under function.parameters.properties
        param_schemas = tool_def.get("parameters", {}).get("properties", {})
        
        for param_name, param_schema in param_schemas.items():
            if not is_datetime_param(param_schema):
                continue
            
            # Resolve the date keys for this parameter
            resolved = resolve_keys_to_datetimes(date_keys, date_context)
            
            if resolved:
                if is_datetime_array(param_schema):
                    tool_call["arguments"][param_name] = resolved
                else:
                    # Single datetime - use first resolved value
                    tool_call["arguments"][param_name] = resolved[0]
    
    return tool_calls


def resolve_keys_to_datetimes(keys: list, date_context: dict) -> list:
    """
    Convert date keys to actual datetime values.
    
    Args:
        keys: List of keys like ["tomorrow", "morning"]
        date_context: The pre-built date context dictionary
    
    Returns:
        List of ISO datetime strings
    """
    resolved = []
    
    for key in keys:
        if key not in date_context:
            continue
            
        value = date_context[key]
        
        if isinstance(value, str):
            # Direct ISO string
            resolved.append(value)
        elif isinstance(value, dict):
            # Object with utc_start_of_day or similar
            if "utc_start_of_day" in value:
                resolved.append(value["utc_start_of_day"])
            elif "iso" in value:
                resolved.append(value["iso"])
        elif isinstance(value, list):
            # Multi-day periods like "this_weekend"
            for item in value:
                if isinstance(item, dict) and "utc_start_of_day" in item:
                    resolved.append(item["utc_start_of_day"])
                elif isinstance(item, str):
                    resolved.append(item)
    
    return resolved
```

### 4. Date Context Dictionary

The existing `build_date_context()` function returns a nested structure. We need to flatten it into a simple key→value dictionary for date key resolution.

#### Flattening the Existing Date Context

Your current date context structure looks like this:
```json
{
  "current": { "utc_start_of_day": "2026-01-20T00:00:00" },
  "relative_dates": { "tomorrow": { "utc_start_of_day": "2026-01-21T00:00:00" } },
  "weekend": { "this_weekend": [...] },
  "weekdays": { "next_monday": { "utc_start_of_day": "..." } },
  "time_expressions": { "tomorrow morning": "2026-01-21T07:00:00Z" }
}
```

Flatten it to our key format:
```json
{
  "today": "2026-01-20T00:00:00Z",
  "tomorrow": "2026-01-21T00:00:00Z",
  "this_weekend": ["2026-01-24T00:00:00Z", "2026-01-25T00:00:00Z"],
  "next_monday": "2026-01-25T00:00:00Z",
  "morning": "07:00:00",
  "at_3pm": "15:00:00"
}
```

**Flattening function:**

```python
def flatten_date_context(nested_context: dict) -> dict:
    """
    Flatten the nested date context into a simple key->value dictionary.
    
    Keys use underscores (not spaces): "tomorrow_morning" not "tomorrow morning"
    """
    flat = {}
    
    # Current day as "today"
    if "current" in nested_context:
        flat["today"] = nested_context["current"].get("utc_start_of_day")
    
    # Relative dates: tomorrow, yesterday, day_after_tomorrow, etc.
    for key, value in nested_context.get("relative_dates", {}).items():
        # Normalize key: "last_night" stays as-is
        normalized_key = key.replace(" ", "_").lower()
        if isinstance(value, dict):
            # Prefer datetime if available (for last_night), else utc_start_of_day
            flat[normalized_key] = value.get("datetime") or value.get("utc_start_of_day")
        else:
            flat[normalized_key] = value
    
    # Weekends: this_weekend, next_weekend, last_weekend
    for key, days in nested_context.get("weekend", {}).items():
        normalized_key = key.replace(" ", "_").lower()
        # Extract utc_start_of_day from each day
        flat[normalized_key] = [
            day.get("utc_start_of_day") for day in days if isinstance(day, dict)
        ]
    
    # Weeks: this_week, next_week, last_week
    for key, days in nested_context.get("weeks", {}).items():
        normalized_key = key.replace(" ", "_").lower()
        flat[normalized_key] = [
            day.get("utc_start_of_day") for day in days if isinstance(day, dict)
        ]
    
    # Months: this_month, next_month, last_month (start/end pairs)
    for key, dates in nested_context.get("months", {}).items():
        normalized_key = key.replace(" ", "_").lower()
        flat[normalized_key] = [
            d.get("utc_start_of_day") for d in dates if isinstance(d, dict)
        ]
    
    # Years: this_year, next_year, last_year
    for key, dates in nested_context.get("years", {}).items():
        normalized_key = key.replace(" ", "_").lower()
        flat[normalized_key] = [
            d.get("utc_start_of_day") for d in dates if isinstance(d, dict)
        ]
    
    # Weekdays: next_monday, last_tuesday, etc.
    for key, value in nested_context.get("weekdays", {}).items():
        normalized_key = key.replace(" ", "_").lower()
        if isinstance(value, dict):
            flat[normalized_key] = value.get("utc_start_of_day")
        else:
            flat[normalized_key] = value
    
    # Time modifiers (static values, not date-dependent)
    flat["morning"] = "07:00:00"
    flat["afternoon"] = "14:00:00"
    flat["evening"] = "19:00:00"
    flat["night"] = "20:00:00"
    flat["noon"] = "12:00:00"
    flat["midnight"] = "00:00:00"
    
    # Process time_expressions for both specific times and combined keys
    for expr, value in nested_context.get("time_expressions", {}).items():
        # Normalize: "at 3pm" -> "at_3pm", "tomorrow morning" -> "tomorrow_morning"
        normalized = expr.replace(" ", "_").replace(":", "_").lower()
        flat[normalized] = value
    
    # Ensure combined day+time keys exist (from time_expressions)
    # These are standalone keys: tonight, last_night, tomorrow_morning, etc.
    combined_keys = [
        "tonight", "last_night", "tomorrow_night",
        "tomorrow_morning", "tomorrow_afternoon", "tomorrow_evening",
        "yesterday_morning", "yesterday_afternoon", "yesterday_evening",
        "this_morning", "this_afternoon", "this_evening",
        "at_breakfast", "during_lunch", "at_dinner"
    ]
    for key in combined_keys:
        if key not in flat:
            # Try to find it in time_expressions with space separator
            space_key = key.replace("_", " ")
            if space_key in nested_context.get("time_expressions", {}):
                flat[key] = nested_context["time_expressions"][space_key]
    
    # Add this_X weekdays (currently missing from your context)
    # These need to be computed based on current weekday
    _add_this_weekdays(flat, nested_context)
    
    return flat


def _add_this_weekdays(flat: dict, nested_context: dict):
    """
    Add this_monday, this_tuesday, etc. based on this_week data.
    
    The 'this_week' array contains all days of the current week.
    """
    this_week = nested_context.get("weeks", {}).get("this_week", [])
    
    day_map = {
        "sunday": 0, "monday": 1, "tuesday": 2, "wednesday": 3,
        "thursday": 4, "friday": 5, "saturday": 6
    }
    
    for day_data in this_week:
        if not isinstance(day_data, dict):
            continue
        day_name = day_data.get("day", "").lower()
        if day_name in day_map:
            flat[f"this_{day_name}"] = day_data.get("utc_start_of_day")
```

#### Key Naming Convention

| Spoken Form | Key Format |
|-------------|------------|
| `tomorrow` | `tomorrow` |
| `next Tuesday` | `next_tuesday` |
| `last weekend` | `last_weekend` |
| `this morning` | `morning` (modifier, applied to date) |
| `at 3pm` | `at_3pm` |
| `at 9:30am` | `at_9_30am` |

#### Caching the Date Keys Vocabulary

Cache the supported keys from the API. Refresh on startup and every 24 hours:

```python
import time
from functools import lru_cache

_date_keys_cache = None
_date_keys_cache_time = 0
DATE_KEYS_CACHE_TTL = 24 * 60 * 60  # 24 hours in seconds


def get_supported_date_keys(force_refresh: bool = False) -> list:
    """
    Fetch the list of supported date keys from the LLM proxy.
    
    Cached for 24 hours. Call with force_refresh=True on startup.
    """
    global _date_keys_cache, _date_keys_cache_time
    
    now = time.time()
    cache_expired = (now - _date_keys_cache_time) > DATE_KEYS_CACHE_TTL
    
    if _date_keys_cache is None or cache_expired or force_refresh:
        try:
            response = requests.get(
                f"{LLM_PROXY_URL}/v1/adapters/date-keys",
                timeout=5
            )
            response.raise_for_status()
            _date_keys_cache = response.json()["keys"]
            _date_keys_cache_time = now
        except Exception as e:
            # If refresh fails but we have cached data, use it
            if _date_keys_cache is not None:
                print(f"Warning: Failed to refresh date keys, using cache: {e}")
            else:
                raise
    
    return _date_keys_cache


def validate_date_context_coverage(date_context: dict) -> list:
    """
    Check if date_context covers all supported keys.
    
    Returns list of missing keys.
    """
    supported = set(get_supported_date_keys())
    provided = set(date_context.keys())
    return list(supported - provided)
```

**Call on startup:**
```python
# In app initialization
get_supported_date_keys(force_refresh=True)
```

### 5. Deprecate Pre-LLM Date Injection

**Decision:** The new `date_keys` system **replaces** the old pre-LLM relative date injection.

The old approach had the LLM try to understand dates from a complex JSON blob. The new approach:
1. LLM focuses on intent/command extraction
2. Date extraction happens via dedicated adapter
3. Date resolution happens post-LLM in application code

**Migration path:**
1. Implement new date key resolution
2. Remove date context from system prompt
3. Remove pre-LLM date injection logic
4. Monitor for regressions
5. Delete old code after confirming new path works

### 6. Handling Date Keys

Date keys come in two flavors:

**1. Standalone combined keys** - common phrases returned as single keys:

| User says | Date keys returned |
|-----------|-------------------|
| "tomorrow morning" | `["tomorrow_morning"]` |
| "tonight" | `["tonight"]` |
| "last night" | `["last_night"]` |
| "this weekend" | `["this_weekend"]` |

**2. Decomposed keys** - weekday + time still separate:

| User says | Date keys returned |
|-----------|-------------------|
| "next Tuesday at 3pm" | `["next_tuesday", "at_3pm"]` |
| "next Friday morning" | `["next_friday", "morning"]` |

When resolving, check for standalone keys first, then combine if needed:

```python
def resolve_date_keys(date_keys: list, date_context: dict) -> str:
    """
    Resolve date keys to a datetime value.
    
    Priority:
    1. Direct lookup for standalone combined keys (tonight, tomorrow_morning, etc.)
    2. Combine decomposed keys (next_tuesday + morning)
    """
    # If single key, try direct lookup
    if len(date_keys) == 1:
        key = date_keys[0]
        if key in date_context:
            return _extract_datetime(date_context[key])
    
    # Check for standalone combined key first
    COMBINED_KEYS = {
        "tonight", "last_night", "tomorrow_night",
        "tomorrow_morning", "tomorrow_afternoon", "tomorrow_evening",
        "yesterday_morning", "yesterday_afternoon", "yesterday_evening",
        "this_morning", "this_afternoon", "this_evening"
    }
    
    for key in date_keys:
        if key in COMBINED_KEYS and key in date_context:
            return _extract_datetime(date_context[key])
    
    # Fall back to combining decomposed keys
    return _combine_decomposed_keys(date_keys, date_context)


def _extract_datetime(value) -> str:
    """Extract datetime string from various value formats."""
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value.get("datetime") or value.get("utc_start_of_day") or value.get("iso")
    elif isinstance(value, list) and value:
        # Multi-day period - return first day
        return _extract_datetime(value[0])
    return None


def _combine_decomposed_keys(date_keys: list, date_context: dict) -> str:
    """
    Combine decomposed date + time keys.
    
    Example: ["next_tuesday", "morning"] -> "2026-01-27T07:00:00Z"
    """
    base_date = None
    time_modifier = None
    
    DATE_KEYS = {"today", "tomorrow", "yesterday", "day_after_tomorrow", 
                 "day_before_yesterday", "this_weekend", "next_weekend", 
                 "last_weekend", "this_week", "next_week", "last_week",
                 "this_month", "next_month", "last_month",
                 "this_year", "next_year", "last_year"}
    DATE_KEYS.update(f"{prefix}_{day}" for prefix in ["this", "next", "last"] 
                     for day in ["monday", "tuesday", "wednesday", "thursday", 
                                 "friday", "saturday", "sunday"])
    
    TIME_KEYS = {"morning", "afternoon", "evening", "night", "noon", "midnight"}
    
    for key in date_keys:
        if key in DATE_KEYS:
            base_date = date_context.get(key)
        elif key in TIME_KEYS:
            time_modifier = key
        elif key.startswith("at_"):
            time_modifier = key
    
    if base_date is None:
        return None
    
    # Get base datetime
    result = _extract_datetime(base_date)
    
    # Apply time modifier if present
    if time_modifier and result:
        result = apply_time_modifier(result, time_modifier)
    
    return result


def apply_time_modifier(base_datetime: str, modifier: str) -> str:
    """
    Apply a time modifier to a base datetime.
    
    Modifiers:
    - morning: 07:00
    - afternoon: 13:00
    - evening: 18:00
    - night: 21:00
    - noon: 12:00
    - midnight: 00:00
    - at_Xam/at_Xpm: specific hour
    """
    from datetime import datetime
    
    dt = datetime.fromisoformat(base_datetime.replace("Z", "+00:00"))
    
    TIME_MAP = {
        "morning": 7,
        "afternoon": 13,
        "evening": 18,
        "night": 21,
        "noon": 12,
        "midnight": 0,
    }
    
    if modifier in TIME_MAP:
        dt = dt.replace(hour=TIME_MAP[modifier], minute=0, second=0, microsecond=0)
    elif modifier.startswith("at_"):
        # Parse "at_3pm", "at_9_30am", etc.
        time_str = modifier[3:]  # Remove "at_"
        hour, minute = parse_time_string(time_str)
        dt = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    return dt.isoformat().replace("+00:00", "Z")


def parse_time_string(time_str: str) -> tuple[int, int]:
    """Parse time strings like '3pm', '9am', '9_30am' into (hour, minute)."""
    import re
    
    # Handle "X_30am/pm" format
    match = re.match(r"(\d+)_(\d+)(am|pm)", time_str)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if match.group(3) == "pm" and hour != 12:
            hour += 12
        elif match.group(3) == "am" and hour == 12:
            hour = 0
        return hour, minute
    
    # Handle "Xam/pm" format
    match = re.match(r"(\d+)(am|pm)", time_str)
    if match:
        hour = int(match.group(1))
        if match.group(2) == "pm" and hour != 12:
            hour += 12
        elif match.group(2) == "am" and hour == 12:
            hour = 0
        return hour, 0
    
    return 0, 0
```

### 7. Missing Date References

If `date_keys` is empty but a tool requires a datetime parameter:

**This is command-specific.** Each tool/command decides its default behavior:
- Some tools default to "today" (e.g., weather)
- Some tools require explicit dates (e.g., calendar event creation)
- Some tools have no sensible default (e.g., "what happened on X date?")

Configure this per-tool in the tool schema or command configuration.

## Testing

1. **Unit tests** for:
   - `is_datetime_param()` - JSON Schema detection
   - `resolve_keys_to_datetimes()` - key to datetime conversion
   - `combine_date_and_time_keys()` - date + time modifier combination
   - `parse_time_string()` - time string parsing

2. **Integration tests** with LLM proxy using `include_date_context=True`

3. **Edge cases:**
   - No date keys returned, but datetime param required
   - Multiple datetime params in one tool
   - Multi-day periods (weekends, weeks)
   - Time modifiers combined with dates
   - Three keys: `["next_tuesday", "morning", "at_9am"]` - at_9am should win over morning

## Rollout

1. ✅ Update request model to include `include_date_context`
2. ✅ Implement response parsing for `date_keys`
3. ✅ Implement date key resolution logic
4. ✅ Implement date key caching (24h TTL + startup refresh)
5. Update date context dictionary to match key vocabulary
6. Remove old pre-LLM date injection
7. Test with existing commands
8. Monitor for regressions in date handling

## API Reference

### GET /v1/adapters/date-keys

Returns the vocabulary of supported date keys.

**Authentication:** None required (public vocabulary)

```bash
curl http://llm-proxy:8000/v1/adapters/date-keys
```

```json
{
  "version": "1.0",
  "keys": [
    "today", "tomorrow", "yesterday",
    "day_after_tomorrow", "day_before_yesterday",
    "morning", "afternoon", "evening", "night", "noon", "midnight",
    "next_monday", "next_tuesday", "next_wednesday", "next_thursday", "next_friday", "next_saturday", "next_sunday",
    "last_monday", "last_tuesday", "last_wednesday", "last_thursday", "last_friday", "last_saturday", "last_sunday",
    "this_monday", "this_tuesday", "this_wednesday", "this_thursday", "this_friday", "this_saturday", "this_sunday",
    "this_week", "next_week", "last_week",
    "this_weekend", "next_weekend", "last_weekend",
    "this_month", "next_month", "last_month"
  ],
  "patterns": {
    "time": "at_Xam, at_Xpm, at_X_30am, at_X_30pm (X = 1-12)"
  }
}
```

Use this endpoint to:
- Validate your date context dictionary coverage
- Generate client-side constants
- Stay in sync with supported keys

## Decisions Log

| Question | Decision | Rationale |
|----------|----------|-----------|
| Use JSON Schema conventions? | ✅ Yes | JCC uses JSON Schema; resolver now checks `format: "date-time"` |
| Fix schema path? | ✅ Yes | Updated to `tool["function"]["parameters"]["properties"]` |
| Replace or layer on old date injection? | **Replace** | Cleaner architecture, one source of truth |
| Always include_date_context or conditional? | **Always true** | Adapter is lightweight; false negatives worse than false positives |
| Cache date keys? TTL? | **Yes, 24h + startup** | Keys are static vocabulary; no need to hit API repeatedly |
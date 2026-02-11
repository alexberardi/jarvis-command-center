# Relative Time Resolution in Command Center

## Overview

The date key system handles semantic keys (`tomorrow`, `morning`, `next_monday`) but not relative time offsets ("in 30 minutes", "in 2 hours", "in 3 days"). This is a general-purpose feature — any command that accepts datetime parameters benefits from relative time resolution.

This PRD describes the command-center changes needed to support relative time date keys.

## Background

### Current Date Key Flow

```
User: "What's the weather tomorrow morning?"
  │
  ▼
LLM Proxy (date-key adapter) extracts: ["tomorrow", "morning"]
  │
  ▼
Command Center resolves using date context:
  tomorrow = 2026-02-11
  morning = 07:00
  → 2026-02-11T07:00:00-05:00
  │
  ▼
Injected into tool call parameters
```

### The Gap

```
User: "Remind me in 30 minutes to check the oven"
  │
  ▼
LLM Proxy (date-key adapter) extracts: []  ← No matching key!
```

Relative expressions like "in 30 minutes", "in 2 hours", "in 3 days" don't map to any existing date key.

## Proposed Solution

### New Date Keys: Relative Time Offsets

Add a new category of date keys for relative time:

| Key Pattern | Example Input | Resolved Value |
|------------|---------------|----------------|
| `in_N_minutes` | "in 30 minutes" | now + 30 min |
| `in_N_hours` | "in 2 hours" | now + 2 hours |
| `in_N_days` | "in 3 days" | now + 3 days |
| `in_N_hours_N_minutes` | "in 1 hour 30 minutes" | now + 90 min |

These are **dynamic keys** — unlike `tomorrow` which maps to a fixed date, these depend on the current time at resolution.

### Date Resolution Changes

#### `date_resolution.py`

Add resolution logic for relative time keys. Use the existing `current.datetime` field from the date context (no new context sections needed):

```python
import re

RELATIVE_TIME_PATTERN = re.compile(
    r"^in_(\d+)_(minutes|hours|days)(?:_(\d+)_(minutes))?$"
)

def resolve_relative_time(key: str, date_context: dict) -> str:
    """Resolve 'in_30_minutes' → ISO datetime string."""
    match = RELATIVE_TIME_PATTERN.match(key)
    if not match:
        raise ValueError(f"Invalid relative time key: {key}")

    amount1 = int(match.group(1))
    unit1 = match.group(2)
    amount2 = int(match.group(3)) if match.group(3) else 0
    # unit2 is always minutes when present

    total_minutes = 0
    if unit1 == "days":
        total_minutes += amount1 * 1440
    elif unit1 == "hours":
        total_minutes += amount1 * 60
    else:
        total_minutes += amount1
    total_minutes += amount2

    current_time = datetime.fromisoformat(
        date_context["current"]["datetime"]
    )
    resolved = current_time + timedelta(minutes=total_minutes)
    return resolved.isoformat()
```

In `resolve_date_keys()`, try matching relative time keys before falling through to semantic key resolution:

```python
def resolve_date_keys(
    date_keys: list[str],
    date_context: dict,
) -> list[str]:
    resolved = []
    for key in date_keys:
        # Try relative time first
        match = RELATIVE_TIME_PATTERN.match(key)
        if match:
            resolved.append(resolve_relative_time(key, date_context))
            continue

        # Fall through to existing semantic key resolution
        resolved.append(resolve_semantic_key(key, date_context))

    return resolved
```

### `RelativeDateKeys` Constants Update

The existing `relative_date_keys.py` in jarvis-node-setup is auto-generated from the LLM proxy. Relative time keys are dynamic (not enumerable), so they should NOT be added to the constants file.

Instead, document the pattern in a new section of the date key API response:

```json
{
  "static_keys": ["today", "tomorrow", ...],
  "dynamic_patterns": [
    {
      "pattern": "in_{N}_minutes",
      "description": "Relative offset: N minutes from now",
      "examples": ["in_5_minutes", "in_30_minutes", "in_90_minutes"]
    },
    {
      "pattern": "in_{N}_hours",
      "description": "Relative offset: N hours from now",
      "examples": ["in_1_hours", "in_2_hours"]
    },
    {
      "pattern": "in_{N}_days",
      "description": "Relative offset: N days from now",
      "examples": ["in_1_days", "in_3_days", "in_7_days"]
    },
    {
      "pattern": "in_{N}_hours_{M}_minutes",
      "description": "Relative offset: N hours and M minutes from now",
      "examples": ["in_1_hours_30_minutes", "in_2_hours_15_minutes"]
    }
  ]
}
```

## Date Detector Changes

### `date_detector.py`

Add regex patterns to detect relative time expressions in voice commands:

```python
RELATIVE_TIME_PATTERNS = [
    r"in\s+(\d+)\s+minutes?",
    r"in\s+(\d+)\s+hours?",
    r"in\s+(\d+)\s+days?",
    r"in\s+(\d+)\s+hours?\s+(?:and\s+)?(\d+)\s+minutes?",
    r"in\s+an?\s+hour",          # "in an hour" → in_1_hours
    r"in\s+half\s+an?\s+hour",   # "in half an hour" → in_30_minutes
]
```

## Testing

### Unit Tests

```
1. test_resolve_relative_minutes
   - "in_30_minutes" → now + 30 min
2. test_resolve_relative_hours
   - "in_2_hours" → now + 120 min
3. test_resolve_relative_days
   - "in_3_days" → now + 3 days
4. test_resolve_relative_compound
   - "in_1_hours_30_minutes" → now + 90 min
5. test_resolve_mixed_keys
   - ["tomorrow", "in_30_minutes"] → both resolved correctly
6. test_resolve_invalid_relative_key
   - "in_abc_minutes" → raises ValueError
7. test_date_detector_detects_relative
   - "remind me in 30 minutes" → detects relative time pattern
8. test_date_detector_compound
   - "remind me in 1 hour and 30 minutes" → detected
9. test_date_detector_days
   - "remind me in 3 days" → detected
```

## Implementation Notes

- The LLM proxy adapter must also be trained to extract relative time keys (see companion PRD).
- Resolution uses `date_context["current"]["datetime"]` — no changes to `general_context.py` needed.
- Once both sides are updated, the `relative_minutes` fallback parameter on `set_reminder` can be removed entirely.

## Dependencies

- `jarvis-llm-proxy-api/prds/relative-time-resolution.md` — adapter training for relative time extraction

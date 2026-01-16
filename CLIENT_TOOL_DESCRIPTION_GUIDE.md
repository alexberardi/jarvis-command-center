# Client Tool Description Best Practices

## The Problem

Your LLM couldn't extract "miami" from "What is the weather in miami tomorrow?" because the tool description was too vague:

**Current (Bad)**:
```json
{
  "name": "open_weather_command",
  "description": "Gets the current weather or forecast for an optional city or optional date range",
  "parameters": {
    "properties": {
      "city": {
        "type": "string",
        "description": "The city of weather to get."  ❌ TOO VAGUE!
      }
    }
  }
}
```

The LLM sees "miami" but the description doesn't tell it to **extract** city names from the user's message.

## The Solution

Parameter descriptions should be **extraction instructions**, not just definitions.

### ✅ Good Parameter Descriptions

**Pattern**: `Extract [thing] from [where]. [Examples]. [Default/fallback behavior].`

#### Weather Tool (Fixed)

```json
{
  "name": "open_weather_command",
  "description": "Gets current weather or forecast for a specific location and date",
  "parameters": {
    "properties": {
      "city": {
        "type": "string",
        "description": "Extract the city name from the user's request (e.g., 'Miami', 'New York', 'San Francisco'). If no city is mentioned, leave empty to use user's default location."
      },
      "unit_system": {
        "type": "string",
        "description": "Temperature unit system: 'metric' (Celsius) or 'imperial' (Fahrenheit). Default to 'imperial' for US cities, 'metric' for others."
      },
      "datetimes": {
        "type": "string",
        "description": "For relative dates (tomorrow, next week, etc.), first use the resolve_relative_date tool to convert to ISO datetime, then pass the result here. For current weather, leave empty."
      }
    }
  }
}
```

#### Calendar Tool (Fixed)

```json
{
  "name": "read_calendar_command",
  "description": "Reads calendar events from configured calendar service",
  "parameters": {
    "properties": {
      "datetimes": {
        "type": "string",
        "description": "For relative dates like 'tomorrow' or 'next week', first use the resolve_relative_date tool to get the ISO datetime(s), then pass them here as an array. Example: ['2025-11-08T05:00:00Z']"
      }
    }
  }
}
```

#### Sports Schedule Tool (Fixed)

```json
{
  "name": "sports_schedule_command",
  "description": "Get upcoming sports schedules and game times (future events only)",
  "parameters": {
    "properties": {
      "team_name": {
        "type": "string",
        "description": "Extract the EXACT team name as spoken by the user, including city/state if mentioned. Examples: 'Giants' (if user just said Giants), 'New York Giants' (if user specified), 'Carolina Panthers', 'Ohio State Buckeyes'. Keep the full name as spoken."
      },
      "datetimes": {
        "type": "string",
        "description": "For relative dates (tomorrow, this weekend, next week), first call resolve_relative_date tool to convert to ISO datetime(s), then pass the result(s) here. For 'when do they play next', use today's date from resolve_relative_date tool."
      }
    },
    "required": ["team_name", "datetimes"]
  }
}
```

## Key Principles

### 1. **Be Explicit About Extraction**

❌ Bad: "The city name"
✅ Good: "Extract the city name from the user's request"

### 2. **Provide Examples**

❌ Bad: "The team name"
✅ Good: "The team name (e.g., 'Giants', 'New York Giants', 'Carolina Panthers')"

### 3. **Explain Default Behavior**

❌ Bad: "Optional city parameter"
✅ Good: "If no city is mentioned, leave empty to use user's default location"

### 4. **Reference Other Tools When Needed**

❌ Bad: "Array of ISO datetime strings for dates"
✅ Good: "For relative dates like 'tomorrow', first use resolve_relative_date tool to convert, then pass result here"

### 5. **Don't Say "Optional" in Description**

❌ Bad: "Optional city parameter (not required)"
✅ Good: Just don't mark it as required, and explain default: "If not mentioned, uses default location"

### 6. **Be Specific About Format**

❌ Bad: "Temperature units"
✅ Good: "Temperature unit: 'metric' (Celsius) or 'imperial' (Fahrenheit)"

## Common Patterns

### Location Parameters

```json
{
  "city": {
    "type": "string",
    "description": "Extract the city name from user's request (e.g., 'Miami', 'London', 'Tokyo'). If not mentioned, leave empty for default location."
  }
}
```

### Date/Time Parameters

```json
{
  "datetimes": {
    "type": "string",
    "description": "For relative dates ('tomorrow', 'next week'), first use resolve_relative_date tool to convert to ISO datetime, then pass result here. For current/now, leave empty."
  }
}
```

### Name/Identifier Parameters

```json
{
  "team_name": {
    "type": "string",
    "description": "Extract the complete team name exactly as spoken by user, including any city/state mentioned. Examples: 'Giants', 'New York Giants', 'Buckeyes', 'Ohio State Buckeyes'."
  }
}
```

### Enum/Choice Parameters

```json
{
  "operation": {
    "type": "string",
    "description": "Math operation to perform. Extract from user's words: 'add' (plus, sum), 'subtract' (minus, take away), 'multiply' (times), 'divide' (divided by). Must be one of: 'add', 'subtract', 'multiply', 'divide'."
  }
}
```

### Numeric Parameters

```json
{
  "value": {
    "type": "number",
    "description": "Extract the numeric value from user's request. Example: 'convert 5 miles' = 5. If no number mentioned (e.g., 'how many cups in a gallon'), default to 1."
  }
}
```

## Before & After Examples

### Example 1: Weather Command

**Before**:
```
User: "What's the weather in miami tomorrow?"
LLM: "I need to know the city..." ❌ FAILS
```

**After** (with better descriptions):
```
User: "What's the weather in miami tomorrow?"
LLM: {
  "message": "Let me check the weather...",
  "tool_calls": [
    {"name": "resolve_relative_date", "arguments": {"relative_term": "tomorrow"}},
  ]
}
Then: {
  "tool_calls": [
    {"name": "open_weather_command", "arguments": {
      "city": "Miami",  ✅ EXTRACTED!
      "datetimes": "2025-11-08T05:00:00Z"
    }}
  ]
}
```

### Example 2: Calendar Command

**Before**:
```
User: "What's on my calendar next week?"
LLM: {"tool_calls": [{"name": "read_calendar_command", "arguments": {"datetimes": "next week"}}]} ❌ WRONG FORMAT
```

**After**:
```
User: "What's on my calendar next week?"
LLM: {
  "tool_calls": [
    {"name": "resolve_relative_date", "arguments": {"relative_term": "next week"}}
  ]
}
Then: {
  "tool_calls": [
    {"name": "read_calendar_command", "arguments": {
      "datetimes": ["2025-11-09T05:00:00Z", "2025-11-10T05:00:00Z", ...]  ✅ CORRECT!
    }}
  ]
}
```

## How to Update Your Client Tools

Your client tools are stored in a database. You'll need to update them there. Here's a script template:

```python
# update_tool_descriptions.py

# Connect to your database
# For each tool, update the description and parameter descriptions

tools_to_update = {
    "open_weather_command": {
        "description": "Gets current weather or forecast for a specific location and date",
        "parameters": {
            "city": "Extract the city name from the user's request (e.g., 'Miami', 'New York'). If no city mentioned, leave empty for default location.",
            "unit_system": "Temperature unit: 'metric' (Celsius) or 'imperial' (Fahrenheit). Default to 'imperial' for US cities.",
            "datetimes": "For relative dates, first use resolve_relative_date tool to convert to ISO datetime. For current weather, leave empty."
        }
    },
    "read_calendar_command": {
        "description": "Reads calendar events from configured calendar service",
        "parameters": {
            "datetimes": "For relative dates like 'tomorrow', first use resolve_relative_date tool to get ISO datetime(s), then pass as array."
        }
    },
    "sports_schedule_command": {
        "description": "Get upcoming sports schedules and game times (future events only)",
        "parameters": {
            "team_name": "Extract the complete team name as spoken, including city/state if mentioned. Examples: 'Giants', 'New York Giants', 'Ohio State Buckeyes'.",
            "datetimes": "For relative dates, first call resolve_relative_date to convert, then pass result. For 'when do they play next', use today."
        }
    },
    "sports_score_command": {
        "description": "Get scores for past/completed games",
        "parameters": {
            "team_name": "Extract the complete team name as spoken, including city/state if mentioned.",
            "datetimes": "For relative dates, first use resolve_relative_date tool to convert to ISO datetime array."
        }
    }
}

# Update each tool in database
```

## Testing Tool Descriptions

After updating, test with these prompts:

1. ✅ "What's the weather in miami tomorrow?" - Should extract "Miami" and use resolve_relative_date
2. ✅ "When do the Panthers play next?" - Should extract "Panthers" and figure out date
3. ✅ "What's on my calendar this weekend?" - Should use resolve_relative_date first
4. ✅ "Convert 5 miles to kilometers" - Should extract 5, miles, and kilometers
5. ✅ "What's 25 + 37?" - Should extract 25, 37, and "add" operation

## Summary

**The Golden Rule**: Write parameter descriptions as if you're instructing a junior developer on what to extract from the user's message, not just defining what the parameter is.

❌ "The city name"
✅ "Extract the city name from the user's request. Examples: 'Miami', 'New York'. If not mentioned, leave empty."


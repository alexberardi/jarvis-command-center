# Parameter Extraction Fix

## The Problem

LLM wasn't extracting "Miami" from "What's the weather in miami tomorrow?"

**Debug Output**:
```
User: "What is the weather in miami tomorrow?"
LLM: "I need to know the city you are referring to for weather information. Would you like me to look up Miami or another city?"
```

The user **clearly said "Miami"**, but the LLM asked for clarification anyway! ❌

## Root Causes

### 1. **Vague Client Tool Descriptions**

The tool descriptions in your database are too vague:

```json
{
  "city": {
    "type": "string",
    "description": "The city of weather to get."  ❌ DOESN'T TELL LLM TO EXTRACT
  }
}
```

This just defines what the parameter IS, not how to GET it from the user's message.

### 2. **Outdated References**

Some client tools still mention "DateContext" which we removed:
- "convert them to actual ISO datetime values using the DateContext"

### 3. **No Clear Extraction Guidance**

The system prompt didn't explicitly tell the LLM to extract parameters from the user's message.

## The Fix

### ✅ Part 1: Enhanced System Prompt

Added explicit parameter extraction guidelines to `app/core/models/jarvis_tool_model.py`:

```python
## Guidelines for Using Tools

1. **Extract parameters directly from the user's message**
   - If user says "weather in Miami", extract city="Miami"
   - If user says "turn on bedroom light", extract room="bedroom"
   - If user says "tomorrow", use resolve_relative_date tool first
   - Don't ask for information that's already in the message!

4. **Ask for clarification ONLY when truly ambiguous**
   - Ambiguous locations: user says "the light" but has lights in multiple rooms
   - Missing required parameters: user says "play music" but which service?
   - Multiple possible interpretations: user says "Giants" (NY Giants or SF Giants?)
   - DON'T ask if the user already provided the information!
```

Also added a clear example:

```
### Example 4: DON'T Ask When Info is Provided
User: "What's the weather in Seattle?"
WRONG: {"message": "Which city would you like weather for?"}  ❌ User already said Seattle!
CORRECT: {"message": "Checking Seattle's weather.", "tool_calls": [{"name": "open_weather_command", "arguments": {"city": "Seattle"}}]}  ✅
```

### ✅ Part 2: Client Tool Description Guide

Created `CLIENT_TOOL_DESCRIPTION_GUIDE.md` with best practices for writing tool descriptions.

**Key principle**: Write descriptions as **extraction instructions**, not just definitions.

**Instead of**:
```json
{
  "city": {
    "description": "The city name"  ❌
  }
}
```

**Use**:
```json
{
  "city": {
    "description": "Extract the city name from the user's request (e.g., 'Miami', 'New York'). If no city mentioned, leave empty for default location."  ✅
  }
}
```

## What You Need To Do

### 1. **Update Client Tool Descriptions in Your Database**

Your client tools are stored in a database (not in this codebase). You need to update them with better descriptions.

See `CLIENT_TOOL_DESCRIPTION_GUIDE.md` for:
- Complete examples for all tools
- Before/after comparisons
- Best practices
- Testing checklist

### Priority Updates:

**open_weather_command**:
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
      "datetimes": {
        "type": "string",
        "description": "For relative dates (tomorrow, next week, etc.), first use the resolve_relative_date tool to convert to ISO datetime, then pass the result here. For current weather, leave empty."
      }
    }
  }
}
```

**sports_schedule_command** & **sports_score_command**:
```json
{
  "parameters": {
    "properties": {
      "team_name": {
        "type": "string",
        "description": "Extract the complete team name as spoken, including city/state if mentioned. Examples: 'Giants', 'New York Giants', 'Ohio State Buckeyes'."
      },
      "datetimes": {
        "type": "string",
        "description": "For relative dates, first call resolve_relative_date to convert, then pass result."
      }
    }
  }
}
```

**read_calendar_command**:
```json
{
  "parameters": {
    "properties": {
      "datetimes": {
        "type": "string",
        "description": "For relative dates like 'tomorrow' or 'next week', first use resolve_relative_date tool to get ISO datetime(s), then pass as array."
      }
    }
  }
}
```

### 2. **Remove "DateContext" References**

Update any mentions of "using the DateContext" to say "use the resolve_relative_date tool".

### 3. **Test After Updating**

Try these prompts to verify:
```
✅ "What's the weather in Miami tomorrow?"
✅ "When do the Panthers play next?"
✅ "What's on my calendar this weekend?"
✅ "Turn on the bedroom light"
```

All should extract parameters correctly without asking for clarification!

## Files Modified in This Codebase

1. **`app/core/models/jarvis_tool_model.py`**
   - Added parameter extraction guidelines
   - Updated examples to show correct extraction
   - Added "DON'T ask when info is provided" example

2. **`CLIENT_TOOL_DESCRIPTION_GUIDE.md`** (NEW)
   - Complete guide for writing better tool descriptions
   - Before/after examples for each tool
   - Testing checklist

3. **`PARAMETER_EXTRACTION_FIX.md`** (THIS FILE)
   - Summary of the issue and fix

## Expected Behavior After Fix

**Before**:
```
User: "What's the weather in miami tomorrow?"
LLM: "I need to know the city..." ❌ FAILS
```

**After** (with updated client tool descriptions):
```
User: "What's the weather in miami tomorrow?"
LLM: {
  "message": "Let me check the weather for Miami tomorrow.",
  "tool_calls": [
    {"name": "resolve_relative_date", "arguments": {"relative_term": "tomorrow"}}
  ]
}
→ [Tool returns date]
LLM: {
  "message": "Getting the forecast...",
  "tool_calls": [
    {"name": "open_weather_command", "arguments": {
      "city": "Miami",  ✅ EXTRACTED!
      "datetimes": "2025-11-08T05:00:00Z"
    }}
  ]
}
→ [Client executes weather command]
LLM: {
  "message": "Tomorrow in Miami will be sunny with a high of 78°F.",
  "tool_calls": null
}
```

## Summary

**System prompt improvements** (✅ Done in this PR):
- Added explicit parameter extraction guidelines
- Added clear examples of correct vs incorrect extraction
- Emphasized not asking when info is already provided

**Client tool updates** (⏳ You need to do):
- Update tool descriptions in your database
- Change from definitions to extraction instructions
- Remove "DateContext" references
- Add examples in descriptions

After you update the client tools, your LLM should correctly extract parameters! 🎉


# JarvisToolModel Usage Guide

## Overview

`JarvisToolModel` is a modern model implementation designed specifically for tool-based conversations. It uses prompt engineering to teach generic open source models (Llama, Mistral, etc.) how to use tools through JSON-formatted responses.

## Key Features

- ✅ **Natural language final responses** - Returns conversational text that can be spoken
- ✅ **Comprehensive system prompts** - Clear instructions with examples
- ✅ **Automatic server tool execution** - Transparent to the client
- ✅ **Clarifying questions** - Built-in support for validation
- ✅ **Multi-turn conversations** - Handles tool execution loops
- ✅ **No fine-tuning required** - Works with any good instruct model

## Setup

### 1. Set Environment Variable

```bash
export JARVIS_MODEL_INTERFACE="JarvisToolModel"
```

### 2. Restart Server

```bash
./run-dev.sh
```

The model will be automatically discovered and loaded!

## How It Works

### System Prompt Structure

The model builds comprehensive system prompts that include:

1. **Role and context** - "You are Jarvis, an intelligent voice assistant..."
2. **Environment info** - Room, user, response style
3. **Date/time context** - Current date, relative dates, etc.
4. **JSON format instructions** - Exact format for tool calls
5. **Available tools** - Detailed tool descriptions
6. **Guidelines** - How to use tools effectively
7. **Examples** - Concrete examples of tool usage

### Response Flow

```
User: "What time is it?"
  ↓
LLM: {"message": "Let me check the time.", "tool_calls": [{"name": "get_current_date_time", ...}]}
  ↓
[Server executes tool automatically]
  ↓
LLM: {"message": "It's 2:30 PM.", "tool_calls": null}
  ↓
User hears: "It's 2:30 PM." ✅
```

### Final Messages Are Natural

The key design goal: **The final "message" field will be spoken to the user**, so it's always natural and conversational:

- ✅ "It's 72 degrees in the bedroom."
- ✅ "The bedroom light is now on."
- ✅ "I've set the temperature to 68 degrees."
- ❌ "Tool executed successfully: {result: true}"
- ❌ "Command completed with status 200"

## System Prompt Examples

### Tool Usage Example

```
User: "What time is it?"
You: {"message": "Let me check the current time for you.", "tool_calls": [{"name": "get_current_date_time", "arguments": {}}]}
[Tool returns time data]
You: {"message": "It's 2:30 PM.", "tool_calls": null}
```

### Clarification Example

```
User: "Turn on the light"
You: {"message": "I need to know which room.", "tool_calls": [{"name": "request_validation", "arguments": {"question": "Which room's light would you like me to turn on?", "parameter_name": "room", "options": ["bedroom", "living room", "kitchen"]}}]}
```

### Multiple Tools Example

```
User: "What's the temperature in the bedroom?"
You: {"message": "Let me check the temperature.", "tool_calls": [{"name": "get_temperature", "arguments": {"room": "bedroom"}}]}
[Tool returns temp]
You: {"message": "It's 72 degrees in the bedroom.", "tool_calls": null}
```

## Configuration

### Voice Mode

The model adapts to the user's preferred voice mode (from node context):

- **Brief**: Short, concise responses
- **Conversational**: More natural, friendly tone
- **Detailed**: Thorough explanations

Example with `voice_mode: "brief"`:
```json
{"message": "It's 2:30 PM.", "tool_calls": null}
```

Example with `voice_mode: "conversational"`:
```json
{"message": "The current time is 2:30 PM on Friday afternoon.", "tool_calls": null}
```

### Temperature Settings

For best JSON consistency:

```bash
# In LLM proxy or model settings
temperature: 0.1  # Lower = more consistent
top_p: 0.9
```

### Model Recommendations

Works best with:
- **Llama 3.2 3B** or **Llama 3.2 8B** ✅ Great instruction following
- **Mistral 7B+** ✅ Strong JSON output
- **Mixtral 8x7B** ✅ Excellent for complex tool chains
- **Qwen 2.5** ✅ Good JSON consistency

## Usage with Different APIs

### Standard API Usage

```python
# Conversation start
response = requests.post("/api/v0/conversation/start", json={
    "conversation_id": "conv-123",
    "client_tools": [...]
})

# Send command
response = requests.post("/api/v0/voice/command", json={
    "voice_command": "Turn on the bedroom light",
    "conversation_id": "conv-123"
})

# If stop_reason == "tool_calls", execute and continue
if response["stop_reason"] == "tool_calls":
    # Execute client tools
    results = execute_tools(response["tool_calls"])
    
    # Continue conversation
    response = requests.post("/api/v0/voice/command/continue", json={
        "conversation_id": "conv-123",
        "tool_results": results
    })
```

### ModelService API

If using the model directly through `ModelService`:

```python
from app.core.model_service import ModelService

model_service = ModelService("JarvisToolModel")

# Warmup (tool-based)
await model_service.warmup_conversation_with_tools(
    node_context={"room": "bedroom", "user": "Alex", "voice_mode": "brief"},
    conversation_id="conv-123",
    timezone="America/New_York",
    client_tools=[...]
)

# Process command
result = await model_service.process_voice_command(
    voice_command="What time is it?",
    conversation_id="conv-123"
)

# result["s"] = True
# result["p"]["message"] = "It's 2:30 PM."
```

## Prompt Customization

### Adding Few-Shot Examples

If your model struggles with JSON format, edit `jarvis_tool_model.py` and add more examples to `_build_system_prompt`:

```python
## Response Examples

### Example 1: ...
### Example 2: ...
### Example 3: Your new example
User: "Set alarm for 7am"
You: {"message": "I'll set an alarm for 7 AM.", "tool_calls": [{"name": "set_alarm", "arguments": {"time": "07:00"}}]}
```

### Adjusting Voice Mode Instructions

In `_build_system_prompt`, modify the guidelines:

```python
# For more formal responses
f"Be professional and {voice_mode} in your responses"

# For more casual responses  
f"Be friendly, casual, and {voice_mode} in your responses"
```

## Troubleshooting

### Model doesn't output JSON

**Check:**
1. Model capabilities (needs instruction following)
2. Temperature (should be low, 0.1-0.3)
3. System prompt reaching model correctly

**Solutions:**
- Add more JSON examples to prompt
- Use a better instruct model
- Enable JSON mode in vLLM if available
- Check LLM proxy logs for actual prompt

### JSON is malformed

**Check:**
- Model logs for actual output
- Temperature settings
- Context length (prompt might be truncated)

**Solutions:**
- Lower temperature to 0.0
- Simplify system prompt
- Use model with better JSON following
- Add JSON validation examples

### Responses are too technical

**Fix:** Adjust the guidelines in system prompt:

```python
# Current
"Be conversational but {voice_mode}"

# More natural
"Respond as if speaking to a friend. Be {voice_mode} but warm and natural. Your response will be spoken aloud, so make it sound conversational."
```

### Tools not being called

**Check:**
1. Tools are properly formatted in prompt
2. Tool descriptions are clear
3. Model understands when to use tools

**Solutions:**
- Make tool descriptions more specific
- Add negative examples (when NOT to use tools)
- Emphasize tool usage in system prompt

### Wrong tool arguments

**Check:**
- Parameter descriptions in tool definitions
- Examples in system prompt
- Model's understanding of requirements

**Solutions:**
- Make parameter descriptions more explicit
- Add examples with correct arguments
- Use enum values to constrain options

## Capabilities

```python
model.get_capabilities()
```

Returns:
```python
{
    "supports_single_shot": False,
    "supports_streaming": False,
    "supports_tools": True,
    "supports_multi_turn": True,
    "supports_clarification": True,
    "max_context_length": None,
    "supported_languages": ["en"],
    "custom_features": {
        "tool_based": True,
        "prompt_engineering": True,
        "json_parsing": True
    }
}
```

## Best Practices

### 1. Tool Descriptions

Be specific and actionable:

```python
# ❌ Bad
"description": "Control light"

# ✅ Good
"description": "Turn on or adjust brightness of a light in a specific room. Only use when user explicitly mentions lights, lamps, or brightness."
```

### 2. Response Style

Match the voice mode:

```python
# Brief
"It's 72°F."

# Conversational  
"The temperature in the bedroom is 72 degrees."

# Detailed
"I checked the bedroom temperature for you. It's currently 72 degrees Fahrenheit, which is quite comfortable."
```

### 3. Error Handling

Always provide helpful messages:

```python
# ❌ Bad
"Error: tool_not_found"

# ✅ Good
"I apologize, but I don't have access to that control. Could you try a different command?"
```

### 4. Clarifications

Ask specific questions:

```python
# ❌ Bad
"Which one?"

# ✅ Good  
"Which room's light would you like me to control? You can say bedroom, living room, or kitchen."
```

## Model Support

The tool-based flow is now the only supported runtime path. Legacy command/parameter inference models have been removed.

## Future Enhancements

Planned improvements:
- Streaming support for real-time responses
- Multiple language support
- Voice tone/personality customization
- Context-aware response length
- Automatic summarization for long tool results

## Support

For issues or questions:
1. Check logs: `tail -f logs/jarvis.log | grep "🤖\|🎯\|📥"`
2. Test JSON parsing: `python examples/test_tool_parsing.py`
3. Verify model loading: Check startup logs for "Found model: JarvisToolModel"

## Summary

`JarvisToolModel` is the **recommended model for new deployments** because:

- ✅ Natural, conversational responses
- ✅ Works with any good instruct model
- ✅ No fine-tuning required
- ✅ Built-in clarification support
- ✅ Automatic tool execution
- ✅ Easy to customize and debug

Just set `JARVIS_MODEL_INTERFACE=JarvisToolModel` and you're ready to go!


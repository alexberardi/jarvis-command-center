# LLM Proxy API Requirements for Tool Support (Prompt-Based)

This document outlines the minimal changes needed in the LLM Proxy API to support the tool-based architecture using **prompt-based tool calling** for generic open source models.

## Overview

For local models (Llama, Mistral, etc.) running on vLLM/llama.cpp that don't have native tool calling support, we use a **prompt-based approach**:

1. **Tools are included in the system prompt** as text
2. **LLM outputs JSON** with tool calls
3. **This project parses the JSON** to extract tool calls
4. **LLM proxy just passes messages through** - no special tool handling needed

## Why This Approach?

- ✅ Works with any generic open source model
- ✅ No fine-tuning required
- ✅ Minimal changes to LLM proxy
- ✅ Simple to implement and debug
- ✅ Compatible with vLLM/llama.cpp

## Required Changes

### Minimal/No Changes Needed! 🎉

The LLM proxy can stay almost exactly as is. The only requirement is that it:

1. **Accepts messages** (including system messages with tool definitions)
2. **Returns raw LLM output** as message content
3. **That's it!**

### Current Request Format (Should Already Work)

```json
{
  "model": "jarvis-llm",
  "temperature": 0,
  "messages": [
    {
      "role": "system",
      "content": "You are Jarvis... [includes tool definitions and JSON format instructions]"
    },
    {
      "role": "user",
      "content": "Turn on the bedroom light"
    }
  ],
  "conversation_id": "optional-conv-id"
}
```

### Current Response Format (OpenAI-Compatible)

The LLM proxy should return **OpenAI-compatible chat completion format**:

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "your-model-name",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"message\": \"I'll turn on the light.\", \"tool_calls\": [{\"name\": \"turn_on_light\", \"arguments\": {\"room\": \"bedroom\"}}]}"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  }
}
```

The key parts:
- `choices[0].message.content` contains the LLM's raw output (JSON string)
- `choices[0].finish_reason` indicates completion ("stop" for normal completion)
- We parse the JSON content on this side

## Expected LLM Output Format

The model will be prompted to output JSON in this format:

### With Tool Calls

```json
{
  "message": "I'll help you with that.",
  "tool_calls": [
    {
      "name": "turn_on_light",
      "arguments": {
        "room": "bedroom",
        "brightness": 50
      }
    }
  ]
}
```

### Without Tool Calls (Final Response)

```json
{
  "message": "The bedroom light is now on.",
  "tool_calls": null
}
```

## System Prompt Format

The Jarvis Command Center automatically builds system prompts like this:

```
You are Jarvis, a voice-controlled assistant...

IMPORTANT - Response Format:
You MUST respond with valid JSON in this exact format:

When you need to use tools:
{
  "message": "Your response to the user",
  "tool_calls": [
    {"name": "tool_name", "arguments": {"param_name": "param_value"}}
  ]
}

When you don't need tools (final response):
{
  "message": "Your response to the user",
  "tool_calls": null
}

Available Tools:

Tool: turn_on_light
Description: Turn on a light in the specified room
Parameters:
  - room (string) [REQUIRED]: The room where the light is located
  - brightness (number): Brightness level from 0-100

Tool: get_current_date_time
Description: Get current date and time context
Parameters:
  - timezone (string): Optional timezone string

...
```

## Message History with Tools

When tools are executed, results are added to the conversation:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are Jarvis... [with tools]"
    },
    {
      "role": "user",
      "content": "Turn on the bedroom light"
    },
    {
      "role": "assistant",
      "content": "{\"message\": \"I'll turn on the light.\", \"tool_calls\": [{\"name\": \"turn_on_light\", \"arguments\": {\"room\": \"bedroom\"}}]}"
    },
    {
      "role": "tool",
      "tool_call_id": "call_abc123",
      "content": "{\"success\": true, \"message\": \"Light turned on\"}"
    }
  ]
}
```

**Note:** The `role: "tool"` messages might be new. If your LLM proxy doesn't support this role:

### Option A: Use "user" role for tool results

```json
{
  "role": "user",
  "content": "Tool result for turn_on_light: {\"success\": true}"
}
```

### Option B: Use "system" role for tool results

```json
{
  "role": "system",
  "content": "[Tool Result] turn_on_light: {\"success\": true}"
}
```

We can configure which approach to use based on what works best with your setup.

## Implementation Checklist

### Required (Minimal)
- [x] Accept messages with `role: "system"` (should already work)
- [x] Accept messages with `role: "user"` (should already work)
- [x] Accept messages with `role: "assistant"` (should already work)
- [ ] Accept messages with `role: "tool"` (or use alternative - see above)
- [x] Return raw LLM output in `message.content` (should already work)

### Optional (Improvements)
- [ ] Add JSON validation/fixing if LLM outputs malformed JSON
- [ ] Add retry logic for non-JSON responses
- [ ] Log when LLM doesn't follow JSON format
- [ ] Support for longer tool definitions (context window)

## Testing

### Test 1: Basic Message Passing

```bash
curl -X POST http://your-llm-proxy/api/v0/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "Respond with: {\"message\": \"test\", \"tool_calls\": null}"},
      {"role": "user", "content": "hi"}
    ]
  }'
```

Expected: LLM outputs JSON with message field

### Test 2: Tool Role Messages

```bash
curl -X POST http://your-llm-proxy/api/v0/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant"},
      {"role": "user", "content": "hi"},
      {"role": "assistant", "content": "hello"},
      {"role": "tool", "content": "tool result"}
    ]
  }'
```

Expected: Should handle `role: "tool"` or error clearly

## Model Considerations

### For Generic Open Source Models

Your model needs to:
1. **Follow instructions** in the system prompt
2. **Output valid JSON** consistently
3. **Understand tool concepts** from examples

**Tips for better results:**
- Use models with strong instruction following (Llama 3+, Mistral 7B+, etc.)
- Consider using JSON mode if available in vLLM
- Add few-shot examples in system prompt if needed
- Adjust temperature (lower = more consistent JSON)

### Few-Shot Examples (Optional)

If your model struggles with JSON format, you can add examples to the system prompt:

```
Example 1:
User: "What time is it?"
Assistant: {"message": "Let me check the time.", "tool_calls": [{"name": "get_current_date_time", "arguments": {}}]}
[Tool Result]: {"current": {"time": "2:30 PM"}}
Assistant: {"message": "It's 2:30 PM.", "tool_calls": null}

Example 2:
User: "Turn on the bedroom light"
Assistant: {"message": "I'll turn on the bedroom light.", "tool_calls": [{"name": "turn_on_light", "arguments": {"room": "bedroom"}}]}
```

## Comparison: Prompt-Based vs Native Tool Calling

| Feature | Prompt-Based (This) | Native (OpenAI-style) |
|---------|-------------------|---------------------|
| Model Support | Any instruct model | Fine-tuned models only |
| LLM Proxy Changes | Minimal/None | Significant |
| Parsing | This project | LLM proxy |
| Reliability | Depends on model | Very high |
| Flexibility | High | Medium |
| Setup Time | Minutes | Hours/Days |

## Configuration

You may want to add these environment variables:

```bash
# JSON mode for vLLM (if supported)
VLLM_RESPONSE_FORMAT=json

# Temperature for more consistent JSON
JARVIS_LLM_TEMPERATURE=0.1

# Max retries for malformed JSON
JARVIS_MAX_JSON_RETRIES=3
```

## Troubleshooting

### Issue: LLM doesn't output JSON

**Solutions:**
1. Emphasize JSON format in system prompt more strongly
2. Add few-shot examples
3. Use a model with better instruction following
4. Try JSON mode in vLLM (`--guided-decoding`)

### Issue: JSON is malformed

**Solutions:**
1. Lower temperature to 0.0-0.2
2. Add JSON validation examples
3. Use regex to clean common issues
4. Implement fallback parsing in tool_call_parser.py

### Issue: Tool role not supported

**Solution:**
- Use "user" role with prefix: `"[Tool Result] tool_name: {output}"`
- Update model_service.py to format accordingly

### Issue: Model calls wrong tools

**Solutions:**
1. Make tool descriptions more specific
2. Add negative examples in prompt
3. Limit tools per conversation
4. Add tool selection examples

## Migration from Current System

If you have an existing LLM proxy:

1. **No changes needed** if it already:
   - Accepts multi-turn conversations
   - Returns raw LLM output
   - Supports system messages

2. **Small change needed** if:
   - It doesn't support `role: "tool"` → Use alternative role

3. **Test first** with simple messages to verify JSON output

## Questions / Clarifications

1. **Does your LLM proxy support `role: "tool"` messages?**
   - If not, we'll use "user" role with a prefix

2. **What model are you using?**
   - This affects how we write the system prompt

3. **Does vLLM/llama.cpp have JSON mode enabled?**
   - This could improve consistency

4. **Any token/context limits we should know about?**
   - Tool definitions can be verbose

## Next Steps

1. **Test current LLM proxy** with JSON-formatted system prompts
2. **Verify message history** with tool role or alternatives
3. **Tune system prompt** based on model behavior
4. **Add few-shot examples** if needed
5. **Iterate on JSON parsing** for edge cases

## Summary

**TL;DR:** Your LLM proxy probably doesn't need any changes! Just make sure it:
- Passes through system messages with tool definitions
- Returns raw LLM output
- Handles multi-turn conversations

We handle all the tool parsing and execution on this side. 🎉

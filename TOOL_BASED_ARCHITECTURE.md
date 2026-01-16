# Tool-Based Architecture Implementation

This document describes the new tool-based architecture for the Jarvis Command Center and provides examples of how to use it.

## Overview

The system has been refactored to support an OpenAI-compatible tool-based conversation flow. Instead of single-shot command inference, the LLM can now:

1. Call server-side tools (like date/time retrieval)
2. Call client-side tools (custom commands defined by the client)
3. Request validation/clarification from users
4. Maintain multi-turn conversations with tool execution loops

## Architecture Components

### Server-Side Components

#### 1. Tool Registry (`app/core/tool_registry.py`)
- Manages server-side tool definitions in OpenAI format
- Currently includes:
  - `get_current_date_time`: Returns comprehensive date/time context
  - `request_validation`: Stub for validation requests (future)

#### 2. Tool Executor (`app/core/tool_executor.py`)
- Executes server-side tools
- Separates server vs client tool calls
- Handles tool execution loops transparently

#### 3. Enhanced Conversation Cache (`app/core/conversation_cache.py`)
- Stores combined tool lists (server + client)
- Maintains full message history including tool calls/results
- New methods: `get_tools()`, `add_messages()`, `update_messages()`

#### 4. Updated Model Service (`app/core/model_service.py`)
- New methods:
  - `warmup_conversation_with_tools()`
  - `process_voice_command_with_tools()`
  - `continue_conversation_with_tool_results()`
- Implements automatic server-side tool execution loop
- Returns client tool calls for execution

### API Endpoints

#### 1. `POST /api/v0/conversation/start`
**Enhanced to support tools:**

```json
{
  "conversation_id": "uuid-here",
  "node_context": {
    "timezone": "America/New_York"
  },
  "available_commands": [...],  // Legacy, optional
  "client_tools": [              // NEW: Tool definitions
    {
      "type": "function",
      "function": {
        "name": "turn_on_light",
        "description": "Turn on a light in the specified room",
        "parameters": {
          "type": "object",
          "properties": {
            "room": {
              "type": "string",
              "description": "The room where the light is located"
            },
            "brightness": {
              "type": "number",
              "description": "Brightness level from 0-100"
            }
          },
          "required": ["room"]
        }
      }
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "conversation_id": "uuid-here"
}
```

#### 2. `POST /api/v0/voice/command`
**Now returns different response based on conversation type:**

**Request:**
```json
{
  "voice_command": "Turn on the bedroom light at 50% brightness",
  "conversation_id": "uuid-here"
}
```

**Response (Tool-Based):**
```json
{
  "commands": [],
  "request_information": {
    "voice_command": "Turn on the bedroom light at 50% brightness",
    "conversation_id": "uuid-here"
  },
  "stop_reason": "tool_calls",
  "assistant_message": "I'll turn on the bedroom light for you.",
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "turn_on_light",
        "arguments": "{\"room\": \"bedroom\", \"brightness\": 50}"
      }
    }
  ],
  "validation_request": null
}
```

**Stop Reasons:**
- `complete`: Conversation finished, no action needed
- `tool_calls`: Client must execute tools and call continue endpoint
- `validation_required`: User clarification needed

#### 3. `POST /api/v0/voice/command/continue` (NEW)
**Submit tool execution results:**

**Request:**
```json
{
  "conversation_id": "uuid-here",
  "tool_results": [
    {
      "tool_call_id": "call_abc123",
      "output": {
        "success": true,
        "message": "Bedroom light turned on at 50% brightness"
      }
    }
  ]
}
```

**Response:**
Same format as `/voice/command` response. May return more tool calls or complete.

## Client Implementation Guide

### Tool Execution Loop

```python
import requests
import json

API_KEY = "your-api-key"
BASE_URL = "http://localhost:8000/api/v0"

headers = {"X-API-Key": API_KEY}

# 1. Start conversation with tools
client_tools = [
    {
        "type": "function",
        "function": {
            "name": "turn_on_light",
            "description": "Turn on a light",
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {"type": "string"},
                    "brightness": {"type": "number"}
                },
                "required": ["room"]
            }
        }
    }
]

start_response = requests.post(
    f"{BASE_URL}/conversation/start",
    headers=headers,
    json={
        "conversation_id": "conv-123",
        "node_context": {"timezone": "America/New_York"},
        "client_tools": client_tools
    }
)

# 2. Send voice command
def execute_tool(tool_name, arguments):
    """Execute a client-side tool."""
    if tool_name == "turn_on_light":
        # Your implementation here
        return {"success": True, "message": f"Light turned on"}
    # Add more tools...
    return {"error": "Unknown tool"}

def process_command(command_text, conversation_id):
    """Process a command with automatic tool loop."""
    response = requests.post(
        f"{BASE_URL}/voice/command",
        headers=headers,
        json={
            "voice_command": command_text,
            "conversation_id": conversation_id
        }
    ).json()
    
    # Loop until complete
    while response.get("stop_reason") == "tool_calls":
        print(f"Assistant: {response.get('assistant_message')}")
        
        # Execute tools
        tool_results = []
        for tool_call in response["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            
            print(f"Executing tool: {tool_name}({arguments})")
            result = execute_tool(tool_name, arguments)
            
            tool_results.append({
                "tool_call_id": tool_call["id"],
                "output": result
            })
        
        # Continue conversation with results
        response = requests.post(
            f"{BASE_URL}/voice/command/continue",
            headers=headers,
            json={
                "conversation_id": conversation_id,
                "tool_results": tool_results
            }
        ).json()
    
    # Handle validation requests
    if response.get("stop_reason") == "validation_required":
        validation = response["validation_request"]
        print(f"Clarification needed: {validation['question']}")
        if validation.get("options"):
            print(f"Options: {', '.join(validation['options'])}")
        
        # Get user input and continue...
        user_response = input("Your choice: ")
        # Send user response back...
    
    # Complete
    if response.get("stop_reason") == "complete":
        print(f"Done: {response.get('assistant_message')}")
    
    return response

# Example usage
process_command("Turn on the bedroom light", "conv-123")
```

## Server-Side Tool Development

### Adding New Server Tools

Edit `app/core/tool_registry.py`:

```python
# In _register_core_tools()
self.register_tool(
    name="my_new_tool",
    description="Description for the LLM",
    parameters={
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Parameter description"
            }
        },
        "required": ["param1"]
    },
    handler=self._handle_my_new_tool
)

# Add handler method
def _handle_my_new_tool(self, param1: str) -> Dict[str, Any]:
    """Handler for my_new_tool."""
    logger.info(f"Executing my_new_tool with param1={param1}")
    try:
        # Your implementation
        result = do_something(param1)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
```

## LLM Proxy Requirements

The LLM proxy API must support:

### Request Format
```json
{
  "model": "jarvis-llm",
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "tool_name",
        "description": "...",
        "parameters": {...}
      }
    }
  ]
}
```

### Response Format
```json
{
  "finish_reason": "tool_calls",  // or "stop"
  "message": {
    "role": "assistant",
    "content": "Response text",
    "tool_calls": [
      {
        "id": "call_xyz",
        "type": "function",
        "function": {
          "name": "tool_name",
          "arguments": "{\"param\": \"value\"}"
        }
      }
    ]
  }
}
```

## Testing

### Manual Testing

```bash
# Start conversation
curl -X POST http://localhost:8000/api/v0/conversation/start \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "test-123",
    "client_tools": [{
      "type": "function",
      "function": {
        "name": "test_tool",
        "description": "A test tool",
        "parameters": {"type": "object", "properties": {}}
      }
    }]
  }'

# Send command
curl -X POST http://localhost:8000/api/v0/voice/command \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "voice_command": "What time is it?",
    "conversation_id": "test-123"
  }'
```

## Migration Notes

### From Old System

The old command inference system still works if `client_tools` is not provided:

```python
# OLD: Still works
{
  "conversation_id": "conv-123",
  "available_commands": [...]  # CommandDefinition format
}

# NEW: Tool-based
{
  "conversation_id": "conv-123",
  "client_tools": [...]  # OpenAI function format
}
```

The system automatically detects which mode to use based on presence of tools in the cache.

### Breaking Changes

None - both systems can coexist. However, the tool-based system is the recommended approach going forward.

## Benefits

1. **Flexible Tool Execution**: LLM decides when and how to use tools
2. **Server-Side Tools**: Date/time, validation, future utilities
3. **Client Autonomy**: Clients define their own tools
4. **Multi-Turn Conversations**: Natural back-and-forth with tool results
5. **Validation Support**: Built-in clarification mechanism (stubbed)
6. **OpenAI Compatible**: Standard format for easy integration

## Future Enhancements

1. **Validation System**: Complete implementation of validation requests
2. **Tool Categories**: Group tools by capability
3. **Tool Authentication**: Secure tool access control
4. **Streaming**: Support streaming responses with tool calls
5. **Tool Composition**: Chain multiple tools automatically
6. **Tool Discovery**: Dynamic tool registration from plugins


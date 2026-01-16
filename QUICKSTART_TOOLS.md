# Quick Start: Tool-Based Architecture

Get up and running with the new tool-based conversation system in 5 minutes.

## Prerequisites

- Jarvis Command Center running
- LLM Proxy (see `LLM_PROXY_REQUIREMENTS.md` - minimal/no changes needed!)
- Valid API key
- Model that can follow JSON formatting instructions (Llama 3+, Mistral, etc.)

## Basic Usage

### 1. Define Your Tools

```python
client_tools = [
    {
        "type": "function",
        "function": {
            "name": "turn_on_light",
            "description": "Turn on a light in a room",
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
```

### 2. Start a Conversation

```python
import requests

API_KEY = "your-api-key"
headers = {"X-API-Key": API_KEY}

response = requests.post(
    "http://localhost:8000/api/v0/conversation/start",
    headers=headers,
    json={
        "conversation_id": "my-conv-123",
        "client_tools": client_tools
    }
)
```

### 3. Send a Command

```python
response = requests.post(
    "http://localhost:8000/api/v0/voice/command",
    headers=headers,
    json={
        "voice_command": "Turn on the bedroom light",
        "conversation_id": "my-conv-123"
    }
).json()

print(response["stop_reason"])  # "tool_calls"
print(response["tool_calls"])    # List of tools to execute
```

### 4. Execute Tools and Continue

```python
# Execute the tool
tool_call = response["tool_calls"][0]
result = turn_on_light(room="bedroom", brightness=100)

# Send results back
response = requests.post(
    "http://localhost:8000/api/v0/voice/command/continue",
    headers=headers,
    json={
        "conversation_id": "my-conv-123",
        "tool_results": [{
            "tool_call_id": tool_call["id"],
            "output": result
        }]
    }
).json()

print(response["stop_reason"])  # "complete"
```

## Server-Side Tools

Server tools execute automatically - you never see them!

### Available Server Tools

#### `get_current_date_time`
Returns comprehensive date/time context.

```python
# Just ask!
response = requests.post(
    "http://localhost:8000/api/v0/voice/command",
    headers=headers,
    json={
        "voice_command": "What time is it?",
        "conversation_id": "my-conv-123"
    }
).json()

# Server automatically:
# 1. Calls get_current_date_time
# 2. Feeds result back to LLM
# 3. Returns complete response

print(response["stop_reason"])       # "complete"
print(response["assistant_message"]) # "It's 2:30 PM on Friday..."
```

## Stop Reasons

### `"complete"`
Conversation finished. Show the assistant message and you're done.

```python
if response["stop_reason"] == "complete":
    print(response["assistant_message"])
```

### `"tool_calls"`
Execute tools and call continue endpoint.

```python
if response["stop_reason"] == "tool_calls":
    results = []
    for tool_call in response["tool_calls"]:
        result = execute_tool(tool_call)
        results.append({
            "tool_call_id": tool_call["id"],
            "output": result
        })
    
    # Continue
    response = requests.post(
        f"{base_url}/voice/command/continue",
        headers=headers,
        json={"conversation_id": conv_id, "tool_results": results}
    ).json()
```

### `"validation_required"`
Ask user for clarification (future feature, currently stubbed).

```python
if response["stop_reason"] == "validation_required":
    validation = response["validation_request"]
    user_answer = input(validation["question"])
    # Send answer back...
```

## Complete Example

```python
import requests
import json
import uuid

class JarvisClient:
    def __init__(self, api_key, base_url="http://localhost:8000"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key}
    
    def start_conversation(self, tools):
        conv_id = str(uuid.uuid4())
        requests.post(
            f"{self.base_url}/api/v0/conversation/start",
            headers=self.headers,
            json={"conversation_id": conv_id, "client_tools": tools}
        )
        return conv_id
    
    def execute_tool(self, tool_name, arguments):
        # Your tool implementations
        if tool_name == "turn_on_light":
            return {"success": True, "message": "Light on"}
        return {"error": "Unknown tool"}
    
    def send_command(self, command, conv_id):
        response = requests.post(
            f"{self.base_url}/api/v0/voice/command",
            headers=self.headers,
            json={"voice_command": command, "conversation_id": conv_id}
        ).json()
        
        # Handle tool execution loop
        while response["stop_reason"] == "tool_calls":
            results = []
            for tc in response["tool_calls"]:
                args = json.loads(tc["function"]["arguments"])
                result = self.execute_tool(tc["function"]["name"], args)
                results.append({"tool_call_id": tc["id"], "output": result})
            
            response = requests.post(
                f"{self.base_url}/api/v0/voice/command/continue",
                headers=self.headers,
                json={"conversation_id": conv_id, "tool_results": results}
            ).json()
        
        return response["assistant_message"]

# Usage
client = JarvisClient("your-api-key")
tools = [{
    "type": "function",
    "function": {
        "name": "turn_on_light",
        "description": "Turn on a light",
        "parameters": {
            "type": "object",
            "properties": {"room": {"type": "string"}},
            "required": ["room"]
        }
    }
}]

conv_id = client.start_conversation(tools)
response = client.send_command("Turn on the bedroom light", conv_id)
print(response)
```

## Adding Server-Side Tools

Edit `app/core/tool_registry.py`:

```python
# In _register_core_tools()
self.register_tool(
    name="my_tool",
    description="What the tool does",
    parameters={
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "..."}
        },
        "required": ["param"]
    },
    handler=self._handle_my_tool
)

def _handle_my_tool(self, param: str) -> Dict[str, Any]:
    """Execute my_tool."""
    result = do_something(param)
    return {"result": result}
```

## Debugging

### Enable Verbose Logging

Check logs for tool execution:
```bash
tail -f logs/jarvis.log | grep -E "🔧|⚙️|🔁"
```

### Test Tools Directly

```bash
# Check tool registry
curl -X POST http://localhost:8000/api/v0/conversation/start \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "test", "client_tools": []}'

# Send test command
curl -X POST http://localhost:8000/api/v0/voice/command \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"voice_command": "What time is it?", "conversation_id": "test"}'
```

## Common Issues

### "Conversation not found"
- Conversation expired (10 min TTL)
- Wrong conversation ID
- Start conversation before sending commands

### No tool calls returned
- LLM proxy doesn't support tools yet
- Tools not passed to LLM
- Check LLM proxy logs

### Tool execution fails
- Check tool handler implementation
- Verify arguments match parameter schema
- Check server logs for errors

## More Resources

- **Full Guide**: `TOOL_BASED_ARCHITECTURE.md`
- **LLM Proxy Setup**: `LLM_PROXY_REQUIREMENTS.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **Example Code**: `examples/tool_based_example.py`

## Support

For issues or questions, check the implementation documentation or reach out to the development team.


# Server Tools Guide

## Overview

Server-side tools are automatically discovered and registered by scanning the `app/core/tools/` directory. Tools implement the `IServerTool` interface and are available to the LLM during conversations.

## How It Works

### Dynamic Discovery

Following the same pattern as `deps.py`, the tool registry:
1. Scans `app/core/tools/` and `app/core/tools/custom/`
2. Finds all classes implementing `IServerTool`
3. Instantiates and registers them automatically
4. No manual registration needed!

### Built-in Tools

#### `resolve_relative_date`
Converts relative date/time expressions to actual dates.

**Usage:**
```json
{
  "name": "resolve_relative_date",
  "arguments": {
    "relative_term": "tomorrow",
    "timezone": "America/New_York"
  }
}
```

**Returns:**
```json
{
  "term": "tomorrow",
  "date": "2025-11-08",
  "datetime": "2025-11-08T00:00:00Z",
  "utc_start_of_day": "2025-11-08T05:00:00Z"
}
```

**Supported Terms:**
- Simple: `today`, `tomorrow`, `yesterday`
- Weekends: `this_weekend`, `next_weekend`, `last_weekend`
- Weeks: `this_week`, `next_week`, `last_week`
- Weekdays: `next_monday` through `next_sunday`, `last_monday` through `last_sunday`
- Times: `tonight`, `tomorrow_morning`, `this_afternoon`

#### `request_validation`
Requests clarification from the user.

**Usage:**
```json
{
  "name": "request_validation",
  "arguments": {
    "question": "Which room's light would you like me to turn on?",
    "parameter_name": "room",
    "options": ["bedroom", "living room", "kitchen"]
  }
}
```

## Creating a New Server Tool

### Step 1: Create Tool Class

Create a new file in `app/core/tools/` (or `app/core/tools/custom/` for custom tools):

```python
# app/core/tools/my_custom_tool.py

import logging
from typing import Dict, Any

from app.core.interfaces.iserver_tool import IServerTool

logger = logging.getLogger("uvicorn")


class MyCustomTool(IServerTool):
    """Your tool description."""
    
    @property
    def name(self) -> str:
        return "my_custom_tool"
    
    @property
    def description(self) -> str:
        return (
            "What your tool does. "
            "When to use it. "
            "What it returns."
        )
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Description of param1"
                },
                "param2": {
                    "type": "number",
                    "description": "Description of param2"
                }
            },
            "required": ["param1"]
        }
    
    def execute(self, param1: str, param2: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """Execute the tool."""
        logger.info(f"Executing my_custom_tool with param1={param1}")
        
        try:
            # Your implementation here
            result = do_something(param1, param2)
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error in my_custom_tool: {e}")
            return {
                "error": str(e),
                "message": "Failed to execute tool"
            }
```

### Step 2: That's It!

The tool will be **automatically discovered** on server startup. No manual registration needed!

Look for this in the logs:
```
🔧 Registered tool: my_custom_tool (from MyCustomTool)
✅ Discovered and registered 3 tool(s)
```

## IServerTool Interface

### Required Properties

#### `name: str`
Unique tool identifier. Used in tool calls from the LLM.

#### `description: str`
Clear description for the LLM explaining:
- What the tool does
- When to use it  
- What it returns

#### `parameters: Dict[str, Any]`
JSON Schema for tool parameters in OpenAI format:

```python
{
    "type": "object",
    "properties": {
        "param_name": {
            "type": "string|number|boolean|array|object",
            "description": "Parameter description",
            "enum": ["option1", "option2"]  # Optional
        }
    },
    "required": ["param_name"]  # Optional
}
```

### Required Method

#### `execute(**kwargs) -> Dict[str, Any]`
Execute the tool with provided arguments.

**Arguments:** Tool parameters as keyword arguments

**Returns:** JSON-serializable dict with results or errors

**Example Success:**
```python
{
    "result": "value",
    "status": "success",
    "data": {...}
}
```

**Example Error:**
```python
{
    "error": "error_code",
    "message": "Human readable error message"
}
```

## Best Practices

### 1. Clear Descriptions

```python
# ❌ Bad
"description": "Gets data"

# ✅ Good
"description": (
    "Retrieves user profile data from the database. "
    "Use this when you need information about a specific user. "
    "Returns user name, email, preferences, and last login time."
)
```

### 2. Specific Parameter Descriptions

```python
# ❌ Bad
"description": "The city"

# ✅ Good
"description": "City name for weather lookup (e.g., 'Miami', 'New York'). Can include state for disambiguation."
```

### 3. Error Handling

```python
def execute(self, **kwargs) -> Dict[str, Any]:
    try:
        # Your logic
        return {"result": value}
    except SpecificError as e:
        return {
            "error": "specific_error",
            "message": f"Helpful error message: {e}"
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {
            "error": "unexpected_error",
            "message": "Something went wrong. Please try again."
        }
```

### 4. Logging

```python
def execute(self, **kwargs) -> Dict[str, Any]:
    logger.info(f"🔧 Executing {self.name} with args: {kwargs}")
    
    result = do_work()
    
    logger.info(f"✅ {self.name} completed successfully")
    return result
```

### 5. Use Type Hints

```python
from typing import Dict, Any, Optional, List

def execute(
    self,
    required_param: str,
    optional_param: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    ...
```

## Special Tool Types

### Validation Tools

Tools that request user clarification should return:

```python
{
    "_validation_request": True,
    "question": "Clarification question?",
    "parameter_name": "param_needing_clarification",
    "options": ["option1", "option2"]  # Optional
}
```

The system will detect this and handle it specially.

### Data Transformation Tools

Tools that transform or enrich data:

```python
def execute(self, data: str, **kwargs) -> Dict[str, Any]:
    transformed = transform(data)
    return {
        "original": data,
        "transformed": transformed,
        "transformation_type": "uppercase"
    }
```

### External API Tools

Tools that call external services:

```python
def execute(self, query: str, **kwargs) -> Dict[str, Any]:
    try:
        response = requests.get(f"https://api.example.com/{query}")
        response.raise_for_status()
        return {
            "data": response.json(),
            "status_code": response.status_code
        }
    except requests.RequestException as e:
        return {
            "error": "api_error",
            "message": f"Failed to reach external API: {e}"
        }
```

## Testing Your Tool

### Unit Test

```python
# tests/test_my_tool.py

from app.core.tools.my_custom_tool import MyCustomTool

def test_my_custom_tool():
    tool = MyCustomTool()
    
    # Test successful execution
    result = tool.execute(param1="test_value")
    assert result["success"] == True
    assert "result" in result
    
    # Test error handling
    result = tool.execute(param1="")
    assert "error" in result
```

### Integration Test

```python
from app.core.tool_registry import tool_registry

def test_tool_registered():
    # Tool should be auto-discovered
    assert tool_registry.has_tool("my_custom_tool")
    
    # Test via registry
    result = tool_registry.execute_tool("my_custom_tool", param1="test")
    assert result is not None
```

### Manual Test

```python
# examples/test_my_tool.py

from app.core.tool_registry import tool_registry

tool = tool_registry.get_tool("my_custom_tool")
result = tool.execute(param1="test")
print(result)
```

## Tool Discovery Paths

Tools are discovered from:

1. **`app/core/tools/`** - Built-in tools (part of the project)
2. **`app/core/tools/custom/`** - Custom tools (user-added)

### Directory Structure

```
app/core/tools/
├── __init__.py
├── resolve_relative_date_tool.py    # Built-in
├── request_validation_tool.py       # Built-in
└── custom/
    ├── __init__.py
    └── my_custom_tool.py            # Your custom tools
```

## Debugging

### Check Registered Tools

```python
from app.core.tool_registry import tool_registry

# List all tools
print(tool_registry.get_tool_names())
# ['resolve_relative_date', 'request_validation', 'my_custom_tool']

# Get tool details
tool = tool_registry.get_tool("my_custom_tool")
print(tool.description)
print(tool.parameters)
```

### Check Logs

Look for tool registration at startup:
```
🔧 Registered tool: resolve_relative_date (from ResolveRelativeDateTool)
🔧 Registered tool: request_validation (from RequestValidationTool)
🔧 Registered tool: my_custom_tool (from MyCustomTool)
✅ Discovered and registered 3 tool(s)
```

### Test Tool Execution

```python
result = tool_registry.execute_tool(
    "my_custom_tool",
    param1="test_value"
)
print(result)
```

## FAQs

**Q: Do I need to register my tool?**
A: No! Just create a class implementing `IServerTool` in the tools directory. It's automatically discovered.

**Q: Can I override a built-in tool?**
A: Tool names must be unique. If there's a conflict, the first one found wins.

**Q: How do I remove a tool?**
A: Just delete the file. It won't be discovered on next restart.

**Q: Can tools call other tools?**
A: Yes! Import and use `tool_registry`:
```python
from app.core.tool_registry import tool_registry

result = tool_registry.execute_tool("other_tool", param="value")
```

**Q: Are tools thread-safe?**
A: Make sure your `execute()` method is thread-safe if calling external resources.

**Q: Can I have tool-specific configuration?**
A: Yes! Use environment variables or config files:
```python
import os

def execute(self, **kwargs):
    api_key = os.getenv("MY_TOOL_API_KEY")
    ...
```

## Summary

Creating server tools is simple:
1. ✅ Create a class implementing `IServerTool`
2. ✅ Place in `app/core/tools/` or `app/core/tools/custom/`
3. ✅ Restart server
4. ✅ Tool is automatically available!

No manual registration, no configuration needed. Just code and go! 🚀


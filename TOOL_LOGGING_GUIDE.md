# Tool Execution Logging Guide

## Overview

Enhanced logging for server-side tool execution with clear emoji indicators and hierarchical formatting.

## Logging Hierarchy

### Level 1: Tool Type Classification
```
🔧 [SERVER TOOL] Executing: resolve_relative_date
📲 [CLIENT TOOL] Forwarding to client: open_weather_command
```

### Level 2: Tool-Specific Details
```
      📅 Resolving: 'tomorrow' (tz: America/New_York)
      🔍 Requesting validation for: room
```

### Level 3: Results & Status
```
      └─ ✅ Success
      └─ Resolved to: 2025-11-08
      └─ Resolved to 7 dates
      └─ ⚠️  Tool error: invalid_parameter
      └─ 🔍 Validation request: Which room?
```

### Summary Statistics
```
📊 Executed 2 server tool(s)
📊 Forwarded 1 client tool(s)
```

## Emoji Reference

| Emoji | Meaning | Used For |
|-------|---------|----------|
| 🔧 | Server Tool | Tool executed on server-side |
| 📲 | Client Tool | Tool forwarded to client for execution |
| ✅ | Success | Tool completed successfully |
| ⚠️ | Warning/Error | Tool returned error or warning |
| 🔍 | Validation | Validation request to user |
| 📊 | Statistics | Summary counts |
| 📅 | Date | Date resolution operations |
| 🔨 | Debug | Low-level execution details (debug level) |
| ❌ | Error | Critical errors |

## Example Output

### Simple Tool Execution
```
🔧 [SERVER TOOL] Executing: resolve_relative_date
      📅 Resolving: 'tomorrow' (tz: America/New_York)
      └─ Resolved to: 2025-11-08
      └─ ✅ Success
```

### Validation Request
```
🔧 [SERVER TOOL] Executing: request_validation
      🔍 Requesting validation for: room
      └─ 🔍 Validation request: Which room's light?
```

### Tool Error
```
🔧 [SERVER TOOL] Executing: resolve_relative_date
      📅 Resolving: 'invalid_term' (tz: server)
      └─ ⚠️  Could not resolve term: 'invalid_term'
```

### Mixed Server & Client Tools
```
🔧 [SERVER TOOL] Executing: resolve_relative_date
      📅 Resolving: 'tomorrow' (tz: America/New_York)
      └─ Resolved to: 2025-11-08
      └─ ✅ Success

📲 [CLIENT TOOL] Forwarding to client: open_weather_command

📊 Executed 1 server tool(s)
📊 Forwarded 1 client tool(s)
```

## Log Levels

### INFO Level (Default)
- Tool type classification (🔧 / 📲)
- Tool-specific operations (📅 / 🔍)
- Success/error indicators (✅ / ⚠️)
- Summary statistics (📊)

### DEBUG Level
- Detailed arguments
- Low-level execution steps (🔨)
- Full result payloads

### ERROR Level
- Critical failures (❌)
- Unhandled exceptions

## Files Modified

1. **`app/core/tool_executor.py`**
   - Added server/client tool classification logging
   - Added hierarchical result logging
   - Added summary statistics

2. **`app/core/tool_registry.py`**
   - Added debug logging for tool execution
   - Enhanced error logging

3. **`app/core/tools/resolve_relative_date_tool.py`**
   - Added concise result logging
   - Enhanced error messages

4. **`app/core/tools/request_validation_tool.py`**
   - Added validation request logging

## Testing

Run the test to see the logging in action:
```bash
python examples/test_tool_logging.py
```

## Benefits

1. **Easy Debugging** - Quickly see which tools are server vs client
2. **Clear Hierarchy** - Indentation shows relationships
3. **Visual Scanning** - Emojis make it easy to spot issues
4. **Production Ready** - INFO level is clean, DEBUG available when needed
5. **Performance Tracking** - Summary stats show tool execution counts

## Custom Tool Logging

When creating custom tools, follow this pattern:

```python
class MyCustomTool(IServerTool):
    def execute(self, param1: str, **kwargs) -> Dict[str, Any]:
        # Tool-specific operation log
        logger.info(f"      🎯 Processing: {param1}")
        
        try:
            result = do_work(param1)
            
            # Success log
            logger.info(f"      └─ Processed {result['count']} items")
            
            return result
            
        except Exception as e:
            # Error log
            logger.error(f"      └─ ❌ Error: {e}")
            return {"error": str(e)}
```

### Logging Guidelines for Custom Tools

1. **Use consistent indentation** - 6 spaces for tool-specific logs
2. **Use tree characters** - `└─` for results under operations
3. **Pick appropriate emoji** - See reference table above
4. **Log concisely** - Don't dump entire result objects at INFO level
5. **Use DEBUG for details** - Full payloads go in debug logs

### Example Custom Tool Emojis

```python
# API calls
logger.info(f"      🌐 Calling API: {endpoint}")

# Database operations
logger.info(f"      💾 Querying database: {table}")

# File operations
logger.info(f"      📁 Reading file: {filename}")

# Calculations
logger.info(f"      🧮 Computing: {operation}")

# External services
logger.info(f"      🔌 Connecting to: {service}")
```

## Real-World Example

A typical conversation flow with logging:

```
User: "What's the weather in Miami tomorrow?"

LLM Response: {"tool_calls": [{"name": "resolve_relative_date", ...}]}

🔧 [SERVER TOOL] Executing: resolve_relative_date
      📅 Resolving: 'tomorrow' (tz: America/New_York)
      └─ Resolved to: 2025-11-08
      └─ ✅ Success
📊 Executed 1 server tool(s)

LLM Response: {"tool_calls": [{"name": "open_weather_command", ...}]}

📲 [CLIENT TOOL] Forwarding to client: open_weather_command
📊 Forwarded 1 client tool(s)

[Client executes weather command and returns result]

LLM Response: {"message": "Tomorrow in Miami will be sunny..."}
```

## Summary

Enhanced logging makes debugging tool execution easy and visual. The hierarchical structure with emojis provides immediate clarity about what's happening during tool execution.

**Quick Reference:**
- 🔧 = Server executes
- 📲 = Client executes  
- ✅ = Success
- ⚠️ = Warning/Error
- 🔍 = Validation needed
- 📊 = Summary stats


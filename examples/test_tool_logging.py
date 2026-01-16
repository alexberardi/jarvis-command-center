"""
Test to demonstrate the new tool execution logging with emojis!

This shows the clear, hierarchical logging for server-side tool calls.
"""

import logging
import json
from app.core.tool_registry import tool_registry
from app.core.tool_executor import tool_executor

# Set up logging to see all the emoji goodness
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

print("=" * 80)
print("🔧 SERVER TOOL LOGGING TEST")
print("=" * 80)
print()

# Test 1: Direct tool execution via registry
print("Test 1: Direct execution via registry")
print("-" * 80)
result = tool_registry.execute_tool(
    "resolve_relative_date",
    relative_term="tomorrow",
    timezone="America/New_York"
)
print(f"Result: {json.dumps(result, indent=2)}")
print()

# Test 2: Execute multiple tools via executor
print("\nTest 2: Multiple tool execution via executor")
print("-" * 80)

# Simulate LLM tool calls
tool_calls = [
    {
        "id": "call_1",
        "function": {
            "name": "resolve_relative_date",
            "arguments": json.dumps({
                "relative_term": "next week",
                "timezone": "America/Los_Angeles"
            })
        }
    },
    {
        "id": "call_2",
        "function": {
            "name": "request_validation",
            "arguments": json.dumps({
                "question": "Which room's light?",
                "parameter_name": "room",
                "options": ["bedroom", "living room", "kitchen"]
            })
        }
    },
    {
        "id": "call_3",
        "function": {
            "name": "open_weather_command",  # This is a client tool
            "arguments": json.dumps({
                "city": "Miami"
            })
        }
    }
]

server_results, client_tools = tool_executor.execute_tool_calls(tool_calls)

print("\nServer Results:")
for result in server_results:
    print(f"  - {result['name']}: {len(result['content'])} chars")

print("\nClient Tools:")
for tool in client_tools:
    print(f"  - {tool['function']['name']}")

print()
print("=" * 80)
print("LOGGING KEY:")
print("=" * 80)
print("""
🔧 [SERVER TOOL] - Tool executed on server
📲 [CLIENT TOOL] - Tool forwarded to client
   └─ ✅ Success - Tool completed successfully
   └─ ⚠️  - Warning or error
   └─ 🔍 - Validation request
📊 - Summary statistics
📅 - Date resolution
🔨 - Low-level execution details (debug level)
""")


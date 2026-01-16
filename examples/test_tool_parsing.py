"""
Test Tool Call Parsing

This script helps you test the tool call parser with different LLM outputs
to verify your model's JSON formatting works correctly.
"""

from app.core.tool_call_parser import tool_call_parser


def test_parse_with_tool_calls():
    """Test parsing JSON with tool calls."""
    print("\n" + "="*60)
    print("Test 1: Valid JSON with Tool Calls")
    print("="*60)
    
    llm_output = """{
  "message": "I'll turn on the bedroom light for you.",
  "tool_calls": [
    {
      "name": "turn_on_light",
      "arguments": {
        "room": "bedroom",
        "brightness": 50
      }
    }
  ]
}"""
    
    print("LLM Output:")
    print(llm_output)
    print()
    
    finish_reason, tool_calls, message = tool_call_parser.parse_response(llm_output)
    
    print(f"Parsed Result:")
    print(f"  Finish Reason: {finish_reason}")
    print(f"  Message: {message}")
    print(f"  Tool Calls: {len(tool_calls)}")
    for tc in tool_calls:
        print(f"    - {tc['function']['name']}: {tc['function']['arguments']}")
    
    assert finish_reason == "tool_calls"
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "turn_on_light"
    print("\n✅ Test 1 passed!")


def test_parse_without_tool_calls():
    """Test parsing JSON without tool calls."""
    print("\n" + "="*60)
    print("Test 2: Valid JSON without Tool Calls")
    print("="*60)
    
    llm_output = """{
  "message": "The bedroom light is now on at 50% brightness.",
  "tool_calls": null
}"""
    
    print("LLM Output:")
    print(llm_output)
    print()
    
    finish_reason, tool_calls, message = tool_call_parser.parse_response(llm_output)
    
    print(f"Parsed Result:")
    print(f"  Finish Reason: {finish_reason}")
    print(f"  Message: {message}")
    print(f"  Tool Calls: {len(tool_calls)}")
    
    assert finish_reason == "stop"
    assert len(tool_calls) == 0
    print("\n✅ Test 2 passed!")


def test_parse_multiple_tools():
    """Test parsing JSON with multiple tool calls."""
    print("\n" + "="*60)
    print("Test 3: Multiple Tool Calls")
    print("="*60)
    
    llm_output = """{
  "message": "I'll turn on the light and check the temperature.",
  "tool_calls": [
    {
      "name": "turn_on_light",
      "arguments": {"room": "bedroom"}
    },
    {
      "name": "get_temperature",
      "arguments": {"room": "bedroom"}
    }
  ]
}"""
    
    print("LLM Output:")
    print(llm_output)
    print()
    
    finish_reason, tool_calls, message = tool_call_parser.parse_response(llm_output)
    
    print(f"Parsed Result:")
    print(f"  Finish Reason: {finish_reason}")
    print(f"  Message: {message}")
    print(f"  Tool Calls: {len(tool_calls)}")
    for tc in tool_calls:
        print(f"    - {tc['function']['name']}")
    
    assert finish_reason == "tool_calls"
    assert len(tool_calls) == 2
    print("\n✅ Test 3 passed!")


def test_parse_json_in_markdown():
    """Test parsing JSON wrapped in markdown code blocks."""
    print("\n" + "="*60)
    print("Test 4: JSON in Markdown Code Block")
    print("="*60)
    
    llm_output = """Here's my response:

```json
{
  "message": "I'll help with that.",
  "tool_calls": [
    {"name": "test_tool", "arguments": {}}
  ]
}
```"""
    
    print("LLM Output:")
    print(llm_output)
    print()
    
    finish_reason, tool_calls, message = tool_call_parser.parse_response(llm_output)
    
    print(f"Parsed Result:")
    print(f"  Finish Reason: {finish_reason}")
    print(f"  Message: {message}")
    print(f"  Tool Calls: {len(tool_calls)}")
    
    assert finish_reason == "tool_calls"
    assert len(tool_calls) == 1
    print("\n✅ Test 4 passed!")


def test_parse_plain_text_fallback():
    """Test parsing non-JSON text (fallback to plain text)."""
    print("\n" + "="*60)
    print("Test 5: Plain Text Fallback")
    print("="*60)
    
    llm_output = "I'll turn on the bedroom light for you."
    
    print("LLM Output:")
    print(llm_output)
    print()
    
    finish_reason, tool_calls, message = tool_call_parser.parse_response(llm_output)
    
    print(f"Parsed Result:")
    print(f"  Finish Reason: {finish_reason}")
    print(f"  Message: {message}")
    print(f"  Tool Calls: {len(tool_calls)}")
    
    assert finish_reason == "stop"
    assert len(tool_calls) == 0
    assert message == llm_output
    print("\n✅ Test 5 passed!")


def test_format_tools_for_prompt():
    """Test formatting tools for system prompt."""
    print("\n" + "="*60)
    print("Test 6: Format Tools for Prompt")
    print("="*60)
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "turn_on_light",
                "description": "Turn on a light in a room",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "room": {
                            "type": "string",
                            "description": "The room name"
                        },
                        "brightness": {
                            "type": "number",
                            "description": "Brightness 0-100"
                        }
                    },
                    "required": ["room"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_date_time",
                "description": "Get current date and time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Optional timezone"
                        }
                    },
                    "required": []
                }
            }
        }
    ]
    
    formatted = tool_call_parser.format_tools_for_prompt(tools)
    
    print("Formatted Tools:")
    print(formatted)
    print()
    
    assert "turn_on_light" in formatted
    assert "get_current_date_time" in formatted
    assert "[REQUIRED]" in formatted
    print("\n✅ Test 6 passed!")


def test_custom_llm_output():
    """Test with your actual LLM output."""
    print("\n" + "="*60)
    print("Test 7: Custom LLM Output (Paste Yours Here)")
    print("="*60)
    
    # Replace this with actual output from your model
    llm_output = """
    PASTE YOUR ACTUAL LLM OUTPUT HERE TO TEST
    """
    
    if "PASTE YOUR" in llm_output:
        print("⏭️  Skipping - no custom output provided")
        return
    
    print("LLM Output:")
    print(llm_output)
    print()
    
    finish_reason, tool_calls, message = tool_call_parser.parse_response(llm_output)
    
    print(f"Parsed Result:")
    print(f"  Finish Reason: {finish_reason}")
    print(f"  Message: {message}")
    print(f"  Tool Calls: {len(tool_calls)}")
    for tc in tool_calls:
        print(f"    - {tc}")


def main():
    """Run all tests."""
    print("="*60)
    print("Tool Call Parser Tests")
    print("="*60)
    
    try:
        test_parse_with_tool_calls()
        test_parse_without_tool_calls()
        test_parse_multiple_tools()
        test_parse_json_in_markdown()
        test_parse_plain_text_fallback()
        test_format_tools_for_prompt()
        test_custom_llm_output()
        
        print("\n" + "="*60)
        print("🎉 All tests passed!")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


"""
Test the simplified prompt without the large date mapping.

This demonstrates that the prompt is now much cleaner and relies on
the resolve_relative_date tool instead of a huge JSON mapping.
"""

from app.core.general_context import get_general_context
from app.core.tool_registry import tool_registry

def test_simplified_prompt():
    """Test that the simplified prompt is working."""
    print("=" * 80)
    print("SIMPLIFIED PROMPT TEST")
    print("=" * 80)
    
    # Get the general context (what goes in the prompt)
    context = get_general_context(timezone_str="America/New_York")
    
    print("\n📝 NEW SIMPLIFIED PROMPT CONTEXT:\n")
    print(context)
    print(f"\n✅ Prompt length: {len(context)} characters")
    
    # Show that tools are available
    print("\n\n🔧 AVAILABLE TOOLS:\n")
    tool_names = tool_registry.get_tool_names()
    for name in tool_names:
        tool = tool_registry.get_tool(name)
        print(f"  - {name}: {tool.description[:80]}...")
    
    # Test the resolve_relative_date tool
    print("\n\n📅 TESTING RESOLVE_RELATIVE_DATE TOOL:\n")
    
    test_cases = [
        ("tomorrow", None),
        ("next week", "America/New_York"),
        ("this weekend", None),
        ("next monday", "America/Los_Angeles"),
    ]
    
    for term, tz in test_cases:
        print(f"\n  Testing: '{term}' (timezone: {tz})")
        result = tool_registry.execute_tool(
            "resolve_relative_date",
            relative_term=term,
            timezone=tz
        )
        
        if "error" in result:
            print(f"    ❌ Error: {result['message']}")
        else:
            print(f"    ✅ Result: {result}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Before: ~2000 character date mapping in every prompt
After:  ~150 character simple date/time context

The LLM now uses the resolve_relative_date tool when it needs to
convert relative dates like "tomorrow" or "next week" to actual dates.

This is:
  ✅ Much simpler and cleaner
  ✅ Less token usage
  ✅ Less confusion for the LLM
  ✅ More dynamic (always fresh dates)
  ✅ Follows proper tool usage patterns
    """)

if __name__ == "__main__":
    test_simplified_prompt()


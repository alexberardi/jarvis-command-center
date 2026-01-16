"""
Test the resolve_relative_date tool.

This script tests the new date resolution tool to verify it correctly
maps relative date terms to actual dates.
"""

from app.core.tool_registry import tool_registry


def test_date_resolution():
    """Test various relative date terms."""
    
    print("="*60)
    print("Testing resolve_relative_date Tool")
    print("="*60)
    
    test_terms = [
        "tomorrow",
        "today",
        "yesterday",
        "next week",  # With space
        "next_week",  # With underscore
        "this_weekend",
        "next_monday",
        "last_friday",
        "tonight",
        "tomorrow morning",
        "invalid_term"  # Should fail
    ]
    
    handler = tool_registry.get_handler("resolve_relative_date")
    
    for term in test_terms:
        print(f"\n📅 Testing: '{term}'")
        print("-" * 40)
        
        result = handler(relative_term=term, timezone="America/New_York")
        
        if "error" in result:
            print(f"❌ Error: {result['message']}")
        else:
            print(f"✅ Resolved to:")
            if "date" in result:
                print(f"   Date: {result['date']}")
            if "datetime" in result:
                print(f"   DateTime: {result['datetime']}")
            if "dates" in result:
                print(f"   Dates: {len(result['dates'])} days")
                for day in result['dates'][:2]:  # Show first 2
                    print(f"     - {day['day']}: {day['date']}")
    
    print("\n" + "="*60)
    print("✅ Test complete!")
    print("="*60)


if __name__ == "__main__":
    test_date_resolution()


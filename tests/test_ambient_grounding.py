"""Deterministic grounding guard — catches a response inventing a temperature or
percentage the ambient bundle never contained (the 'we're lying' failure)."""
from app.core.ambient_grounding import unsupported_numbers, is_number_grounded

BUNDLE = (
    "As of 3:15 PM, Wednesday, Jul 30.\n"
    "Currently 58°F, cloudy — 72% chance of rain after 3 PM, high 61.\n"
    "Today: 2 events — Team standup 9:00 AM; Dentist 4:30 PM."
)


class TestUnsupportedNumbers:
    def test_grounded_response_is_clean(self):
        r = "It's 58°F and cloudy with a 72% chance of rain — bring an umbrella!"
        assert unsupported_numbers(r, BUNDLE) == []
        assert is_number_grounded(r, BUNDLE)

    def test_invented_temperature_is_flagged(self):
        r = "It's a warm 68°F out there."  # bundle says 58, not 68
        assert "68°F" in unsupported_numbers(r, BUNDLE) or "68 F" in " ".join(unsupported_numbers(r, BUNDLE))
        assert not is_number_grounded(r, BUNDLE)

    def test_invented_percentage_is_flagged(self):
        r = "There's a 90% chance of rain."  # bundle says 72%
        assert not is_number_grounded(r, BUNDLE)
        assert any("90" in x for x in unsupported_numbers(r, BUNDLE))

    def test_high_number_reused_from_bundle_is_ok(self):
        # 61 is the high; a response mentioning "high of 61" must not be flagged.
        r = "Cloudy, high of 61°F today."
        assert unsupported_numbers(r, BUNDLE) == []

    def test_no_numbers_is_grounded(self):
        assert unsupported_numbers("Looks like a busy day ahead!", BUNDLE) == []

    def test_empty_bundle_flags_any_specific(self):
        assert unsupported_numbers("It's 70°F.", "") != []

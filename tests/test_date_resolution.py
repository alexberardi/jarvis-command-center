"""
Tests for date resolution helpers.

These tests cover the date resolution functionality extracted from model_service.py.
"""
import pytest
from datetime import datetime, timedelta, timezone


class TestNormalizeDateKey:
    """Tests for normalizing date key strings."""

    def test_basic_string(self):
        from app.core.date_resolution import normalize_date_key
        assert normalize_date_key("tomorrow") == "tomorrow"

    def test_with_spaces(self):
        from app.core.date_resolution import normalize_date_key
        assert normalize_date_key("next week") == "next_week"

    def test_with_multiple_spaces(self):
        from app.core.date_resolution import normalize_date_key
        assert normalize_date_key("next   week") == "next_week"

    def test_with_colon(self):
        from app.core.date_resolution import normalize_date_key
        assert normalize_date_key("at:noon") == "at_noon"

    def test_uppercase(self):
        from app.core.date_resolution import normalize_date_key
        assert normalize_date_key("TOMORROW") == "tomorrow"

    def test_mixed_case_and_spaces(self):
        from app.core.date_resolution import normalize_date_key
        assert normalize_date_key("Next Week") == "next_week"

    def test_whitespace_trimmed(self):
        from app.core.date_resolution import normalize_date_key
        assert normalize_date_key("  tomorrow  ") == "tomorrow"


class TestFlattenDateContext:
    """Tests for flattening nested date context objects."""

    def test_empty_context(self):
        from app.core.date_resolution import flatten_date_context
        assert flatten_date_context({}) == {}

    def test_non_dict_input(self):
        from app.core.date_resolution import flatten_date_context
        assert flatten_date_context(None) == {}
        assert flatten_date_context([]) == {}

    def test_current_today(self):
        from app.core.date_resolution import flatten_date_context
        context = {
            "current": {
                "utc_start_of_day": "2025-01-15T00:00:00Z"
            }
        }
        result = flatten_date_context(context)
        assert result["today"] == "2025-01-15T00:00:00Z"

    def test_relative_dates(self):
        from app.core.date_resolution import flatten_date_context
        context = {
            "relative_dates": {
                "tomorrow": {"utc_start_of_day": "2025-01-16T00:00:00Z"},
                "yesterday": {"utc_start_of_day": "2025-01-14T00:00:00Z"}
            }
        }
        result = flatten_date_context(context)
        assert result["tomorrow"] == "2025-01-16T00:00:00Z"
        assert result["yesterday"] == "2025-01-14T00:00:00Z"

    def test_relative_dates_with_datetime_field(self):
        from app.core.date_resolution import flatten_date_context
        context = {
            "relative_dates": {
                "now": {"datetime": "2025-01-15T10:30:00Z"}
            }
        }
        result = flatten_date_context(context)
        assert result["now"] == "2025-01-15T10:30:00Z"

    def test_weekdays(self):
        from app.core.date_resolution import flatten_date_context
        context = {
            "weekdays": {
                "next_monday": {"utc_start_of_day": "2025-01-20T00:00:00Z"},
                "next_friday": {"utc_start_of_day": "2025-01-24T00:00:00Z"}
            }
        }
        result = flatten_date_context(context)
        assert result["next_monday"] == "2025-01-20T00:00:00Z"
        assert result["next_friday"] == "2025-01-24T00:00:00Z"

    def test_this_week_entries(self):
        from app.core.date_resolution import flatten_date_context
        context = {
            "weeks": {
                "this_week": [
                    {"day": "Monday", "utc_start_of_day": "2025-01-13T00:00:00Z"},
                    {"day": "Tuesday", "utc_start_of_day": "2025-01-14T00:00:00Z"}
                ]
            }
        }
        result = flatten_date_context(context)
        assert result["this_monday"] == "2025-01-13T00:00:00Z"
        assert result["this_tuesday"] == "2025-01-14T00:00:00Z"

    def test_time_expressions(self):
        from app.core.date_resolution import flatten_date_context
        context = {
            "time_expressions": {
                "at 3pm": "2025-01-15T15:00:00Z",
                "at noon": "2025-01-15T12:00:00Z"
            }
        }
        result = flatten_date_context(context)
        assert result["at_3pm"] == "2025-01-15T15:00:00Z"
        assert result["at_noon"] == "2025-01-15T12:00:00Z"

    def test_bucket_lists(self):
        from app.core.date_resolution import flatten_date_context
        context = {
            "weekend": {
                "this_weekend": [
                    {"utc_start_of_day": "2025-01-18T00:00:00Z"},
                    {"utc_start_of_day": "2025-01-19T00:00:00Z"}
                ]
            }
        }
        result = flatten_date_context(context)
        assert result["this_weekend"] == ["2025-01-18T00:00:00Z", "2025-01-19T00:00:00Z"]


class TestParseTimeString:
    """Tests for parsing time strings."""

    def test_simple_am(self):
        from app.core.date_resolution import parse_time_string
        hour, minute = parse_time_string("9am")
        assert hour == 9
        assert minute == 0

    def test_simple_pm(self):
        from app.core.date_resolution import parse_time_string
        hour, minute = parse_time_string("3pm")
        assert hour == 15
        assert minute == 0

    def test_with_minutes_am(self):
        from app.core.date_resolution import parse_time_string
        hour, minute = parse_time_string("9_30am")
        assert hour == 9
        assert minute == 30

    def test_with_minutes_pm(self):
        from app.core.date_resolution import parse_time_string
        hour, minute = parse_time_string("3_45pm")
        assert hour == 15
        assert minute == 45

    def test_12am_midnight(self):
        from app.core.date_resolution import parse_time_string
        hour, minute = parse_time_string("12am")
        assert hour == 0
        assert minute == 0

    def test_12pm_noon(self):
        from app.core.date_resolution import parse_time_string
        hour, minute = parse_time_string("12pm")
        assert hour == 12
        assert minute == 0

    def test_separatorless_and_dotted_never_overflow_the_hour(self):
        """Regression: '730pm' (STT dropping the ':') gave hour=730 → +12 → crash
        ('hour must be in 0..23') in apply_time_modifier. Parse it as 7:30pm, and never
        return an out-of-range hour."""
        from app.core.date_resolution import apply_time_modifier, parse_time_string
        assert parse_time_string("730pm") == (19, 30)
        assert parse_time_string("1230pm") == (12, 30)
        assert parse_time_string("7.30pm") == (19, 30)   # dotted
        assert parse_time_string("1930") == (19, 30)     # bare 24h
        assert parse_time_string("99pm") == (0, 0)       # garbage → safe, no overflow
        # the crash path is now safe end-to-end
        assert apply_time_modifier("2099-01-01T00:00:00Z", "at_730pm") == "2099-01-01T19:30:00Z"

    def test_invalid_format(self):
        from app.core.date_resolution import parse_time_string
        hour, minute = parse_time_string("invalid")
        assert hour == 0
        assert minute == 0


class TestApplyTimeModifier:
    """Tests for applying time modifiers to base datetime."""

    def test_morning_modifier(self):
        from app.core.date_resolution import apply_time_modifier
        result = apply_time_modifier("2025-01-15T00:00:00Z", "morning")
        assert result == "2025-01-15T07:00:00Z"

    def test_afternoon_modifier(self):
        from app.core.date_resolution import apply_time_modifier
        result = apply_time_modifier("2025-01-15T00:00:00Z", "afternoon")
        assert result == "2025-01-15T13:00:00Z"

    def test_evening_modifier(self):
        from app.core.date_resolution import apply_time_modifier
        result = apply_time_modifier("2025-01-15T00:00:00Z", "evening")
        assert result == "2025-01-15T18:00:00Z"

    def test_night_modifier(self):
        from app.core.date_resolution import apply_time_modifier
        result = apply_time_modifier("2025-01-15T00:00:00Z", "night")
        assert result == "2025-01-15T21:00:00Z"

    def test_noon_modifier(self):
        from app.core.date_resolution import apply_time_modifier
        result = apply_time_modifier("2025-01-15T00:00:00Z", "noon")
        assert result == "2025-01-15T12:00:00Z"

    def test_midnight_modifier(self):
        from app.core.date_resolution import apply_time_modifier
        result = apply_time_modifier("2025-01-15T00:00:00Z", "midnight")
        assert result == "2025-01-15T00:00:00Z"

    def test_at_specific_time(self):
        from app.core.date_resolution import apply_time_modifier
        result = apply_time_modifier("2025-01-15T00:00:00Z", "at_3pm")
        assert result == "2025-01-15T15:00:00Z"

    def test_at_specific_time_with_minutes(self):
        from app.core.date_resolution import apply_time_modifier
        result = apply_time_modifier("2025-01-15T00:00:00Z", "at_3_30pm")
        assert result == "2025-01-15T15:30:00Z"

    def test_invalid_base_datetime(self):
        from app.core.date_resolution import apply_time_modifier
        result = apply_time_modifier("invalid", "morning")
        assert result is None

    def test_handles_non_z_timezone(self):
        from app.core.date_resolution import apply_time_modifier
        result = apply_time_modifier("2025-01-15T00:00:00+00:00", "noon")
        assert result == "2025-01-15T12:00:00Z"


class TestIsDatetimeParam:
    """Tests for detecting datetime parameters in JSON schema."""

    def test_datetime_format(self):
        from app.core.date_resolution import is_datetime_param
        schema = {"type": "string", "format": "date-time"}
        assert is_datetime_param(schema) is True

    def test_array_of_datetime(self):
        from app.core.date_resolution import is_datetime_param
        schema = {"type": "array", "items": {"type": "string", "format": "date-time"}}
        assert is_datetime_param(schema) is True

    def test_regular_string(self):
        from app.core.date_resolution import is_datetime_param
        schema = {"type": "string"}
        assert is_datetime_param(schema) is False

    def test_non_dict(self):
        from app.core.date_resolution import is_datetime_param
        assert is_datetime_param(None) is False
        assert is_datetime_param("string") is False


class TestIsDatetimeArray:
    """Tests for detecting datetime array parameters."""

    def test_datetime_array(self):
        from app.core.date_resolution import is_datetime_array
        schema = {"type": "array", "items": {"type": "string", "format": "date-time"}}
        assert is_datetime_array(schema) is True

    def test_single_datetime(self):
        from app.core.date_resolution import is_datetime_array
        schema = {"type": "string", "format": "date-time"}
        assert is_datetime_array(schema) is False

    def test_array_of_strings(self):
        from app.core.date_resolution import is_datetime_array
        schema = {"type": "array", "items": {"type": "string"}}
        assert is_datetime_array(schema) is False

    def test_non_dict(self):
        from app.core.date_resolution import is_datetime_array
        assert is_datetime_array(None) is False


class TestResolveDateKeys:
    """Tests for resolving date keys to datetime strings."""

    def test_single_key(self):
        from app.core.date_resolution import resolve_date_keys
        date_context = {
            "relative_dates": {
                "tomorrow": {"utc_start_of_day": "2025-01-16T00:00:00Z"}
            }
        }
        resolved, unresolved = resolve_date_keys(["tomorrow"], date_context)
        assert resolved == ["2025-01-16T00:00:00Z"]
        assert unresolved == []

    def test_multiple_keys(self):
        from app.core.date_resolution import resolve_date_keys
        date_context = {
            "relative_dates": {
                "tomorrow": {"utc_start_of_day": "2025-01-16T00:00:00Z"},
                "yesterday": {"utc_start_of_day": "2025-01-14T00:00:00Z"}
            }
        }
        resolved, unresolved = resolve_date_keys(["tomorrow", "yesterday"], date_context)
        assert "2025-01-16T00:00:00Z" in resolved
        assert "2025-01-14T00:00:00Z" in resolved

    def test_unresolved_keys(self):
        from app.core.date_resolution import resolve_date_keys
        date_context = {
            "relative_dates": {
                "tomorrow": {"utc_start_of_day": "2025-01-16T00:00:00Z"}
            }
        }
        resolved, unresolved = resolve_date_keys(["next_fortnight"], date_context)
        assert resolved == []
        assert unresolved == ["next_fortnight"]

    def test_date_with_time_modifier(self):
        from app.core.date_resolution import resolve_date_keys
        date_context = {
            "relative_dates": {
                "tomorrow": {"utc_start_of_day": "2025-01-16T00:00:00Z"}
            }
        }
        resolved, unresolved = resolve_date_keys(["tomorrow", "morning"], date_context)
        # Should combine tomorrow + morning
        assert "2025-01-16T07:00:00Z" in resolved

    def test_time_modifiers_not_unresolved(self):
        from app.core.date_resolution import resolve_date_keys
        date_context = {
            "relative_dates": {
                "tomorrow": {"utc_start_of_day": "2025-01-16T00:00:00Z"}
            }
        }
        # Time modifiers alone should not be in unresolved
        resolved, unresolved = resolve_date_keys(["morning"], date_context)
        assert "morning" not in unresolved

    def test_empty_input(self):
        from app.core.date_resolution import resolve_date_keys
        resolved, unresolved = resolve_date_keys([], {})
        assert resolved == []
        assert unresolved == []

    def test_list_values_in_context(self):
        from app.core.date_resolution import resolve_date_keys
        date_context = {
            "weekend": {
                "this_weekend": [
                    {"utc_start_of_day": "2025-01-18T00:00:00Z"},
                    {"utc_start_of_day": "2025-01-19T00:00:00Z"}
                ]
            }
        }
        resolved, unresolved = resolve_date_keys(["this_weekend"], date_context)
        assert "2025-01-18T00:00:00Z" in resolved
        assert "2025-01-19T00:00:00Z" in resolved

    def test_deduplication(self):
        from app.core.date_resolution import resolve_date_keys
        date_context = {
            "relative_dates": {
                "tomorrow": {"utc_start_of_day": "2025-01-16T00:00:00Z"}
            }
        }
        resolved, unresolved = resolve_date_keys(["tomorrow", "tomorrow"], date_context)
        # Should deduplicate
        assert resolved.count("2025-01-16T00:00:00Z") == 1

    def test_relative_time_key(self):
        from app.core.date_resolution import resolve_date_keys
        now = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        date_context = {
            "current": {"datetime": now.isoformat().replace("+00:00", "Z")}
        }
        resolved, unresolved = resolve_date_keys(["in_30_minutes"], date_context)
        expected = (now + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
        assert resolved == [expected]
        assert unresolved == []

    def test_mixed_semantic_and_relative(self):
        from app.core.date_resolution import resolve_date_keys
        now = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        date_context = {
            "current": {"datetime": now.isoformat().replace("+00:00", "Z")},
            "relative_dates": {
                "tomorrow": {"utc_start_of_day": "2025-01-16T00:00:00Z"}
            }
        }
        resolved, unresolved = resolve_date_keys(["tomorrow", "in_30_minutes"], date_context)
        assert "2025-01-16T00:00:00Z" in resolved
        expected_relative = (now + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
        assert expected_relative in resolved
        assert unresolved == []


class TestResolveRelativeTime:
    """Tests for resolving relative time keys to ISO datetimes."""

    def _make_context(self, now: datetime) -> dict:
        return {"current": {"datetime": now.isoformat().replace("+00:00", "Z")}}

    def test_resolve_relative_minutes(self):
        from app.core.date_resolution import resolve_relative_time
        now = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = resolve_relative_time("in_30_minutes", self._make_context(now))
        expected = (now + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
        assert result == expected

    def test_resolve_relative_hours(self):
        from app.core.date_resolution import resolve_relative_time
        now = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = resolve_relative_time("in_2_hours", self._make_context(now))
        expected = (now + timedelta(hours=2)).isoformat().replace("+00:00", "Z")
        assert result == expected

    def test_resolve_relative_days(self):
        from app.core.date_resolution import resolve_relative_time
        now = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = resolve_relative_time("in_3_days", self._make_context(now))
        expected = (now + timedelta(days=3)).isoformat().replace("+00:00", "Z")
        assert result == expected

    def test_resolve_relative_compound(self):
        from app.core.date_resolution import resolve_relative_time
        now = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = resolve_relative_time("in_1_hours_30_minutes", self._make_context(now))
        expected = (now + timedelta(hours=1, minutes=30)).isoformat().replace("+00:00", "Z")
        assert result == expected

    def test_resolve_invalid_relative_key(self):
        from app.core.date_resolution import resolve_relative_time
        now = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = resolve_relative_time("in_abc_minutes", self._make_context(now))
        assert result is None

    def test_resolve_non_matching_key(self):
        from app.core.date_resolution import resolve_relative_time
        now = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = resolve_relative_time("tomorrow", self._make_context(now))
        assert result is None

    def test_resolve_missing_current_datetime(self):
        from app.core.date_resolution import resolve_relative_time
        result = resolve_relative_time("in_30_minutes", {})
        assert result is None


class TestNoTimezoneStillProducesValidISO:
    """Regression: when no timezone is available (CC restart, conversation TTL
    expiry, a node that omits timezone, or the speaker-mismatch rebuild that
    used to clobber the cached zone to None), relative dates like "today" must
    still resolve to a tz-aware ISO value (ending in 'Z').

    Otherwise is_iso_datetime() rejects them, param validation fails on
    resolved_datetimes=["today"], the invalid-param retry fires, iter-2
    degrades to an empty response, and the node speaks "Task completed."
    """

    def test_today_is_iso_valid_without_timezone(self):
        from app.core.general_context import generate_date_context_object
        from app.core.date_resolution import flatten_date_context
        from app.core.param_validation import is_iso_datetime

        flat = flatten_date_context(generate_date_context_object(None))

        for key in ("today", "tomorrow", "yesterday"):
            val = flat.get(key)
            assert isinstance(val, str) and val, f"missing resolved value for {key!r}"
            assert val.endswith("Z"), f"{key} resolved without tz: {val!r}"
            assert is_iso_datetime(val), f"{key} failed is_iso_datetime: {val!r}"

    def test_today_is_iso_valid_with_empty_string_timezone(self):
        from app.core.general_context import generate_date_context_object
        from app.core.date_resolution import flatten_date_context
        from app.core.param_validation import is_iso_datetime

        flat = flatten_date_context(generate_date_context_object(""))
        val = flat.get("today")
        assert isinstance(val, str) and val.endswith("Z")
        assert is_iso_datetime(val)

    def test_valid_timezone_still_iso_valid(self):
        from app.core.general_context import generate_date_context_object
        from app.core.date_resolution import flatten_date_context
        from app.core.param_validation import is_iso_datetime

        flat = flatten_date_context(generate_date_context_object("America/New_York"))
        val = flat.get("today")
        assert isinstance(val, str) and val.endswith("Z")
        assert is_iso_datetime(val)

"""
Tests for date detection in user utterances.

Covers relative time offset detection and normalization in date_detector.py.
"""
from app.core.date_detector import detect_relative_dates


class TestDetectRelativeMinutes:
    def test_simple_minutes(self):
        result = detect_relative_dates("remind me in 30 minutes")
        assert "in_30_minutes" in result

    def test_singular_minute(self):
        result = detect_relative_dates("in 1 minute")
        assert "in_1_minute" in result


class TestDetectRelativeHours:
    def test_simple_hours(self):
        result = detect_relative_dates("in 2 hours")
        assert "in_2_hours" in result

    def test_singular_hour(self):
        result = detect_relative_dates("in 1 hour do something")
        assert "in_1_hour" in result


class TestDetectRelativeDays:
    def test_simple_days(self):
        result = detect_relative_dates("in 3 days")
        assert "in_3_days" in result

    def test_singular_day(self):
        result = detect_relative_dates("in 1 day")
        assert "in_1_day" in result


class TestDetectRelativeCompound:
    def test_compound_with_and(self):
        result = detect_relative_dates("in 1 hour and 30 minutes")
        assert "in_1_hours_30_minutes" in result

    def test_compound_without_and(self):
        result = detect_relative_dates("in 1 hour 30 minutes")
        assert "in_1_hours_30_minutes" in result


class TestDetectAnHour:
    def test_an_hour(self):
        result = detect_relative_dates("in an hour")
        assert "in_1_hours" in result

    def test_a_hour(self):
        result = detect_relative_dates("in a hour")
        assert "in_1_hours" in result


class TestDetectHalfAnHour:
    def test_half_an_hour(self):
        result = detect_relative_dates("in half an hour")
        assert "in_30_minutes" in result

    def test_half_a_hour(self):
        result = detect_relative_dates("in half a hour")
        assert "in_30_minutes" in result


class TestNoFalsePositives:
    def test_past_tense_not_detected(self):
        result = detect_relative_dates("30 minutes ago")
        # Should not contain any relative time offsets
        relative_time_keys = [k for k in result if k.startswith("in_")]
        assert relative_time_keys == []

    def test_duration_not_detected(self):
        result = detect_relative_dates("it lasted 2 hours")
        relative_time_keys = [k for k in result if k.startswith("in_")]
        assert relative_time_keys == []


class TestMixedDetection:
    def test_relative_and_semantic(self):
        result = detect_relative_dates("remind me in 30 minutes and also tomorrow")
        assert "in_30_minutes" in result
        assert "tomorrow" in result

    def test_multiple_relative(self):
        result = detect_relative_dates("set timer in 5 minutes and alarm in 2 hours")
        assert "in_5_minutes" in result
        assert "in_2_hours" in result


class TestEmptyInput:
    def test_empty_string(self):
        assert detect_relative_dates("") == []

    def test_no_dates(self):
        result = detect_relative_dates("hello world")
        assert result == []

"""
Tests for parameter validation helpers.

These tests cover the parameter validation functionality extracted from model_service.py.
"""
import pytest


class TestNormalizeParamType:
    """Tests for normalizing parameter type strings."""

    def test_simple_string(self):
        from app.core.param_validation import normalize_param_type
        base_type, is_array = normalize_param_type("string")
        assert base_type == "string"
        assert is_array is False

    def test_simple_integer(self):
        from app.core.param_validation import normalize_param_type
        base_type, is_array = normalize_param_type("integer")
        assert base_type == "integer"
        assert is_array is False

    def test_array_angle_brackets(self):
        from app.core.param_validation import normalize_param_type
        base_type, is_array = normalize_param_type("array<string>")
        assert base_type == "string"
        assert is_array is True

    def test_array_square_brackets(self):
        from app.core.param_validation import normalize_param_type
        base_type, is_array = normalize_param_type("array[datetime]")
        assert base_type == "datetime"
        assert is_array is True

    def test_array_suffix(self):
        from app.core.param_validation import normalize_param_type
        base_type, is_array = normalize_param_type("string[]")
        assert base_type == "string"
        assert is_array is True

    def test_uppercase_normalized(self):
        from app.core.param_validation import normalize_param_type
        base_type, is_array = normalize_param_type("STRING")
        assert base_type == "string"
        assert is_array is False

    def test_whitespace_trimmed(self):
        from app.core.param_validation import normalize_param_type
        base_type, is_array = normalize_param_type("  string  ")
        assert base_type == "string"
        assert is_array is False

    def test_none_input(self):
        from app.core.param_validation import normalize_param_type
        base_type, is_array = normalize_param_type(None)
        assert base_type is None
        assert is_array is False


class TestIsIsoDate:
    """Tests for ISO date format validation."""

    def test_valid_date(self):
        from app.core.param_validation import is_iso_date
        assert is_iso_date("2025-01-15") is True

    def test_valid_date_different_month(self):
        from app.core.param_validation import is_iso_date
        assert is_iso_date("2025-12-31") is True

    def test_invalid_format_slash(self):
        from app.core.param_validation import is_iso_date
        assert is_iso_date("2025/01/15") is False

    def test_invalid_format_short_year(self):
        from app.core.param_validation import is_iso_date
        assert is_iso_date("25-01-15") is False

    def test_datetime_not_date(self):
        from app.core.param_validation import is_iso_date
        assert is_iso_date("2025-01-15T00:00:00Z") is False

    def test_empty_string(self):
        from app.core.param_validation import is_iso_date
        assert is_iso_date("") is False


class TestIsIsoDatetime:
    """Tests for ISO datetime format validation."""

    def test_valid_utc_z(self):
        from app.core.param_validation import is_iso_datetime
        assert is_iso_datetime("2025-01-15T10:30:00Z") is True

    def test_valid_utc_offset(self):
        from app.core.param_validation import is_iso_datetime
        assert is_iso_datetime("2025-01-15T10:30:00+00:00") is True

    def test_valid_positive_offset(self):
        from app.core.param_validation import is_iso_datetime
        assert is_iso_datetime("2025-01-15T10:30:00+05:30") is True

    def test_valid_negative_offset(self):
        from app.core.param_validation import is_iso_datetime
        assert is_iso_datetime("2025-01-15T10:30:00-08:00") is True

    def test_date_only_not_datetime(self):
        from app.core.param_validation import is_iso_datetime
        assert is_iso_datetime("2025-01-15") is False

    def test_missing_timezone(self):
        from app.core.param_validation import is_iso_datetime
        assert is_iso_datetime("2025-01-15T10:30:00") is False

    def test_invalid_format(self):
        from app.core.param_validation import is_iso_datetime
        assert is_iso_datetime("not-a-datetime") is False

    def test_empty_string(self):
        from app.core.param_validation import is_iso_datetime
        assert is_iso_datetime("") is False


class TestValidateScalar:
    """Tests for scalar value validation."""

    def test_string_valid(self):
        from app.core.param_validation import validate_scalar
        assert validate_scalar("hello", "string") is True

    def test_string_invalid(self):
        from app.core.param_validation import validate_scalar
        assert validate_scalar(123, "string") is False

    def test_integer_valid(self):
        from app.core.param_validation import validate_scalar
        assert validate_scalar(42, "integer") is True
        assert validate_scalar(42, "int") is True

    def test_integer_bool_not_int(self):
        from app.core.param_validation import validate_scalar
        assert validate_scalar(True, "integer") is False

    def test_integer_invalid_float(self):
        from app.core.param_validation import validate_scalar
        assert validate_scalar(3.14, "integer") is False

    def test_float_valid(self):
        from app.core.param_validation import validate_scalar
        assert validate_scalar(3.14, "float") is True
        assert validate_scalar(3.14, "number") is True
        assert validate_scalar(3.14, "double") is True

    def test_float_int_is_valid(self):
        from app.core.param_validation import validate_scalar
        assert validate_scalar(42, "float") is True

    def test_float_bool_not_float(self):
        from app.core.param_validation import validate_scalar
        assert validate_scalar(True, "float") is False

    def test_bool_valid(self):
        from app.core.param_validation import validate_scalar
        assert validate_scalar(True, "bool") is True
        assert validate_scalar(False, "boolean") is True

    def test_bool_invalid(self):
        from app.core.param_validation import validate_scalar
        assert validate_scalar(1, "bool") is False
        assert validate_scalar("true", "bool") is False

    def test_date_valid(self):
        from app.core.param_validation import validate_scalar
        assert validate_scalar("2025-01-15", "date") is True

    def test_date_invalid(self):
        from app.core.param_validation import validate_scalar
        assert validate_scalar("2025/01/15", "date") is False

    def test_datetime_valid(self):
        from app.core.param_validation import validate_scalar
        assert validate_scalar("2025-01-15T10:30:00Z", "datetime") is True

    def test_datetime_invalid(self):
        from app.core.param_validation import validate_scalar
        assert validate_scalar("2025-01-15", "datetime") is False

    def test_unknown_type_passes(self):
        from app.core.param_validation import validate_scalar
        # Unknown types should pass (permissive)
        assert validate_scalar("anything", "unknown_type") is True


class TestValidateValue:
    """Tests for validating values against types (scalar and array)."""

    def test_scalar_string(self):
        from app.core.param_validation import validate_value
        assert validate_value("hello", "string", False) is True

    def test_scalar_invalid(self):
        from app.core.param_validation import validate_value
        assert validate_value(123, "string", False) is False

    def test_array_valid(self):
        from app.core.param_validation import validate_value
        assert validate_value(["a", "b", "c"], "string", True) is True

    def test_array_invalid_not_list(self):
        from app.core.param_validation import validate_value
        assert validate_value("not a list", "string", True) is False

    def test_array_invalid_elements(self):
        from app.core.param_validation import validate_value
        assert validate_value(["a", 123, "c"], "string", True) is False

    def test_array_of_integers(self):
        from app.core.param_validation import validate_value
        assert validate_value([1, 2, 3], "integer", True) is True

    def test_array_of_datetimes(self):
        from app.core.param_validation import validate_value
        datetimes = ["2025-01-15T10:00:00Z", "2025-01-16T10:00:00Z"]
        assert validate_value(datetimes, "datetime", True) is True

    def test_empty_array(self):
        from app.core.param_validation import validate_value
        assert validate_value([], "string", True) is True


class TestFindInvalidParams:
    """Tests for finding invalid parameters in tool calls."""

    def test_no_invalid_params(self):
        from app.core.param_validation import find_invalid_params

        param_types = {
            "my_tool": {"name": "string", "count": "integer"}
        }
        tool_calls = [{
            "function": {
                "name": "my_tool",
                "arguments": '{"name": "test", "count": 5}'
            }
        }]

        invalid = find_invalid_params(tool_calls, param_types)
        assert invalid == []

    def test_invalid_string_param(self):
        from app.core.param_validation import find_invalid_params

        param_types = {
            "my_tool": {"name": "string"}
        }
        tool_calls = [{
            "function": {
                "name": "my_tool",
                "arguments": '{"name": 123}'
            }
        }]

        invalid = find_invalid_params(tool_calls, param_types)
        assert len(invalid) == 1
        assert "my_tool.name" in invalid[0]

    def test_invalid_datetime_param(self):
        from app.core.param_validation import find_invalid_params

        param_types = {
            "set_reminder": {"when": "datetime"}
        }
        tool_calls = [{
            "function": {
                "name": "set_reminder",
                "arguments": '{"when": "tomorrow"}'
            }
        }]

        invalid = find_invalid_params(tool_calls, param_types)
        assert len(invalid) == 1
        assert "set_reminder.when" in invalid[0]

    def test_tool_not_in_param_types(self):
        from app.core.param_validation import find_invalid_params

        param_types = {}
        tool_calls = [{
            "function": {
                "name": "unknown_tool",
                "arguments": '{"name": 123}'
            }
        }]

        # Unknown tools should not be flagged
        invalid = find_invalid_params(tool_calls, param_types)
        assert invalid == []

    def test_missing_param_not_invalid(self):
        from app.core.param_validation import find_invalid_params

        param_types = {
            "my_tool": {"name": "string", "optional_count": "integer"}
        }
        tool_calls = [{
            "function": {
                "name": "my_tool",
                "arguments": '{"name": "test"}'
            }
        }]

        # Missing params are not considered invalid (may be optional)
        invalid = find_invalid_params(tool_calls, param_types)
        assert invalid == []

    def test_invalid_array_param(self):
        from app.core.param_validation import find_invalid_params

        param_types = {
            "batch_tool": {"dates": "array<datetime>"}
        }
        tool_calls = [{
            "function": {
                "name": "batch_tool",
                "arguments": '{"dates": ["tomorrow", "next week"]}'
            }
        }]

        invalid = find_invalid_params(tool_calls, param_types)
        assert len(invalid) == 1
        assert "batch_tool.dates" in invalid[0]

    def test_valid_array_param(self):
        from app.core.param_validation import find_invalid_params

        param_types = {
            "batch_tool": {"dates": "array<datetime>"}
        }
        tool_calls = [{
            "function": {
                "name": "batch_tool",
                "arguments": '{"dates": ["2025-01-15T10:00:00Z", "2025-01-16T10:00:00Z"]}'
            }
        }]

        invalid = find_invalid_params(tool_calls, param_types)
        assert invalid == []

    def test_arguments_as_dict(self):
        from app.core.param_validation import find_invalid_params

        param_types = {
            "my_tool": {"name": "string"}
        }
        tool_calls = [{
            "function": {
                "name": "my_tool",
                "arguments": {"name": "test"}  # Dict instead of JSON string
            }
        }]

        invalid = find_invalid_params(tool_calls, param_types)
        assert invalid == []

    def test_malformed_json_arguments(self):
        from app.core.param_validation import find_invalid_params

        param_types = {
            "my_tool": {"name": "string"}
        }
        tool_calls = [{
            "function": {
                "name": "my_tool",
                "arguments": "not valid json"
            }
        }]

        # Malformed JSON should be handled gracefully
        invalid = find_invalid_params(tool_calls, param_types)
        assert invalid == []  # Can't validate, so no invalids reported

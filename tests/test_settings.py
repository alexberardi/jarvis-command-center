"""Tests for the settings service and definitions.

These tests cover:
- Settings definitions
- Settings service behavior
"""

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from jarvis_settings_client import SettingDefinition, SettingsService
from jarvis_settings_client.types import SettingValue

from app.services.settings_definitions import SETTINGS_DEFINITIONS
from app.services.settings_service import get_settings_service, reset_settings_service


class TestSettingsDefinitions:
    """Tests for settings definitions."""

    def test_all_definitions_have_required_fields(self):
        """Test that all definitions have required fields."""
        for definition in SETTINGS_DEFINITIONS:
            assert definition.key, f"Missing key for definition"
            assert definition.category, f"Missing category for {definition.key}"
            assert definition.value_type in ("string", "int", "float", "bool", "json"), \
                f"Invalid value_type for {definition.key}: {definition.value_type}"

    def test_no_duplicate_keys(self):
        """Test that there are no duplicate keys."""
        keys = [d.key for d in SETTINGS_DEFINITIONS]
        assert len(keys) == len(set(keys)), "Duplicate keys found in SETTINGS_DEFINITIONS"

    def test_key_format(self):
        """Test that keys follow the expected format."""
        for definition in SETTINGS_DEFINITIONS:
            # Keys should be lowercase with dots
            assert "." in definition.key, f"Key should contain dots: {definition.key}"
            assert definition.key == definition.key.lower(), \
                f"Key should be lowercase: {definition.key}"

    def test_expected_settings_exist(self):
        """Test that expected command-center settings are defined."""
        keys = [d.key for d in SETTINGS_DEFINITIONS]
        assert "llm.interface" in keys
        assert "llm.proxy.url" in keys
        assert "tool_classifier.enabled" in keys
        assert "tool_classifier.min_confidence" in keys
        assert "tool_router.filter_min_confidence" in keys
        assert "model.small_model_mode" in keys

    def test_categories_are_valid(self):
        """Test that categories are descriptive."""
        categories = set(d.category for d in SETTINGS_DEFINITIONS)
        expected_categories = {"llm", "tool_classifier", "tool_router", "transcription",
                              "prompt", "model", "conversation", "admin", "memory",
                              "network", "oauth", "smart_home", "adapter", "voice",
                              "routines", "web_search", "updates"}
        assert categories == expected_categories


class TestSettingsServiceCache:
    """Tests for SettingsService caching behavior."""

    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return SettingsService(
            definitions=SETTINGS_DEFINITIONS,
            get_db_session=lambda: None,
            setting_model=None,
        )

    def test_cache_hit(self, service):
        """Test that cached values are returned without DB query."""
        # Manually populate cache
        cache_key = service._make_cache_key("llm.interface")
        service._cache[cache_key] = SettingValue(
            value="CachedInterface",
            value_type="string",
            requires_reload=False,
            is_secret=False,
            env_fallback=None,
            from_db=True,
            cached_at=time.time(),
        )

        # Should return cached value without DB query
        result = service.get("llm.interface")
        assert result == "CachedInterface"

    def test_cache_expiry(self, service):
        """Test that expired cache entries are not used."""
        # Populate cache with expired entry (use a setting with env_fallback)
        cache_key = service._make_cache_key("tool_classifier.enabled")
        service._cache[cache_key] = SettingValue(
            value="ExpiredValue",
            value_type="bool",
            requires_reload=False,
            is_secret=False,
            env_fallback="JARVIS_TOOL_CLASSIFIER_ENABLED",
            from_db=True,
            cached_at=time.time() - 120,  # 2 minutes ago (expired)
        )

        # Should fall through to env/default since cache is expired
        with patch.dict(os.environ, {"JARVIS_TOOL_CLASSIFIER_ENABLED": "false"}):
            result = service.get("tool_classifier.enabled")
            assert result is False

    def test_invalidate_all(self, service):
        """Test invalidating entire cache."""
        key1_cache = service._make_cache_key("test.key1")
        key2_cache = service._make_cache_key("test.key2")

        service._cache[key1_cache] = SettingValue(
            value="value1",
            value_type="string",
            requires_reload=False,
            is_secret=False,
            env_fallback=None,
            from_db=True,
            cached_at=time.time(),
        )
        service._cache[key2_cache] = SettingValue(
            value="value2",
            value_type="string",
            requires_reload=False,
            is_secret=False,
            env_fallback=None,
            from_db=True,
            cached_at=time.time(),
        )

        service.invalidate_cache()

        assert len(service._cache) == 0


class TestSettingsServiceEnvFallback:
    """Tests for environment variable fallback."""

    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return SettingsService(
            definitions=SETTINGS_DEFINITIONS,
            get_db_session=lambda: None,
            setting_model=None,
        )

    def test_env_fallback_when_db_unavailable(self, service):
        """Test that env vars are used when DB is unavailable."""
        with patch.dict(os.environ, {"JARVIS_TOOL_CLASSIFIER_ENABLED": "false"}):
            result = service.get("tool_classifier.enabled")
            assert result is False

    def test_llm_interface_uses_definition_default_not_env(self, service):
        """Test that llm.interface uses definition default, not env var."""
        with patch.dict(os.environ, {"JARVIS_MODEL_INTERFACE": "EnvInterface"}):
            result = service.get("llm.interface")
            # Should return definition default, NOT env var (no env_fallback)
            assert result == "Qwen25MediumUntrained"

    def test_default_when_no_env(self, service):
        """Test that defaults are used when no env var is set."""
        with patch.dict(os.environ, {}, clear=True):
            result = service.get("tool_classifier.min_confidence")
            # Should return definition default (0.6)
            assert result == 0.6

    def test_unknown_key_returns_none(self, service):
        """Test that unknown keys return None."""
        result = service.get("unknown.key")
        assert result is None

    def test_unknown_key_returns_provided_default(self, service):
        """Test that unknown keys return provided default."""
        result = service.get("unknown.key", "my_default")
        assert result == "my_default"


class TestSettingsServiceTypedGetters:
    """Tests for typed getter methods."""

    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return SettingsService(
            definitions=SETTINGS_DEFINITIONS,
            get_db_session=lambda: None,
            setting_model=None,
        )

    def test_get_bool(self, service):
        """Test get_bool method."""
        with patch.dict(os.environ, {"JARVIS_TOOL_CLASSIFIER_ENABLED": "true"}):
            result = service.get_bool("tool_classifier.enabled", False)
            assert result is True
            assert isinstance(result, bool)

    def test_get_float(self, service):
        """Test get_float method."""
        with patch.dict(os.environ, {"JARVIS_TOOL_CLASSIFIER_MIN_CONFIDENCE": "0.75"}):
            result = service.get_float("tool_classifier.min_confidence", 0.0)
            assert result == 0.75
            assert isinstance(result, float)

    def test_get_int(self, service):
        """Test get_int method."""
        with patch.dict(os.environ, {"JARVIS_CONVERSATION_MAX_TURNS": "50"}):
            result = service.get_int("conversation.max_turns", 0)
            assert result == 50
            assert isinstance(result, int)

    def test_get_str(self, service):
        """Test get_str method."""
        with patch.dict(os.environ, {"JARVIS_LLM_PROXY_API_URL": "http://test:1234"}):
            result = service.get_str("llm.proxy.url", "")
            assert result == "http://test:1234"
            assert isinstance(result, str)


class TestSettingsServiceListMethods:
    """Tests for listing methods."""

    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return SettingsService(
            definitions=SETTINGS_DEFINITIONS,
            get_db_session=lambda: None,
            setting_model=None,
        )

    def test_list_categories(self, service):
        """Test list_categories returns unique categories."""
        categories = service.list_categories()

        assert isinstance(categories, list)
        assert len(categories) > 0
        assert "llm" in categories
        assert "tool_classifier" in categories
        # Should be sorted
        assert categories == sorted(categories)

    def test_list_all(self, service):
        """Test list_all returns all settings."""
        settings = service.list_all()

        assert isinstance(settings, list)
        assert len(settings) == len(SETTINGS_DEFINITIONS)

        # Check structure of first setting
        first = settings[0]
        assert "key" in first
        assert "value" in first
        assert "value_type" in first
        assert "category" in first
        assert "from_db" in first

    def test_list_all_with_category_filter(self, service):
        """Test list_all with category filter."""
        settings = service.list_all(category="llm")

        assert all(s["category"] == "llm" for s in settings)
        assert len(settings) > 0


class TestSingleton:
    """Tests for singleton behavior via get_settings_service."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset singleton before and after each test."""
        reset_settings_service()
        yield
        reset_settings_service()

    def test_singleton_instance(self):
        """Test that get_settings_service returns same instance."""
        # Mock the db imports to avoid actual DB connection
        mock_setting = MagicMock()
        mock_session_local = MagicMock()

        # Patch at the module level where it's imported inside the function
        with patch("app.db.get_session_local", return_value=mock_session_local):
            with patch("app.models.Setting", mock_setting):
                service1 = get_settings_service()
                service2 = get_settings_service()

                assert service1 is service2

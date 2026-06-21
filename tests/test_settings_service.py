"""Tests for Settings service using jarvis-settings-client.

Tests the cascade lookup behavior: User > Node > Household > Default.
"""

import pytest

from jarvis_settings_client import SettingDefinition, SettingsService

from app.models import Setting


# Test definitions for these tests
TEST_SETTINGS_DEFINITIONS = [
    SettingDefinition(
        key="tts_url",
        category="tts",
        value_type="string",
        default="http://localhost:7707",
        description="TTS service URL",
    ),
    SettingDefinition(
        key="whisper_url",
        category="whisper",
        value_type="string",
        default="http://localhost:7706",
        description="Whisper service URL",
    ),
    SettingDefinition(
        key="tts_provider",
        category="tts",
        value_type="string",
        default="default-provider",
        description="TTS provider",
    ),
]


class TestSettingModel:
    """Tests for the Setting SQLAlchemy model."""

    def test_create_setting_with_all_fields(self, test_db):
        """Test creating a setting with all scope fields."""
        setting = Setting(
            key="tts_url",
            value="http://localhost:7707",
            value_type="string",
            category="tts",
            household_id="h123",
            node_id="kitchen-pi",
        )
        test_db.add(setting)
        test_db.commit()
        test_db.refresh(setting)

        assert setting.id is not None
        assert setting.key == "tts_url"
        assert setting.value == "http://localhost:7707"
        assert setting.household_id == "h123"
        assert setting.node_id == "kitchen-pi"
        assert setting.created_at is not None
        assert setting.updated_at is not None

    def test_create_system_default_setting(self, test_db):
        """Test creating a system default (NULL household and node)."""
        setting = Setting(
            key="tts_url",
            value="http://default:7707",
            value_type="string",
            category="tts",
            household_id=None,
            node_id=None,
        )
        test_db.add(setting)
        test_db.commit()
        test_db.refresh(setting)

        assert setting.household_id is None
        assert setting.node_id is None

    def test_create_household_setting(self, test_db):
        """Test creating a household-level setting (NULL node)."""
        setting = Setting(
            key="tts_url",
            value="http://household:7707",
            value_type="string",
            category="tts",
            household_id="h456",
            node_id=None,
        )
        test_db.add(setting)
        test_db.commit()
        test_db.refresh(setting)

        assert setting.household_id == "h456"
        assert setting.node_id is None

    def test_unique_constraint_prevents_duplicates(self, test_db):
        """Test that same key+household+node+user cannot be duplicated.

        The uq_setting_scope constraint is 4-column (key, household_id, node_id,
        user_id). Both rows set a non-NULL user_id so they collide — with
        user_id=NULL, Postgres' NULLS DISTINCT semantics would treat them as
        distinct and no IntegrityError would be raised.
        """
        setting1 = Setting(
            key="tts_url",
            value="http://first:7707",
            value_type="string",
            category="tts",
            household_id="h123",
            node_id="node1",
            user_id=1,
        )
        test_db.add(setting1)
        test_db.commit()

        setting2 = Setting(
            key="tts_url",
            value="http://second:7707",
            value_type="string",
            category="tts",
            household_id="h123",
            node_id="node1",
            user_id=1,
        )
        test_db.add(setting2)
        with pytest.raises(Exception):  # IntegrityError
            test_db.commit()

    def test_same_key_different_scopes_allowed(self, test_db):
        """Test that same key can exist at different scopes."""
        # System default
        setting1 = Setting(
            key="tts_url",
            value="http://default:7707",
            value_type="string",
            category="tts",
        )
        test_db.add(setting1)

        # Household level
        setting2 = Setting(
            key="tts_url",
            value="http://household:7707",
            value_type="string",
            category="tts",
            household_id="h123",
        )
        test_db.add(setting2)

        # Node level
        setting3 = Setting(
            key="tts_url",
            value="http://node:7707",
            value_type="string",
            category="tts",
            household_id="h123",
            node_id="node1",
        )
        test_db.add(setting3)

        test_db.commit()

        # All three should exist
        settings = test_db.query(Setting).filter(Setting.key == "tts_url").all()
        assert len(settings) == 3


class TestSettingsServiceGet:
    """Tests for SettingsService.get() cascade lookup."""

    @pytest.fixture
    def service(self, test_db):
        """Create a SettingsService for testing."""
        return SettingsService(
            definitions=TEST_SETTINGS_DEFINITIONS,
            get_db_session=lambda: test_db,
            setting_model=Setting,
        )

    def test_returns_node_setting_when_exists(self, test_db, service):
        """Should return node-specific value when it exists."""
        # Create all three levels
        test_db.add(Setting(
            key="tts_url", value="http://default:7707",
            value_type="string", category="tts"
        ))
        test_db.add(Setting(
            key="tts_url", value="http://household:7707",
            value_type="string", category="tts", household_id="h123"
        ))
        test_db.add(Setting(
            key="tts_url", value="http://node:7707",
            value_type="string", category="tts",
            household_id="h123", node_id="node1"
        ))
        test_db.commit()

        value = service.get("tts_url", household_id="h123", node_id="node1")
        assert value == "http://node:7707"

    def test_falls_back_to_household_when_no_node_setting(self, test_db, service):
        """Should fall back to household value when no node-specific setting."""
        test_db.add(Setting(
            key="tts_url", value="http://default:7707",
            value_type="string", category="tts"
        ))
        test_db.add(Setting(
            key="tts_url", value="http://household:7707",
            value_type="string", category="tts", household_id="h123"
        ))
        test_db.commit()

        value = service.get("tts_url", household_id="h123", node_id="node1")
        assert value == "http://household:7707"

    def test_falls_back_to_default_when_no_household_setting(self, test_db, service):
        """Should fall back to system default when no household setting."""
        test_db.add(Setting(
            key="tts_url", value="http://default:7707",
            value_type="string", category="tts"
        ))
        test_db.commit()

        value = service.get("tts_url", household_id="h123", node_id="node1")
        assert value == "http://default:7707"

    def test_returns_definition_default_when_no_setting_exists(self, service):
        """Should return definition default when no setting exists at any level."""
        value = service.get("tts_url", household_id="h123", node_id="node1")
        assert value == "http://localhost:7707"  # From TEST_SETTINGS_DEFINITIONS

    def test_returns_none_for_unknown_key(self, service):
        """Should return None for unknown setting key."""
        value = service.get("nonexistent_key", household_id="h123", node_id="node1")
        assert value is None

    def test_household_lookup_without_node_id(self, test_db, service):
        """Should look up household setting when no node_id provided."""
        test_db.add(Setting(
            key="tts_url", value="http://default:7707",
            value_type="string", category="tts"
        ))
        test_db.add(Setting(
            key="tts_url", value="http://household:7707",
            value_type="string", category="tts", household_id="h123"
        ))
        test_db.commit()

        value = service.get("tts_url", household_id="h123")
        assert value == "http://household:7707"

    def test_default_lookup_without_any_scope(self, test_db, service):
        """Should look up system default when no scope provided."""
        test_db.add(Setting(
            key="tts_url", value="http://default:7707",
            value_type="string", category="tts"
        ))
        test_db.commit()

        value = service.get("tts_url")
        assert value == "http://default:7707"


class TestSettingsServiceSet:
    """Tests for SettingsService.set()."""

    @pytest.fixture
    def service(self, test_db):
        """Create a SettingsService for testing."""
        return SettingsService(
            definitions=TEST_SETTINGS_DEFINITIONS,
            get_db_session=lambda: test_db,
            setting_model=Setting,
        )

    def test_set_system_default(self, test_db, service):
        """Should create a system default setting."""
        service.set("tts_url", "http://new-default:7707")

        setting = test_db.query(Setting).filter(
            Setting.key == "tts_url",
            Setting.household_id.is_(None),
            Setting.node_id.is_(None),
        ).first()

        assert setting is not None
        assert setting.value == "http://new-default:7707"

    def test_set_household_setting(self, test_db, service):
        """Should create a household-level setting."""
        service.set("tts_url", "http://household:7707", household_id="h123")

        setting = test_db.query(Setting).filter(
            Setting.key == "tts_url",
            Setting.household_id == "h123",
            Setting.node_id.is_(None),
        ).first()

        assert setting is not None
        assert setting.value == "http://household:7707"

    def test_set_node_setting(self, test_db, service):
        """Should create a node-level setting."""
        service.set("tts_url", "http://node:7707", household_id="h123", node_id="node1")

        setting = test_db.query(Setting).filter(
            Setting.key == "tts_url",
            Setting.household_id == "h123",
            Setting.node_id == "node1",
        ).first()

        assert setting is not None
        assert setting.value == "http://node:7707"

    def test_update_existing_setting(self, test_db, service):
        """Should update value of existing setting."""
        # Create initial setting
        test_db.add(Setting(
            key="tts_url", value="http://old:7707",
            value_type="string", category="tts", household_id="h123"
        ))
        test_db.commit()

        service.set("tts_url", "http://new:7707", household_id="h123")

        settings = test_db.query(Setting).filter(
            Setting.key == "tts_url",
            Setting.household_id == "h123",
        ).all()

        assert len(settings) == 1
        assert settings[0].value == "http://new:7707"


class TestSettingsServiceListAll:
    """Tests for SettingsService.list_all() merged lookup."""

    @pytest.fixture
    def service(self, test_db):
        """Create a SettingsService for testing."""
        return SettingsService(
            definitions=TEST_SETTINGS_DEFINITIONS,
            get_db_session=lambda: test_db,
            setting_model=Setting,
        )

    def test_returns_all_definitions(self, service):
        """Should return all settings from definitions."""
        settings = service.list_all()

        assert len(settings) == 3  # 3 definitions
        keys = [s["key"] for s in settings]
        assert "tts_url" in keys
        assert "whisper_url" in keys
        assert "tts_provider" in keys

    def test_returns_db_values_when_set(self, test_db, service):
        """Should return database values when set."""
        test_db.add(Setting(
            key="tts_url", value="http://custom:7707",
            value_type="string", category="tts"
        ))
        test_db.commit()

        settings = service.list_all()
        tts_setting = next(s for s in settings if s["key"] == "tts_url")

        assert tts_setting["value"] == "http://custom:7707"
        assert tts_setting["from_db"] is True

    def test_returns_defaults_when_no_db_values(self, service):
        """Should return definition defaults when no DB values."""
        settings = service.list_all()
        tts_setting = next(s for s in settings if s["key"] == "tts_url")

        assert tts_setting["value"] == "http://localhost:7707"  # Definition default
        assert tts_setting["from_db"] is False

    def test_filter_by_category(self, service):
        """Should filter by category."""
        settings = service.list_all(category="tts")

        assert len(settings) == 2
        assert all(s["category"] == "tts" for s in settings)

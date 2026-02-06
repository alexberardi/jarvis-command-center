"""Settings service for jarvis-command-center.

Provides runtime configuration that can be modified without restarting.
Settings are stored in the database with fallback to environment variables.
Uses the shared jarvis-settings-client library.
"""

from jarvis_settings_client import SettingsService

from app.services.settings_definitions import SETTINGS_DEFINITIONS

# Global singleton
_settings_service: SettingsService | None = None


def get_settings_service() -> SettingsService:
    """Get the global SettingsService instance."""
    global _settings_service
    if _settings_service is None:
        from app.models import Setting
        from app.db import get_session_local

        SessionLocal = get_session_local()
        _settings_service = SettingsService(
            definitions=SETTINGS_DEFINITIONS,
            get_db_session=SessionLocal,
            setting_model=Setting,
        )
    return _settings_service


def reset_settings_service() -> None:
    """Reset the settings service singleton (for testing)."""
    global _settings_service
    _settings_service = None

"""Settings service for jarvis-command-center.

Provides runtime configuration that can be modified without restarting.
Settings are stored in the database with fallback to environment variables.
Uses the shared jarvis-settings-client library.
"""

import logging

from jarvis_settings_client import SettingsService as BaseSettingsService

from app.services.settings_definitions import SETTINGS_DEFINITIONS

logger = logging.getLogger(__name__)


class CommandCenterSettingsService(BaseSettingsService):
    """Settings service for command-center with helper methods."""

    def get_llm_config(self) -> dict:
        """Get LLM configuration settings as a dict."""
        return {
            "interface": self.get("llm.interface"),
            "proxy_url": self.get("llm.proxy.url"),
        }

    def get_tool_classifier_config(self) -> dict:
        """Get tool classifier configuration settings."""
        return {
            "enabled": self.get_bool("tool_classifier.enabled", True),
            "min_confidence": self.get_float("tool_classifier.min_confidence", 0.6),
        }

    def get_tool_router_config(self) -> dict:
        """Get tool router configuration settings."""
        return {
            "filter_min_confidence": self.get_float("tool_router.filter_min_confidence", 0.85),
        }

    def get_prompt_config(self) -> dict:
        """Get prompt generation configuration settings."""
        return {
            "include_antipatterns": self.get_bool("prompt.include_antipatterns", True),
            "include_param_descriptions": self.get_bool("prompt.include_param_descriptions", True),
            "small_model_mode": self.get_bool("model.small_model_mode", True),
        }

    def get_conversation_config(self) -> dict:
        """Get conversation configuration settings."""
        return {
            "max_turns": self.get_int("conversation.max_turns", 20),
            "cache_ttl_seconds": self.get_int("conversation.cache_ttl_seconds", 3600),
        }


# Global singleton
_settings_service: CommandCenterSettingsService | None = None


def get_settings_service() -> CommandCenterSettingsService:
    """Get the global SettingsService instance."""
    global _settings_service
    if _settings_service is None:
        from app.models import Setting
        from app.db import get_session_local

        SessionLocal = get_session_local()
        _settings_service = CommandCenterSettingsService(
            definitions=SETTINGS_DEFINITIONS,
            get_db_session=SessionLocal,
            setting_model=Setting,
        )
    return _settings_service


def reset_settings_service() -> None:
    """Reset the settings service singleton (for testing)."""
    global _settings_service
    _settings_service = None

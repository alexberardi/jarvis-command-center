"""
Service configuration via jarvis-config-client.

Provides centralized access to service URLs fetched from jarvis-config-service.
Falls back to environment variables and defaults when services aren't found.
"""

import logging
import os
from typing import Optional

from jarvis_config_client import (
    init as config_init,
    shutdown as config_shutdown,
    get_service_url,
    get_all_services,
)

logger = logging.getLogger(__name__)

# Default URLs (fallbacks when config service and env vars unavailable)
_DEFAULTS = {
    "jarvis-llm-proxy": "http://localhost:8000",
    "jarvis-auth": "http://localhost:8007",
    "jarvis-logs": "http://localhost:8006",
    "jarvis-whisper": "http://localhost:8012",
    "jarvis-tts": "http://localhost:8009",
    "jarvis-mqtt-broker": "mqtt://localhost:1883",
}

# Mapping of service names to legacy env var names
_ENV_VAR_FALLBACKS = {
    "jarvis-llm-proxy": "JARVIS_LLM_PROXY_API_URL",
    "jarvis-auth": "JARVIS_AUTH_BASE_URL",
    "jarvis-logs": "JARVIS_LOGS_URL",
    "jarvis-whisper": "JARVIS_WHISPER_URL",
    "jarvis-tts": "JARVIS_TTS_URL",
    "jarvis-mqtt-broker": "JARVIS_MQTT_BROKER_URL",
}

_initialized = False


def init(db_engine=None) -> bool:
    """
    Initialize service configuration.

    Fetches service URLs from jarvis-config-service and caches them.

    Args:
        db_engine: Optional SQLAlchemy engine for persistent caching

    Returns:
        True if fetch succeeded, False if using cached/fallback data

    Raises:
        ValueError: If JARVIS_CONFIG_URL is not set
    """
    global _initialized

    config_url = os.getenv("JARVIS_CONFIG_URL")
    if not config_url:
        raise ValueError(
            "JARVIS_CONFIG_URL environment variable is required for service discovery"
        )

    success = config_init(
        config_url=config_url,
        refresh_interval_seconds=300,  # 5 minutes
        db_engine=db_engine,
    )

    _initialized = True

    if success:
        services = get_all_services()
        logger.info(f"Service config initialized with {len(services)} services")
    else:
        logger.warning("Service config initialized with cached/fallback data")

    return success


def shutdown() -> None:
    """Shutdown service configuration."""
    global _initialized
    config_shutdown()
    _initialized = False


def is_initialized() -> bool:
    """Check if service config is initialized."""
    return _initialized


def _get_url(service_name: str) -> str:
    """
    Get URL for a service with fallback chain.

    Priority:
    1. Config service (jarvis-config-service)
    2. Environment variable (legacy)
    3. Hardcoded default

    Args:
        service_name: Service name (e.g., "jarvis-llm-proxy")

    Returns:
        Service URL

    Raises:
        RuntimeError: If init() hasn't been called
    """
    if not _initialized:
        raise RuntimeError(
            "Service config not initialized. Call init() first."
        )

    # Try config service first
    url = get_service_url(service_name)
    if url:
        return url

    # Fall back to env var
    env_var = _ENV_VAR_FALLBACKS.get(service_name)
    if env_var:
        env_url = os.getenv(env_var)
        if env_url:
            logger.debug(f"Using env var {env_var} for {service_name}: {env_url}")
            return env_url

    # Fall back to default
    default = _DEFAULTS.get(service_name)
    if default:
        logger.debug(f"Using default for {service_name}: {default}")
        return default

    raise ValueError(f"No URL found for service: {service_name}")


def get_llm_proxy_url() -> str:
    """Get LLM proxy service URL."""
    return _get_url("jarvis-llm-proxy")


def get_auth_url() -> str:
    """Get auth service URL."""
    return _get_url("jarvis-auth")


def get_logs_url() -> str:
    """Get logs service URL."""
    return _get_url("jarvis-logs")


def get_whisper_url() -> str:
    """Get whisper service URL."""
    return _get_url("jarvis-whisper")


def get_tts_url() -> str:
    """Get TTS service URL."""
    return _get_url("jarvis-tts")


def get_mqtt_broker_url() -> str:
    """Get MQTT broker URL."""
    return _get_url("jarvis-mqtt-broker")

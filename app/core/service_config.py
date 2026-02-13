"""Service URL discovery via jarvis-config-client."""

import logging

from jarvis_config_client import (
    init as config_init,
    shutdown as config_shutdown,
    get_all_services,
    get_auth_url,
    get_llm_proxy_url,
    get_logs_url,
    get_mqtt_broker_url,
    get_tts_url,
    get_whisper_url,
)

logger = logging.getLogger(__name__)

# Re-export for convenience
get_auth_url = get_auth_url
get_llm_proxy_url = get_llm_proxy_url
get_logs_url = get_logs_url
get_whisper_url = get_whisper_url
get_tts_url = get_tts_url
get_mqtt_broker_url = get_mqtt_broker_url

_initialized: bool = False


def init(db_engine: object | None = None) -> bool:
    """Initialize service discovery. Call at startup."""
    global _initialized

    success = config_init(
        refresh_interval_seconds=300,
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
    """Shutdown service discovery."""
    global _initialized
    config_shutdown()
    _initialized = False


def is_initialized() -> bool:
    """Check if service config is initialized."""
    return _initialized

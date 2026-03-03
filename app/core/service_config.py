"""Service configuration via jarvis-config-client.

Provides centralized access to service URLs fetched from jarvis-config-service.
Falls back to environment variables when the config client or config service
is unavailable. No hardcoded defaults - config-service is the source of truth.
"""

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_WARNING_BANNER = """
\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
\u2551  JARVIS_CONFIG_URL IS NOT SET                              \u2551
\u2551                                                            \u2551
\u2551  This service is running WITHOUT service discovery.        \u2551
\u2551  URLs will fall back to env vars or FAIL.                  \u2551
\u2551                                                            \u2551
\u2551  Fix: Set JARVIS_CONFIG_URL=http://<config-host>:7700      \u2551
\u2551  Or register this service via jarvis-admin Services page   \u2551
\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d
"""

# Legacy env var names for fallback (with warning)
_ENV_VAR_FALLBACKS: dict[str, str] = {
    "jarvis-auth": "JARVIS_AUTH_BASE_URL",
    "jarvis-llm-proxy-api": "JARVIS_LLM_PROXY_API_URL",
    "jarvis-logs": "JARVIS_LOGS_URL",
    "jarvis-whisper-api": "JARVIS_WHISPER_URL",
    "jarvis-tts": "JARVIS_TTS_URL",
    "jarvis-mqtt-broker": "JARVIS_MQTT_BROKER_URL",
}

_initialized: bool = False
_has_config_client: bool = False
_config_url_set: bool = False
_nag_thread: threading.Thread | None = None

try:
    from jarvis_config_client import (
        init as config_init,
        shutdown as config_shutdown,
        get_service_url,
        get_all_services,
    )
    _has_config_client = True
except ImportError:
    _has_config_client = False


def _nag_loop() -> None:
    """Print warning every 30s until config URL is set."""
    while not _config_url_set:
        logger.warning(_WARNING_BANNER)
        time.sleep(30)


def init(db_engine: object | None = None) -> bool:
    """Initialize service configuration.

    Returns True if config-service fetch succeeded, False if using fallbacks.
    """
    global _initialized, _config_url_set, _nag_thread

    if not _has_config_client:
        logger.info("jarvis-config-client not installed, using env var fallbacks")
        _initialized = True
        return False

    config_url = os.getenv("JARVIS_CONFIG_URL")
    if not config_url:
        logger.warning(_WARNING_BANNER)
        _nag_thread = threading.Thread(target=_nag_loop, daemon=True)
        _nag_thread.start()
        _initialized = True
        return False

    _config_url_set = True

    success = config_init(
        config_url=config_url,
        refresh_interval_seconds=300,
        db_engine=db_engine,
    )

    _initialized = True

    if success:
        services = get_all_services()
        logger.info("Service config initialized with %d services", len(services))
    else:
        logger.warning("Service config initialized with cached/fallback data")

    return success


def shutdown() -> None:
    """Shutdown service configuration."""
    global _initialized, _config_url_set
    if _has_config_client:
        config_shutdown()
    _config_url_set = True  # Stop nag thread
    _initialized = False


def is_initialized() -> bool:
    """Check if service config is initialized."""
    return _initialized


def _get_url(service_name: str) -> str:
    """Get URL for a service with fallback chain.

    Priority:
    1. Config service (jarvis-config-service)
    2. Environment variable (legacy, with warning)
    3. No default - raise clear error
    """
    # Try config service first
    if _has_config_client and _initialized:
        url = get_service_url(service_name)
        if url:
            return url

    # Fall back to env var (with warning)
    env_var = _ENV_VAR_FALLBACKS.get(service_name)
    if env_var:
        env_url = os.getenv(env_var)
        if env_url:
            logger.warning(
                "Using legacy env var %s for %s. "
                "Consider registering in config-service instead.",
                env_var, service_name,
            )
            return env_url

    # No default - raise clear error
    fallback_hint = _ENV_VAR_FALLBACKS.get(service_name, "N/A")
    raise ValueError(
        f"Cannot discover {service_name}. "
        f"Set JARVIS_CONFIG_URL or {fallback_hint}"
    )


def get_auth_url() -> str:
    """Get auth service URL."""
    return _get_url("jarvis-auth")


def get_llm_proxy_url() -> str:
    """Get LLM proxy service URL."""
    return _get_url("jarvis-llm-proxy-api")


def get_logs_url() -> str:
    """Get logs service URL."""
    return _get_url("jarvis-logs")


def get_whisper_url() -> str:
    """Get whisper service URL."""
    return _get_url("jarvis-whisper-api")


def get_tts_url() -> str:
    """Get TTS service URL."""
    return _get_url("jarvis-tts")


def get_mqtt_broker_url() -> str:
    """Get MQTT broker URL."""
    return _get_url("jarvis-mqtt-broker")

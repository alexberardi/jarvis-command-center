"""
Tests for service configuration via jarvis-config-client.

TDD: Write tests first, then implement.
"""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestServiceConfig:
    """Tests for the service_config module."""

    def setup_method(self):
        """Clear any cached config before each test."""
        # Clear the module cache to ensure fresh imports
        import sys
        modules_to_remove = [k for k in sys.modules if k.startswith('app.core.service_config')]
        for mod in modules_to_remove:
            del sys.modules[mod]

    def teardown_method(self):
        """Clean up after each test."""
        # Reset any global state
        try:
            from app.core import service_config
            service_config.shutdown()
        except Exception:
            pass

    def test_init_with_config_url_env_var(self):
        """Test that init() uses JARVIS_CONFIG_URL environment variable."""
        mock_services = {
            "services": [
                {"name": "jarvis-llm-proxy", "host": "localhost", "port": 7704, "url": "http://localhost:7704", "health_path": "/health"},
                {"name": "jarvis-auth", "host": "localhost", "port": 7701, "url": "http://localhost:7701", "health_path": "/health"},
            ]
        }

        with patch.dict(os.environ, {"JARVIS_CONFIG_URL": "http://localhost:7700"}):
            with patch("jarvis_config_client.client.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_services
                mock_response.raise_for_status = MagicMock()
                mock_client.return_value.__enter__.return_value.get.return_value = mock_response

                from app.core import service_config
                result = service_config.init()

                assert result is True

    def test_init_without_config_url_warns_and_returns_false(self):
        """Missing JARVIS_CONFIG_URL is non-fatal: init() logs the nag banner
        and returns False so services can still boot in fallback-only mode."""
        env = {k: v for k, v in os.environ.items() if k != "JARVIS_CONFIG_URL"}

        with patch.dict(os.environ, env, clear=True):
            from app.core import service_config

            assert service_config.init() is False
            assert service_config.is_initialized() is True

    def test_get_llm_proxy_url(self):
        """Test getting LLM proxy URL from config service."""
        mock_services = {
            "services": [
                {"name": "jarvis-llm-proxy-api", "host": "gpu-server", "port": 7704, "url": "http://gpu-server:7704", "health_path": "/health"},
            ]
        }

        with patch.dict(os.environ, {"JARVIS_CONFIG_URL": "http://localhost:7700"}):
            with patch("jarvis_config_client.client.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_services
                mock_response.raise_for_status = MagicMock()
                mock_client.return_value.__enter__.return_value.get.return_value = mock_response

                from app.core import service_config
                service_config.init()

                url = service_config.get_llm_proxy_url()
                assert url == "http://gpu-server:7704"

    def test_get_auth_url(self):
        """Test getting auth service URL from config service."""
        mock_services = {
            "services": [
                {"name": "jarvis-auth", "host": "auth-host", "port": 7701, "url": "http://auth-host:7701", "health_path": "/health"},
            ]
        }

        with patch.dict(os.environ, {"JARVIS_CONFIG_URL": "http://localhost:7700"}):
            with patch("jarvis_config_client.client.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_services
                mock_response.raise_for_status = MagicMock()
                mock_client.return_value.__enter__.return_value.get.return_value = mock_response

                from app.core import service_config
                service_config.init()

                url = service_config.get_auth_url()
                assert url == "http://auth-host:7701"

    def test_get_logs_url(self):
        """Test getting logs service URL from config service."""
        mock_services = {
            "services": [
                {"name": "jarvis-logs", "host": "logs-host", "port": 7702, "url": "http://logs-host:7702", "health_path": "/health"},
            ]
        }

        with patch.dict(os.environ, {"JARVIS_CONFIG_URL": "http://localhost:7700"}):
            with patch("jarvis_config_client.client.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_services
                mock_response.raise_for_status = MagicMock()
                mock_client.return_value.__enter__.return_value.get.return_value = mock_response

                from app.core import service_config
                service_config.init()

                url = service_config.get_logs_url()
                assert url == "http://logs-host:7702"

    def test_fallback_to_env_var_when_service_not_found(self):
        """Test that we fall back to env vars when service not in config."""
        mock_services = {"services": []}  # Empty - no services registered

        with patch.dict(os.environ, {
            "JARVIS_CONFIG_URL": "http://localhost:7700",
            "JARVIS_LLM_PROXY_API_URL": "http://fallback:7704",
        }):
            with patch("jarvis_config_client.client.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_services
                mock_response.raise_for_status = MagicMock()
                mock_client.return_value.__enter__.return_value.get.return_value = mock_response

                from app.core import service_config
                service_config.init()

                # Should fall back to env var
                url = service_config.get_llm_proxy_url()
                assert url == "http://fallback:7704"

    def test_raises_when_no_env_var_and_not_in_config(self):
        """When the service isn't in config-service AND no env var is set,
        the lookup raises rather than guessing a hardcoded default — config
        service is the source of truth."""
        mock_services = {"services": []}  # Empty - no services registered

        # Remove the fallback env var
        env = {k: v for k, v in os.environ.items() if k != "JARVIS_LLM_PROXY_API_URL"}
        env["JARVIS_CONFIG_URL"] = "http://localhost:7700"

        with patch.dict(os.environ, env, clear=True):
            with patch("jarvis_config_client.client.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_services
                mock_response.raise_for_status = MagicMock()
                mock_client.return_value.__enter__.return_value.get.return_value = mock_response

                from app.core import service_config
                service_config.init()

                with pytest.raises(ValueError, match="jarvis-llm-proxy-api"):
                    service_config.get_llm_proxy_url()

    def test_get_url_before_init_raises(self):
        """Test that getting URL before init raises (with no config + no env var)."""
        env = {k: v for k, v in os.environ.items() if k != "JARVIS_LLM_PROXY_API_URL"}
        with patch.dict(os.environ, env, clear=True):
            from app.core import service_config

            with pytest.raises(ValueError, match="jarvis-llm-proxy-api"):
                service_config.get_llm_proxy_url()

    def test_is_initialized(self):
        """Test checking if service config is initialized."""
        mock_services = {"services": []}

        with patch.dict(os.environ, {"JARVIS_CONFIG_URL": "http://localhost:7700"}):
            with patch("jarvis_config_client.client.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_services
                mock_response.raise_for_status = MagicMock()
                mock_client.return_value.__enter__.return_value.get.return_value = mock_response

                from app.core import service_config

                assert service_config.is_initialized() is False
                service_config.init()
                assert service_config.is_initialized() is True

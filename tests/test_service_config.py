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
                {"name": "jarvis-llm-proxy", "host": "localhost", "port": 8000, "url": "http://localhost:8000", "health_path": "/health"},
                {"name": "jarvis-auth", "host": "localhost", "port": 8007, "url": "http://localhost:8007", "health_path": "/health"},
            ]
        }

        with patch.dict(os.environ, {"JARVIS_CONFIG_URL": "http://localhost:8013"}):
            with patch("jarvis_config_client.client.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_services
                mock_response.raise_for_status = MagicMock()
                mock_client.return_value.__enter__.return_value.get.return_value = mock_response

                from app.core import service_config
                result = service_config.init()

                assert result is True

    def test_init_without_config_url_raises(self):
        """Test that init() raises if JARVIS_CONFIG_URL is not set."""
        env = {k: v for k, v in os.environ.items() if k != "JARVIS_CONFIG_URL"}

        with patch.dict(os.environ, env, clear=True):
            from app.core import service_config

            with pytest.raises(ValueError, match="JARVIS_CONFIG_URL"):
                service_config.init()

    def test_get_llm_proxy_url(self):
        """Test getting LLM proxy URL from config service."""
        mock_services = {
            "services": [
                {"name": "jarvis-llm-proxy", "host": "gpu-server", "port": 8000, "url": "http://gpu-server:8000", "health_path": "/health"},
            ]
        }

        with patch.dict(os.environ, {"JARVIS_CONFIG_URL": "http://localhost:8013"}):
            with patch("jarvis_config_client.client.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_services
                mock_response.raise_for_status = MagicMock()
                mock_client.return_value.__enter__.return_value.get.return_value = mock_response

                from app.core import service_config
                service_config.init()

                url = service_config.get_llm_proxy_url()
                assert url == "http://gpu-server:8000"

    def test_get_auth_url(self):
        """Test getting auth service URL from config service."""
        mock_services = {
            "services": [
                {"name": "jarvis-auth", "host": "auth-host", "port": 8007, "url": "http://auth-host:8007", "health_path": "/health"},
            ]
        }

        with patch.dict(os.environ, {"JARVIS_CONFIG_URL": "http://localhost:8013"}):
            with patch("jarvis_config_client.client.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_services
                mock_response.raise_for_status = MagicMock()
                mock_client.return_value.__enter__.return_value.get.return_value = mock_response

                from app.core import service_config
                service_config.init()

                url = service_config.get_auth_url()
                assert url == "http://auth-host:8007"

    def test_get_logs_url(self):
        """Test getting logs service URL from config service."""
        mock_services = {
            "services": [
                {"name": "jarvis-logs", "host": "logs-host", "port": 8006, "url": "http://logs-host:8006", "health_path": "/health"},
            ]
        }

        with patch.dict(os.environ, {"JARVIS_CONFIG_URL": "http://localhost:8013"}):
            with patch("jarvis_config_client.client.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_services
                mock_response.raise_for_status = MagicMock()
                mock_client.return_value.__enter__.return_value.get.return_value = mock_response

                from app.core import service_config
                service_config.init()

                url = service_config.get_logs_url()
                assert url == "http://logs-host:8006"

    def test_fallback_to_env_var_when_service_not_found(self):
        """Test that we fall back to env vars when service not in config."""
        mock_services = {"services": []}  # Empty - no services registered

        with patch.dict(os.environ, {
            "JARVIS_CONFIG_URL": "http://localhost:8013",
            "JARVIS_LLM_PROXY_API_URL": "http://fallback:8000",
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
                assert url == "http://fallback:8000"

    def test_fallback_to_default_when_no_env_var(self):
        """Test that we fall back to defaults when no env var either."""
        mock_services = {"services": []}  # Empty - no services registered

        # Remove the fallback env var
        env = {k: v for k, v in os.environ.items() if k != "JARVIS_LLM_PROXY_API_URL"}
        env["JARVIS_CONFIG_URL"] = "http://localhost:8013"

        with patch.dict(os.environ, env, clear=True):
            with patch("jarvis_config_client.client.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_services
                mock_response.raise_for_status = MagicMock()
                mock_client.return_value.__enter__.return_value.get.return_value = mock_response

                from app.core import service_config
                service_config.init()

                # Should fall back to hardcoded default
                url = service_config.get_llm_proxy_url()
                assert url == "http://localhost:8000"

    def test_get_url_before_init_raises(self):
        """Test that getting URL before init raises RuntimeError."""
        from app.core import service_config

        with pytest.raises(RuntimeError, match="not initialized"):
            service_config.get_llm_proxy_url()

    def test_is_initialized(self):
        """Test checking if service config is initialized."""
        mock_services = {"services": []}

        with patch.dict(os.environ, {"JARVIS_CONFIG_URL": "http://localhost:8013"}):
            with patch("jarvis_config_client.client.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_services
                mock_response.raise_for_status = MagicMock()
                mock_client.return_value.__enter__.return_value.get.return_value = mock_response

                from app.core import service_config

                assert service_config.is_initialized() is False
                service_config.init()
                assert service_config.is_initialized() is True

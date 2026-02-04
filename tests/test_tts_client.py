"""Tests for TTS client with app-to-app auth and context headers.

TDD: These tests define the expected behavior before implementation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Note: Import will fail until we implement the client (RED phase)


class TestTTSClientInit:
    """Tests for TTSClient initialization."""

    def test_init_stores_household_id(self, test_db):
        """Client stores household_id for context headers."""
        from app.core.clients.tts_client import TTSClient

        client = TTSClient(db=test_db, household_id="h123")
        assert client.household_id == "h123"

    def test_init_stores_optional_node_id(self, test_db):
        """Client stores optional node_id for context headers."""
        from app.core.clients.tts_client import TTSClient

        client = TTSClient(db=test_db, household_id="h123", node_id="kitchen-pi")
        assert client.node_id == "kitchen-pi"

    def test_init_stores_optional_user_id(self, test_db):
        """Client stores optional user_id for context headers."""
        from app.core.clients.tts_client import TTSClient

        client = TTSClient(db=test_db, household_id="h123", user_id=42)
        assert client.user_id == 42

    def test_init_gets_base_url_from_settings(self, test_db):
        """Client gets base URL from settings service with cascade lookup."""
        from app.core.clients.tts_client import TTSClient
        from app.services.settings_service import SettingsService

        # Set up a TTS URL in settings
        settings = SettingsService(test_db)
        settings.set_setting("tts_url", "http://custom-tts:8009")

        client = TTSClient(db=test_db, household_id="h123")
        assert client.base_url == "http://custom-tts:8009"

    def test_init_uses_default_url_if_no_setting(self, test_db):
        """Client uses default URL if no setting found."""
        from app.core.clients.tts_client import TTSClient

        client = TTSClient(db=test_db, household_id="h123")
        assert client.base_url == "http://localhost:8009"

    def test_init_cascade_node_overrides_household(self, test_db):
        """Node-specific URL overrides household URL."""
        from app.core.clients.tts_client import TTSClient
        from app.services.settings_service import SettingsService

        settings = SettingsService(test_db)
        settings.set_setting("tts_url", "http://household-tts:8009", household_id="h123")
        settings.set_setting("tts_url", "http://node-tts:8009", household_id="h123", node_id="kitchen-pi")

        client = TTSClient(db=test_db, household_id="h123", node_id="kitchen-pi")
        assert client.base_url == "http://node-tts:8009"


class TestTTSClientHeaders:
    """Tests for header building."""

    def test_build_headers_includes_app_auth(self, test_db):
        """Headers include app-to-app auth headers."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_app_headers") as mock_get_app:
            mock_get_app.return_value = {
                "X-Jarvis-App-Id": "command-center",
                "X-Jarvis-App-Key": "secret123",
            }

            client = TTSClient(db=test_db, household_id="h123")
            headers = client._build_headers()

            assert headers["X-Jarvis-App-Id"] == "command-center"
            assert headers["X-Jarvis-App-Key"] == "secret123"

    def test_build_headers_includes_context_household(self, test_db):
        """Headers include household context."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_app_headers") as mock_get_app:
            mock_get_app.return_value = {}

            client = TTSClient(db=test_db, household_id="h123")
            headers = client._build_headers()

            assert headers["X-Context-Household-Id"] == "h123"

    def test_build_headers_includes_context_node_when_set(self, test_db):
        """Headers include node context when provided."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_app_headers") as mock_get_app:
            mock_get_app.return_value = {}

            client = TTSClient(db=test_db, household_id="h123", node_id="kitchen-pi")
            headers = client._build_headers()

            assert headers["X-Context-Node-Id"] == "kitchen-pi"

    def test_build_headers_includes_context_user_when_set(self, test_db):
        """Headers include user context when provided."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_app_headers") as mock_get_app:
            mock_get_app.return_value = {}

            client = TTSClient(db=test_db, household_id="h123", user_id=42)
            headers = client._build_headers()

            assert headers["X-Context-User-Id"] == "42"


class TestTTSClientSpeak:
    """Tests for the speak method."""

    @pytest.mark.asyncio
    async def test_speak_posts_to_correct_endpoint(self, test_db):
        """Speak method posts to /speak endpoint."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_app_headers") as mock_get_app:
            mock_get_app.return_value = {}

            client = TTSClient(db=test_db, household_id="h123")

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.content = b"audio data"
                mock_response.raise_for_status = MagicMock()
                mock_client.post.return_value = mock_response
                mock_client_class.return_value.__aenter__.return_value = mock_client

                result = await client.speak("Hello world")

                mock_client.post.assert_called_once()
                call_args = mock_client.post.call_args
                assert "/speak" in call_args[0][0]
                assert result == b"audio data"

    @pytest.mark.asyncio
    async def test_speak_sends_text_in_body(self, test_db):
        """Speak method sends text in request body."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_app_headers") as mock_get_app:
            mock_get_app.return_value = {}

            client = TTSClient(db=test_db, household_id="h123")

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.content = b"audio data"
                mock_response.raise_for_status = MagicMock()
                mock_client.post.return_value = mock_response
                mock_client_class.return_value.__aenter__.return_value = mock_client

                await client.speak("Hello world")

                call_kwargs = mock_client.post.call_args[1]
                assert call_kwargs["json"]["text"] == "Hello world"

    @pytest.mark.asyncio
    async def test_speak_sends_headers(self, test_db):
        """Speak method sends auth and context headers."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_app_headers") as mock_get_app:
            mock_get_app.return_value = {"X-Jarvis-App-Id": "command-center"}

            client = TTSClient(db=test_db, household_id="h123", node_id="kitchen-pi")

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.content = b"audio data"
                mock_response.raise_for_status = MagicMock()
                mock_client.post.return_value = mock_response
                mock_client_class.return_value.__aenter__.return_value = mock_client

                await client.speak("Hello world")

                call_kwargs = mock_client.post.call_args[1]
                headers = call_kwargs["headers"]
                assert headers["X-Jarvis-App-Id"] == "command-center"
                assert headers["X-Context-Household-Id"] == "h123"
                assert headers["X-Context-Node-Id"] == "kitchen-pi"

    @pytest.mark.asyncio
    async def test_speak_returns_bytes(self, test_db):
        """Speak method returns audio bytes."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_app_headers") as mock_get_app:
            mock_get_app.return_value = {}

            client = TTSClient(db=test_db, household_id="h123")

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.content = b"\x00\x01\x02\x03"  # Binary audio data
                mock_response.raise_for_status = MagicMock()
                mock_client.post.return_value = mock_response
                mock_client_class.return_value.__aenter__.return_value = mock_client

                result = await client.speak("Hello")

                assert isinstance(result, bytes)
                assert result == b"\x00\x01\x02\x03"

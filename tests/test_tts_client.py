"""Tests for TTS client with app-to-app auth and context headers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestTTSClientInit:
    """Tests for TTSClient initialization."""

    def test_init_stores_household_id(self):
        """Client stores household_id for context headers."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            client = TTSClient(household_id="h123")
            assert client.household_id == "h123"

    def test_init_stores_optional_node_id(self):
        """Client stores optional node_id for context headers."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            client = TTSClient(household_id="h123", node_id="kitchen-pi")
            assert client.node_id == "kitchen-pi"

    def test_init_stores_optional_user_id(self):
        """Client stores optional user_id for context headers."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            client = TTSClient(household_id="h123", user_id=42)
            assert client.user_id == 42

    def test_init_gets_base_url_from_settings(self):
        """Client gets base URL from settings service with cascade lookup."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = "http://custom-tts:8009"

            client = TTSClient(household_id="h123")
            assert client.base_url == "http://custom-tts:8009"

    def test_init_uses_default_url_if_no_setting(self):
        """Client uses default URL if no setting found."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            client = TTSClient(household_id="h123")
            assert client.base_url == "http://localhost:8009"


class TestTTSClientHeaders:
    """Tests for header building."""

    def test_build_headers_includes_app_auth(self):
        """Headers include app-to-app auth headers."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.tts_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {
                    "X-Jarvis-App-Id": "command-center",
                    "X-Jarvis-App-Key": "secret123",
                }

                client = TTSClient(household_id="h123")
                headers = client._build_headers()

                assert headers["X-Jarvis-App-Id"] == "command-center"
                assert headers["X-Jarvis-App-Key"] == "secret123"

    def test_build_headers_includes_context_household(self):
        """Headers include household context."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.tts_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {}

                client = TTSClient(household_id="h123")
                headers = client._build_headers()

                assert headers["X-Context-Household-Id"] == "h123"

    def test_build_headers_includes_context_node_when_set(self):
        """Headers include node context when provided."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.tts_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {}

                client = TTSClient(household_id="h123", node_id="kitchen-pi")
                headers = client._build_headers()

                assert headers["X-Context-Node-Id"] == "kitchen-pi"

    def test_build_headers_includes_context_user_when_set(self):
        """Headers include user context when provided."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.tts_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {}

                client = TTSClient(household_id="h123", user_id=42)
                headers = client._build_headers()

                assert headers["X-Context-User-Id"] == "42"


class TestTTSClientSpeak:
    """Tests for the speak method."""

    @pytest.mark.asyncio
    async def test_speak_posts_to_correct_endpoint(self):
        """Speak method posts to /speak endpoint."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.tts_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {}

                client = TTSClient(household_id="h123")

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
    async def test_speak_sends_text_in_body(self):
        """Speak method sends text in request body."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.tts_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {}

                client = TTSClient(household_id="h123")

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
    async def test_speak_sends_headers(self):
        """Speak method sends auth and context headers."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.tts_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {"X-Jarvis-App-Id": "command-center"}

                client = TTSClient(household_id="h123", node_id="kitchen-pi")

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
    async def test_speak_returns_bytes(self):
        """Speak method returns audio bytes."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.tts_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {}

                client = TTSClient(household_id="h123")

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


class TestTTSClientGenerateWakeResponse:
    """Tests for the generate_wake_response method."""

    @pytest.mark.asyncio
    async def test_generate_wake_response_posts_to_correct_endpoint(self):
        """Generate wake response posts to /generate-wake-response."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.tts_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {}

                client = TTSClient(household_id="h123")

                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_response = MagicMock()
                    mock_response.json.return_value = {"text": "Hello there!"}
                    mock_response.raise_for_status = MagicMock()
                    mock_client.post.return_value = mock_response
                    mock_client_class.return_value.__aenter__.return_value = mock_client

                    result = await client.generate_wake_response()

                    mock_client.post.assert_called_once()
                    call_args = mock_client.post.call_args
                    assert "/generate-wake-response" in call_args[0][0]
                    assert result == "Hello there!"

    @pytest.mark.asyncio
    async def test_generate_wake_response_returns_default_on_missing_text(self):
        """Generate wake response returns default if text missing."""
        from app.core.clients.tts_client import TTSClient

        with patch("app.core.clients.tts_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.tts_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {}

                client = TTSClient(household_id="h123")

                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_response = MagicMock()
                    mock_response.json.return_value = {}
                    mock_response.raise_for_status = MagicMock()
                    mock_client.post.return_value = mock_response
                    mock_client_class.return_value.__aenter__.return_value = mock_client

                    result = await client.generate_wake_response()

                    assert result == "Yes?"

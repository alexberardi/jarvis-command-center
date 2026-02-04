"""Tests for Whisper client with app-to-app auth and context headers.

TDD: These tests define the expected behavior before implementation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestWhisperClientInit:
    """Tests for WhisperClient initialization."""

    def test_init_stores_household_id(self, test_db):
        """Client stores household_id for context headers."""
        from app.core.clients.whisper_client import WhisperClient

        client = WhisperClient(db=test_db, household_id="h123")
        assert client.household_id == "h123"

    def test_init_stores_optional_node_id(self, test_db):
        """Client stores optional node_id for context headers."""
        from app.core.clients.whisper_client import WhisperClient

        client = WhisperClient(db=test_db, household_id="h123", node_id="kitchen-pi")
        assert client.node_id == "kitchen-pi"

    def test_init_stores_optional_user_id(self, test_db):
        """Client stores optional user_id for context headers."""
        from app.core.clients.whisper_client import WhisperClient

        client = WhisperClient(db=test_db, household_id="h123", user_id=42)
        assert client.user_id == 42

    def test_init_gets_base_url_from_settings(self, test_db):
        """Client gets base URL from settings service with cascade lookup."""
        from app.core.clients.whisper_client import WhisperClient
        from app.services.settings_service import SettingsService

        # Set up a Whisper URL in settings
        settings = SettingsService(test_db)
        settings.set_setting("whisper_url", "http://custom-whisper:8012")

        client = WhisperClient(db=test_db, household_id="h123")
        assert client.base_url == "http://custom-whisper:8012"

    def test_init_uses_default_url_if_no_setting(self, test_db):
        """Client uses default URL if no setting found."""
        from app.core.clients.whisper_client import WhisperClient

        client = WhisperClient(db=test_db, household_id="h123")
        assert client.base_url == "http://localhost:8012"

    def test_init_cascade_node_overrides_household(self, test_db):
        """Node-specific URL overrides household URL."""
        from app.core.clients.whisper_client import WhisperClient
        from app.services.settings_service import SettingsService

        settings = SettingsService(test_db)
        settings.set_setting("whisper_url", "http://household-whisper:8012", household_id="h123")
        settings.set_setting("whisper_url", "http://node-whisper:8012", household_id="h123", node_id="kitchen-pi")

        client = WhisperClient(db=test_db, household_id="h123", node_id="kitchen-pi")
        assert client.base_url == "http://node-whisper:8012"


class TestWhisperClientHeaders:
    """Tests for header building."""

    def test_build_headers_includes_app_auth(self, test_db):
        """Headers include app-to-app auth headers."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_app_headers") as mock_get_app:
            mock_get_app.return_value = {
                "X-Jarvis-App-Id": "command-center",
                "X-Jarvis-App-Key": "secret123",
            }

            client = WhisperClient(db=test_db, household_id="h123")
            headers = client._build_headers()

            assert headers["X-Jarvis-App-Id"] == "command-center"
            assert headers["X-Jarvis-App-Key"] == "secret123"

    def test_build_headers_includes_context_household(self, test_db):
        """Headers include household context - critical for voice recognition."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_app_headers") as mock_get_app:
            mock_get_app.return_value = {}

            client = WhisperClient(db=test_db, household_id="h123")
            headers = client._build_headers()

            assert headers["X-Context-Household-Id"] == "h123"

    def test_build_headers_includes_context_node_when_set(self, test_db):
        """Headers include node context when provided."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_app_headers") as mock_get_app:
            mock_get_app.return_value = {}

            client = WhisperClient(db=test_db, household_id="h123", node_id="kitchen-pi")
            headers = client._build_headers()

            assert headers["X-Context-Node-Id"] == "kitchen-pi"

    def test_build_headers_includes_context_user_when_set(self, test_db):
        """Headers include user context when provided."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_app_headers") as mock_get_app:
            mock_get_app.return_value = {}

            client = WhisperClient(db=test_db, household_id="h123", user_id=42)
            headers = client._build_headers()

            assert headers["X-Context-User-Id"] == "42"


class TestWhisperClientTranscribe:
    """Tests for the transcribe method."""

    @pytest.mark.asyncio
    async def test_transcribe_posts_to_correct_endpoint(self, test_db):
        """Transcribe method posts to /transcribe endpoint."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_app_headers") as mock_get_app:
            mock_get_app.return_value = {}

            client = WhisperClient(db=test_db, household_id="h123")

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.json.return_value = {"text": "hello world"}
                mock_response.raise_for_status = MagicMock()
                mock_client.post.return_value = mock_response
                mock_client_class.return_value.__aenter__.return_value = mock_client

                result = await client.transcribe(b"audio data", "recording.wav")

                mock_client.post.assert_called_once()
                call_args = mock_client.post.call_args
                assert "/transcribe" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_transcribe_sends_file_as_multipart(self, test_db):
        """Transcribe method sends audio as multipart file."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_app_headers") as mock_get_app:
            mock_get_app.return_value = {}

            client = WhisperClient(db=test_db, household_id="h123")

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.json.return_value = {"text": "hello world"}
                mock_response.raise_for_status = MagicMock()
                mock_client.post.return_value = mock_response
                mock_client_class.return_value.__aenter__.return_value = mock_client

                await client.transcribe(b"audio data", "recording.wav")

                call_kwargs = mock_client.post.call_args[1]
                # Files should be a dict with "file" key containing tuple (filename, content)
                assert "files" in call_kwargs
                assert "file" in call_kwargs["files"]

    @pytest.mark.asyncio
    async def test_transcribe_sends_headers(self, test_db):
        """Transcribe method sends auth and context headers."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_app_headers") as mock_get_app:
            mock_get_app.return_value = {"X-Jarvis-App-Id": "command-center"}

            client = WhisperClient(db=test_db, household_id="h123", node_id="kitchen-pi")

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.json.return_value = {"text": "hello world"}
                mock_response.raise_for_status = MagicMock()
                mock_client.post.return_value = mock_response
                mock_client_class.return_value.__aenter__.return_value = mock_client

                await client.transcribe(b"audio data", "recording.wav")

                call_kwargs = mock_client.post.call_args[1]
                headers = call_kwargs["headers"]
                assert headers["X-Jarvis-App-Id"] == "command-center"
                assert headers["X-Context-Household-Id"] == "h123"
                assert headers["X-Context-Node-Id"] == "kitchen-pi"

    @pytest.mark.asyncio
    async def test_transcribe_returns_dict(self, test_db):
        """Transcribe method returns JSON response as dict."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_app_headers") as mock_get_app:
            mock_get_app.return_value = {}

            client = WhisperClient(db=test_db, household_id="h123")

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "text": "hello world",
                    "speaker_id": "user1",
                    "confidence": 0.95,
                }
                mock_response.raise_for_status = MagicMock()
                mock_client.post.return_value = mock_response
                mock_client_class.return_value.__aenter__.return_value = mock_client

                result = await client.transcribe(b"audio", "test.wav")

                assert isinstance(result, dict)
                assert result["text"] == "hello world"
                assert result["speaker_id"] == "user1"

    @pytest.mark.asyncio
    async def test_transcribe_passes_extra_params(self, test_db):
        """Transcribe passes additional params like language, task."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_app_headers") as mock_get_app:
            mock_get_app.return_value = {}

            client = WhisperClient(db=test_db, household_id="h123")

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.json.return_value = {"text": "hola mundo"}
                mock_response.raise_for_status = MagicMock()
                mock_client.post.return_value = mock_response
                mock_client_class.return_value.__aenter__.return_value = mock_client

                await client.transcribe(
                    b"audio",
                    "test.wav",
                    language="es",
                    task="translate",
                )

                call_kwargs = mock_client.post.call_args[1]
                # Extra params should be sent as form data
                assert "data" in call_kwargs
                assert call_kwargs["data"]["language"] == "es"
                assert call_kwargs["data"]["task"] == "translate"

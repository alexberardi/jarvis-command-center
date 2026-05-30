"""Tests for Whisper client with app-to-app auth and context headers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestWhisperClientInit:
    """Tests for WhisperClient initialization."""

    def test_init_stores_household_id(self):
        """Client stores household_id for context headers."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            client = WhisperClient(household_id="h123")
            assert client.household_id == "h123"

    def test_init_stores_optional_node_id(self):
        """Client stores optional node_id for context headers."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            client = WhisperClient(household_id="h123", node_id="kitchen-pi")
            assert client.node_id == "kitchen-pi"

    def test_init_stores_optional_user_id(self):
        """Client stores optional user_id for context headers."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            client = WhisperClient(household_id="h123", user_id=42)
            assert client.user_id == 42

    def test_init_stores_household_member_ids(self):
        """Client stores household_member_ids for voice recognition."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            client = WhisperClient(household_id="h123", household_member_ids=[1, 2, 3])
            assert client.household_member_ids == [1, 2, 3]

    def test_init_gets_base_url_from_settings(self):
        """Client gets base URL from settings service with cascade lookup."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = "http://custom-whisper:7706"

            client = WhisperClient(household_id="h123")
            assert client.base_url == "http://custom-whisper:7706"

    def test_init_uses_default_url_if_no_setting(self):
        """Client uses default URL if no setting found."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            client = WhisperClient(household_id="h123")
            assert client.base_url == "http://localhost:7706"


class TestWhisperClientHeaders:
    """Tests for header building."""

    def test_build_headers_includes_app_auth(self):
        """Headers include app-to-app auth headers."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.whisper_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {
                    "X-Jarvis-App-Id": "command-center",
                    "X-Jarvis-App-Key": "secret123",
                }

                client = WhisperClient(household_id="h123")
                headers = client._build_headers()

                assert headers["X-Jarvis-App-Id"] == "command-center"
                assert headers["X-Jarvis-App-Key"] == "secret123"

    def test_build_headers_includes_context_household(self):
        """Headers include household context - critical for voice recognition."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.whisper_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {}

                client = WhisperClient(household_id="h123")
                headers = client._build_headers()

                assert headers["X-Context-Household-Id"] == "h123"

    def test_build_headers_includes_context_node_when_set(self):
        """Headers include node context when provided."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.whisper_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {}

                client = WhisperClient(household_id="h123", node_id="kitchen-pi")
                headers = client._build_headers()

                assert headers["X-Context-Node-Id"] == "kitchen-pi"

    def test_build_headers_includes_household_member_ids(self):
        """Headers include household member IDs for voice recognition."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.whisper_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {}

                client = WhisperClient(household_id="h123", household_member_ids=[1, 2, 3])
                headers = client._build_headers()

                assert headers["X-Context-Household-Member-Ids"] == "1,2,3"


class TestWhisperClientTranscribe:
    """Tests for the transcribe method."""

    @pytest.mark.asyncio
    async def test_transcribe_posts_to_correct_endpoint(self):
        """Transcribe method posts to /transcribe endpoint."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.whisper_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {}

                client = WhisperClient(household_id="h123")

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
    async def test_transcribe_sends_file_as_multipart(self):
        """Transcribe method sends audio as multipart file."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.whisper_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {}

                client = WhisperClient(household_id="h123")

                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_response = MagicMock()
                    mock_response.json.return_value = {"text": "hello world"}
                    mock_response.raise_for_status = MagicMock()
                    mock_client.post.return_value = mock_response
                    mock_client_class.return_value.__aenter__.return_value = mock_client

                    await client.transcribe(b"audio data", "recording.wav")

                    call_kwargs = mock_client.post.call_args[1]
                    # Files is a list of (field_name, (filename, content)) tuples
                    # for multipart upload — same shape httpx expects.
                    assert "files" in call_kwargs
                    file_field_names = [entry[0] for entry in call_kwargs["files"]]
                    assert "file" in file_field_names

    @pytest.mark.asyncio
    async def test_transcribe_sends_headers(self):
        """Transcribe method sends auth and context headers."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.whisper_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {"X-Jarvis-App-Id": "command-center"}

                client = WhisperClient(
                    household_id="h123",
                    node_id="kitchen-pi",
                    household_member_ids=[1, 2],
                )

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
                    assert headers["X-Context-Household-Member-Ids"] == "1,2"

    @pytest.mark.asyncio
    async def test_transcribe_returns_dict(self):
        """Transcribe method returns JSON response as dict."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.whisper_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {}

                client = WhisperClient(household_id="h123")

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
    async def test_transcribe_passes_extra_params(self):
        """Transcribe passes additional params like language, task."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None

            with patch("app.core.clients.whisper_client.get_app_headers") as mock_get_app:
                mock_get_app.return_value = {}

                client = WhisperClient(household_id="h123")

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


class TestEnrollVoiceProfileSampleIndex:
    """Coverage for the multi-take enrollment wiring."""

    @pytest.mark.asyncio
    async def test_omits_sample_index_when_none(self):
        """When sample_index is None, the param must not be in the request."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None
            with patch("app.core.clients.whisper_client.get_app_headers", return_value={}):
                client = WhisperClient(household_id="h123")

                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_response = MagicMock()
                    mock_response.json.return_value = {"status": "enrolled"}
                    mock_response.raise_for_status = MagicMock()
                    mock_client.post.return_value = mock_response
                    mock_client_class.return_value.__aenter__.return_value = mock_client

                    await client.enroll_voice_profile(42, b"audio", "x.wav")

                    params = mock_client.post.call_args[1]["params"]
                    assert "sample_index" not in params
                    assert params["user_id"] == "42"

    @pytest.mark.asyncio
    async def test_passes_sample_index_when_provided(self):
        """Explicit sample_index must be forwarded so retakes overwrite cleanly."""
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = None
            with patch("app.core.clients.whisper_client.get_app_headers", return_value={}):
                client = WhisperClient(household_id="h123")

                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_response = MagicMock()
                    mock_response.json.return_value = {"status": "enrolled", "sample_index": 1}
                    mock_response.raise_for_status = MagicMock()
                    mock_client.post.return_value = mock_response
                    mock_client_class.return_value.__aenter__.return_value = mock_client

                    await client.enroll_voice_profile(
                        42, b"audio", "x.wav", sample_index=1,
                    )

                    params = mock_client.post.call_args[1]["params"]
                    assert params["sample_index"] == "1"


class TestVoiceProfileSamplesEndpoints:
    """Coverage for the list/delete sample helpers (multi-take retake)."""

    @pytest.mark.asyncio
    async def test_list_samples_hits_user_samples_url(self):
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = "http://w:7706"
            with patch("app.core.clients.whisper_client.get_app_headers", return_value={}):
                client = WhisperClient(household_id="h123")

                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_response = MagicMock()
                    mock_response.json.return_value = {"samples": []}
                    mock_response.raise_for_status = MagicMock()
                    mock_client.get.return_value = mock_response
                    mock_client_class.return_value.__aenter__.return_value = mock_client

                    await client.list_voice_profile_samples(42)

                    url = mock_client.get.call_args[0][0]
                    assert url == "http://w:7706/voice-profiles/42/samples"
                    assert mock_client.get.call_args[1]["params"]["household_id"] == "h123"

    @pytest.mark.asyncio
    async def test_delete_sample_hits_indexed_url(self):
        from app.core.clients.whisper_client import WhisperClient

        with patch("app.core.clients.whisper_client.get_settings_service") as mock_settings:
            mock_settings.return_value.get.return_value = "http://w:7706"
            with patch("app.core.clients.whisper_client.get_app_headers", return_value={}):
                client = WhisperClient(household_id="h123")

                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_response = MagicMock()
                    mock_response.json.return_value = {"remaining_samples": 2}
                    mock_response.raise_for_status = MagicMock()
                    mock_client.delete.return_value = mock_response
                    mock_client_class.return_value.__aenter__.return_value = mock_client

                    await client.delete_voice_profile_sample(42, 1)

                    url = mock_client.delete.call_args[0][0]
                    assert url == "http://w:7706/voice-profiles/42/samples/1"

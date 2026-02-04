"""Tests for media proxy endpoints (TTS and Whisper).

TDD: These tests define the expected behavior before implementation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.main import app
from app.deps import get_db, verify_api_key


class TestMediaRouterAuth:
    """Tests for authentication on media endpoints."""

    def test_tts_speak_requires_auth(self, client_with_test_db):
        """TTS speak endpoint requires API key."""
        response = client_with_test_db.post(
            "/api/v0/media/tts/speak",
            json={"text": "Hello world"},
        )
        # App has custom validation handler that returns 400 instead of 422
        assert response.status_code == 400  # Missing header

    def test_whisper_transcribe_requires_auth(self, client_with_test_db):
        """Whisper transcribe endpoint requires API key."""
        response = client_with_test_db.post(
            "/api/v0/media/whisper/transcribe",
            files={"file": ("test.wav", b"audio data")},
        )
        # App has custom validation handler that returns 400 instead of 422
        assert response.status_code == 400  # Missing header

    def test_tts_speak_rejects_invalid_key(self, client_with_test_db):
        """TTS speak endpoint rejects invalid API key."""
        response = client_with_test_db.post(
            "/api/v0/media/tts/speak",
            json={"text": "Hello world"},
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 401

    def test_whisper_transcribe_rejects_invalid_key(self, client_with_test_db):
        """Whisper transcribe endpoint rejects invalid API key."""
        response = client_with_test_db.post(
            "/api/v0/media/whisper/transcribe",
            files={"file": ("test.wav", b"audio data")},
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 401


class TestTTSSpeak:
    """Tests for POST /api/v0/media/tts/speak endpoint."""

    def test_speak_proxies_to_tts_client(self, client_with_test_db, test_node_with_household):
        """Speak endpoint creates TTSClient and calls speak method."""
        node, household_id = test_node_with_household

        with patch("app.api.media.TTSClient") as mock_tts_class:
            mock_client = MagicMock()
            mock_client.speak = AsyncMock(return_value=b"audio data")
            mock_tts_class.return_value = mock_client

            # Mock verify_api_key to return node context with household_id
            with patch("app.api.media.verify_api_key") as mock_verify:
                mock_context = MagicMock()
                mock_context.node = node
                mock_context.household_id = household_id
                mock_verify.return_value = mock_context

                # Override the dependency
                app.dependency_overrides[verify_api_key] = lambda: mock_context

                try:
                    response = client_with_test_db.post(
                        "/api/v0/media/tts/speak",
                        json={"text": "Hello world"},
                        headers={"X-API-Key": node.api_key},
                    )

                    assert response.status_code == 200
                    mock_client.speak.assert_called_once_with("Hello world")
                finally:
                    app.dependency_overrides.pop(verify_api_key, None)

    def test_speak_passes_context_to_client(self, client_with_test_db, test_node_with_household):
        """Speak endpoint passes household_id and node_id to TTSClient."""
        node, household_id = test_node_with_household

        with patch("app.api.media.TTSClient") as mock_tts_class:
            mock_client = MagicMock()
            mock_client.speak = AsyncMock(return_value=b"audio data")
            mock_tts_class.return_value = mock_client

            mock_context = MagicMock()
            mock_context.node = node
            mock_context.household_id = household_id
            app.dependency_overrides[verify_api_key] = lambda: mock_context

            try:
                response = client_with_test_db.post(
                    "/api/v0/media/tts/speak",
                    json={"text": "Hello"},
                    headers={"X-API-Key": node.api_key},
                )

                # Verify TTSClient was created with correct context
                call_kwargs = mock_tts_class.call_args[1]
                assert call_kwargs["household_id"] == household_id
                assert call_kwargs["node_id"] == node.node_id
            finally:
                app.dependency_overrides.pop(verify_api_key, None)

    def test_speak_returns_audio_bytes(self, client_with_test_db, test_node_with_household):
        """Speak endpoint returns audio as bytes with correct content type."""
        node, household_id = test_node_with_household

        with patch("app.api.media.TTSClient") as mock_tts_class:
            mock_client = MagicMock()
            mock_client.speak = AsyncMock(return_value=b"\x00\x01\x02\x03")
            mock_tts_class.return_value = mock_client

            mock_context = MagicMock()
            mock_context.node = node
            mock_context.household_id = household_id
            app.dependency_overrides[verify_api_key] = lambda: mock_context

            try:
                response = client_with_test_db.post(
                    "/api/v0/media/tts/speak",
                    json={"text": "Hello"},
                    headers={"X-API-Key": node.api_key},
                )

                assert response.status_code == 200
                assert response.headers["content-type"] == "audio/wav"
                assert response.content == b"\x00\x01\x02\x03"
            finally:
                app.dependency_overrides.pop(verify_api_key, None)

    def test_speak_validates_text_required(self, client_with_test_db, test_node_with_household):
        """Speak endpoint requires text field."""
        node, household_id = test_node_with_household

        mock_context = MagicMock()
        mock_context.node = node
        mock_context.household_id = household_id
        app.dependency_overrides[verify_api_key] = lambda: mock_context

        try:
            response = client_with_test_db.post(
                "/api/v0/media/tts/speak",
                json={},
                headers={"X-API-Key": node.api_key},
            )
            # App has custom validation handler that returns 400 instead of 422
            assert response.status_code == 400  # Validation error
        finally:
            app.dependency_overrides.pop(verify_api_key, None)


class TestWhisperTranscribe:
    """Tests for POST /api/v0/media/whisper/transcribe endpoint."""

    def test_transcribe_proxies_to_whisper_client(
        self, client_with_test_db, test_node_with_household
    ):
        """Transcribe endpoint creates WhisperClient and calls transcribe method."""
        node, household_id = test_node_with_household

        with patch("app.api.media.WhisperClient") as mock_whisper_class:
            mock_client = MagicMock()
            mock_client.transcribe = AsyncMock(
                return_value={"text": "hello world", "confidence": 0.95}
            )
            mock_whisper_class.return_value = mock_client

            mock_context = MagicMock()
            mock_context.node = node
            mock_context.household_id = household_id
            app.dependency_overrides[verify_api_key] = lambda: mock_context

            try:
                response = client_with_test_db.post(
                    "/api/v0/media/whisper/transcribe",
                    files={"file": ("recording.wav", b"audio data")},
                    headers={"X-API-Key": node.api_key},
                )

                assert response.status_code == 200
                mock_client.transcribe.assert_called_once()
            finally:
                app.dependency_overrides.pop(verify_api_key, None)

    def test_transcribe_passes_context_to_client(
        self, client_with_test_db, test_node_with_household
    ):
        """Transcribe endpoint passes household_id and node_id to WhisperClient."""
        node, household_id = test_node_with_household

        with patch("app.api.media.WhisperClient") as mock_whisper_class:
            mock_client = MagicMock()
            mock_client.transcribe = AsyncMock(return_value={"text": "hello"})
            mock_whisper_class.return_value = mock_client

            mock_context = MagicMock()
            mock_context.node = node
            mock_context.household_id = household_id
            app.dependency_overrides[verify_api_key] = lambda: mock_context

            try:
                response = client_with_test_db.post(
                    "/api/v0/media/whisper/transcribe",
                    files={"file": ("recording.wav", b"audio data")},
                    headers={"X-API-Key": node.api_key},
                )

                # Verify WhisperClient was created with correct context
                call_kwargs = mock_whisper_class.call_args[1]
                assert call_kwargs["household_id"] == household_id
                assert call_kwargs["node_id"] == node.node_id
            finally:
                app.dependency_overrides.pop(verify_api_key, None)

    def test_transcribe_passes_audio_and_filename(
        self, client_with_test_db, test_node_with_household
    ):
        """Transcribe endpoint passes audio bytes and filename to client."""
        node, household_id = test_node_with_household

        with patch("app.api.media.WhisperClient") as mock_whisper_class:
            mock_client = MagicMock()
            mock_client.transcribe = AsyncMock(return_value={"text": "hello"})
            mock_whisper_class.return_value = mock_client

            mock_context = MagicMock()
            mock_context.node = node
            mock_context.household_id = household_id
            app.dependency_overrides[verify_api_key] = lambda: mock_context

            try:
                response = client_with_test_db.post(
                    "/api/v0/media/whisper/transcribe",
                    files={"file": ("my_recording.wav", b"audio data here")},
                    headers={"X-API-Key": node.api_key},
                )

                # Verify transcribe was called with correct args
                call_args = mock_client.transcribe.call_args
                assert call_args[0][0] == b"audio data here"  # audio bytes
                assert call_args[0][1] == "my_recording.wav"  # filename
            finally:
                app.dependency_overrides.pop(verify_api_key, None)

    def test_transcribe_returns_json(self, client_with_test_db, test_node_with_household):
        """Transcribe endpoint returns JSON response."""
        node, household_id = test_node_with_household

        with patch("app.api.media.WhisperClient") as mock_whisper_class:
            mock_client = MagicMock()
            mock_client.transcribe = AsyncMock(
                return_value={
                    "text": "hello world",
                    "speaker_id": "user1",
                    "confidence": 0.95,
                }
            )
            mock_whisper_class.return_value = mock_client

            mock_context = MagicMock()
            mock_context.node = node
            mock_context.household_id = household_id
            app.dependency_overrides[verify_api_key] = lambda: mock_context

            try:
                response = client_with_test_db.post(
                    "/api/v0/media/whisper/transcribe",
                    files={"file": ("test.wav", b"audio")},
                    headers={"X-API-Key": node.api_key},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["text"] == "hello world"
                assert data["speaker_id"] == "user1"
                assert data["confidence"] == 0.95
            finally:
                app.dependency_overrides.pop(verify_api_key, None)

    def test_transcribe_passes_extra_params(
        self, client_with_test_db, test_node_with_household
    ):
        """Transcribe endpoint passes additional parameters like language."""
        node, household_id = test_node_with_household

        with patch("app.api.media.WhisperClient") as mock_whisper_class:
            mock_client = MagicMock()
            mock_client.transcribe = AsyncMock(return_value={"text": "hola mundo"})
            mock_whisper_class.return_value = mock_client

            mock_context = MagicMock()
            mock_context.node = node
            mock_context.household_id = household_id
            app.dependency_overrides[verify_api_key] = lambda: mock_context

            try:
                response = client_with_test_db.post(
                    "/api/v0/media/whisper/transcribe",
                    files={"file": ("test.wav", b"audio")},
                    data={"language": "es", "task": "translate"},
                    headers={"X-API-Key": node.api_key},
                )

                assert response.status_code == 200
                # Verify extra params were passed to transcribe
                call_kwargs = mock_client.transcribe.call_args[1]
                assert call_kwargs.get("language") == "es"
                assert call_kwargs.get("task") == "translate"
            finally:
                app.dependency_overrides.pop(verify_api_key, None)

    def test_transcribe_requires_file(self, client_with_test_db, test_node_with_household):
        """Transcribe endpoint requires file upload."""
        node, household_id = test_node_with_household

        mock_context = MagicMock()
        mock_context.node = node
        mock_context.household_id = household_id
        app.dependency_overrides[verify_api_key] = lambda: mock_context

        try:
            response = client_with_test_db.post(
                "/api/v0/media/whisper/transcribe",
                headers={"X-API-Key": node.api_key},
            )
            # App has custom validation handler that returns 400 instead of 422
            assert response.status_code == 400  # Validation error
        finally:
            app.dependency_overrides.pop(verify_api_key, None)

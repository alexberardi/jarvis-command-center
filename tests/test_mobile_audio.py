"""Tests for mobile audio endpoints (app/api/mobile_audio.py)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _FakeUpload:
    """Minimal stand-in for a Starlette UploadFile."""

    filename = "audio.wav"

    async def read(self) -> bytes:
        return b"audio-bytes"


@pytest.mark.asyncio
async def test_mobile_stt_skips_voice_recognition_and_stamps_jwt_speaker():
    """Mobile /stt must skip Whisper's voice pass and attribute the JWT user.

    On a personal device the speaker is the authenticated user, not a voice
    match — and a memberless voice request is what poisoned the shared whisper
    profile cache in prod. So /stt opts out of the speaker pass and stamps the
    speaker from the JWT deterministically.
    """
    from app.api import mobile_audio

    fake_client = MagicMock()
    fake_client.transcribe = AsyncMock(
        return_value={
            "text": "turn on the lights",
            "speaker": {"user_id": None, "confidence": 0.0},
        }
    )

    with patch.object(mobile_audio, "WhisperClient", return_value=fake_client), patch.object(
        mobile_audio, "verify_household_role"
    ):
        result = await mobile_audio.mobile_stt(
            file=_FakeUpload(),
            household_id="hh-1",
            language=None,
            user=SimpleNamespace(user_id=7),
        )

    # The voice speaker-recognition pass is explicitly disabled for mobile.
    assert fake_client.transcribe.await_args.kwargs["speaker_recognition"] is False
    # Speaker attributed from auth (the JWT user), not from voice matching.
    assert result["raw"]["speaker"] == {
        "user_id": 7,
        "confidence": 1.0,
        "source": "jwt",
    }
    assert result["text"] == "turn on the lights"

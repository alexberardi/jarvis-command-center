"""Mobile audio endpoints: STT and TTS with JWT auth.

These mirror the node-authenticated media endpoints but use JWT auth
for mobile app access.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, UploadFile, File, Form, Response
from pydantic import BaseModel

from app.core.clients import TTSClient, WhisperClient
from app.core.tts_text import clean_for_tts
from app.deps import verify_user_jwt, verify_household_role, AuthenticatedUser

logger = logging.getLogger("uvicorn")

router = APIRouter(tags=["mobile-audio"])


class TTSRequest(BaseModel):
    """Request body for TTS endpoint."""

    text: str
    household_id: str


@router.post("/stt")
async def mobile_stt(
    file: UploadFile = File(...),
    household_id: str = Form(...),
    language: str | None = Form(None),
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict[str, Any]:
    """Transcribe audio to text.

    Accepts an audio file upload and returns the transcribed text.
    Uses JWT auth instead of node X-API-Key auth.
    """
    verify_household_role(user.user_id, household_id, required_role="member")

    client = WhisperClient(
        household_id=household_id,
        user_id=user.user_id,
    )

    audio_bytes = await file.read()
    filename = file.filename or "audio.wav"

    params: dict[str, Any] = {}
    if language:
        params["language"] = language

    result = await client.transcribe(audio_bytes, filename, **params)
    return {"text": result.get("text", ""), "raw": result}


@router.post("/tts")
async def mobile_tts(
    request: TTSRequest,
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> Response:
    """Convert text to speech.

    Returns audio bytes as WAV format.
    """
    verify_household_role(user.user_id, request.household_id, required_role="member")

    client = TTSClient(
        household_id=request.household_id,
        user_id=user.user_id,
    )

    audio = await client.speak(clean_for_tts(request.text))
    return Response(content=audio, media_type="audio/wav")

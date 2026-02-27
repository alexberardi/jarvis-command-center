"""Media proxy endpoints for TTS and Whisper services.

These endpoints allow authenticated nodes to access TTS and Whisper services
through command-center, which handles app-to-app authentication and passes
context headers (household_id, node_id) to downstream services.
"""

from typing import Any

from fastapi import APIRouter, Depends, UploadFile, File, Form, Response
from pydantic import BaseModel

from app.deps import verify_api_key
from app.context_providers.node_context_provider import NodeContextProvider
from app.core.clients import TTSClient, WhisperClient

router = APIRouter(prefix="/media", tags=["media"])


class TTSSpeakRequest(BaseModel):
    """Request body for TTS speak endpoint."""

    text: str


@router.post("/tts/speak")
async def tts_speak(
    request: TTSSpeakRequest,
    node_context: NodeContextProvider = Depends(verify_api_key),
) -> Response:
    """Convert text to speech.

    Proxies the request to the TTS service with app-to-app auth
    and context headers for household/node identification.

    Args:
        request: Request body containing text to speak
        node_context: Authenticated node context (from verify_api_key)

    Returns:
        Audio bytes as WAV with content-type audio/wav
    """
    client = TTSClient(
        household_id=node_context.household_id,
        node_id=node_context.node.node_id,
    )
    audio = await client.speak(request.text)
    return Response(content=audio, media_type="audio/wav")


@router.post("/whisper/transcribe")
async def whisper_transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    task: str | None = Form(None),
    node_context: NodeContextProvider = Depends(verify_api_key),
) -> dict[str, Any]:
    """Transcribe audio to text.

    Proxies the request to the Whisper service with app-to-app auth
    and context headers. The household_id context is critical for
    Whisper's voice recognition feature - it uses it to filter
    voice profiles to the correct household.

    Args:
        file: Audio file to transcribe
        language: Optional language code (e.g., "en", "es")
        task: Optional task ("transcribe" or "translate")
        node_context: Authenticated node context (from verify_api_key)

    Returns:
        Transcription result dict with text, optional speaker_id, etc.
    """
    client = WhisperClient(
        household_id=node_context.household_id,
        node_id=node_context.node.node_id,
        household_member_ids=node_context.household_member_ids,
    )

    # Read file content
    audio_bytes = await file.read()
    filename = file.filename or "audio.wav"

    # Build extra params
    params: dict[str, Any] = {}
    if language:
        params["language"] = language
    if task:
        params["task"] = task

    return await client.transcribe(audio_bytes, filename, **params)


@router.post("/whisper/voice-profiles/enroll")
async def whisper_enroll_voice_profile(
    user_id: int,
    file: UploadFile = File(...),
    node_context: NodeContextProvider = Depends(verify_api_key),
) -> dict[str, Any]:
    """Enroll a voice profile for speaker identification.

    Proxies the request to the Whisper service with app-to-app auth.
    """
    client = WhisperClient(
        household_id=node_context.household_id,
        node_id=node_context.node.node_id,
    )
    audio_bytes = await file.read()
    filename = file.filename or "enrollment.wav"
    return await client.enroll_voice_profile(user_id, audio_bytes, filename)


@router.delete("/whisper/voice-profiles/{user_id}")
async def whisper_delete_voice_profile(
    user_id: int,
    node_context: NodeContextProvider = Depends(verify_api_key),
) -> dict[str, Any]:
    """Delete a voice profile."""
    client = WhisperClient(
        household_id=node_context.household_id,
        node_id=node_context.node.node_id,
    )
    return await client.delete_voice_profile(user_id)


@router.get("/whisper/voice-profiles")
async def whisper_list_voice_profiles(
    node_context: NodeContextProvider = Depends(verify_api_key),
) -> dict[str, Any]:
    """List voice profiles for the node's household."""
    client = WhisperClient(
        household_id=node_context.household_id,
        node_id=node_context.node.node_id,
    )
    return await client.list_voice_profiles()


@router.post("/tts/generate-wake-response")
async def tts_generate_wake_response(
    node_context: NodeContextProvider = Depends(verify_api_key),
) -> dict[str, str]:
    """Generate a dynamic wake response greeting.

    Proxies the request to the TTS service's generate-wake-response endpoint
    with app-to-app auth and context headers.

    Args:
        node_context: Authenticated node context (from verify_api_key)

    Returns:
        Dict with "text" key containing the greeting
    """
    client = TTSClient(
        household_id=node_context.household_id,
        node_id=node_context.node.node_id,
    )
    text = await client.generate_wake_response()
    return {"text": text}

"""Media proxy endpoints for TTS and Whisper services.

These endpoints allow authenticated nodes to access TTS and Whisper services
through command-center, which handles app-to-app authentication and passes
context headers (household_id, node_id) to downstream services.
"""

import asyncio
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, UploadFile, File, Form, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.deps import verify_api_key
from app.context_providers.node_context_provider import NodeContextProvider
from app.core.clients import TTSClient, WhisperClient
from app.core.tts_text import clean_for_tts
from app.core.utils.latency_logger import latency_logger

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
    audio = await client.speak(clean_for_tts(request.text))
    return Response(content=audio, media_type="audio/wav")


@router.post("/tts/speak/stream")
async def tts_speak_stream(
    request: TTSSpeakRequest,
    node_context: NodeContextProvider = Depends(verify_api_key),
) -> StreamingResponse:
    """Stream raw PCM audio from TTS service.

    Proxies to the TTS service's /speak/stream endpoint with app-to-app auth.
    Returns raw PCM audio with format metadata in headers.
    """
    client = TTSClient(
        household_id=node_context.household_id,
        node_id=node_context.node.node_id,
    )
    audio_iter, audio_meta = await client.speak_stream(clean_for_tts(request.text))
    return StreamingResponse(
        audio_iter,
        media_type="audio/raw",
        headers={
            "X-Audio-Sample-Rate": audio_meta["sample_rate"],
            "X-Audio-Channels": audio_meta["channels"],
            "X-Audio-Sample-Width": audio_meta["sample_width"],
        },
    )


@router.post("/whisper/transcribe")
async def whisper_transcribe(
    file: UploadFile = File(...),
    speaker_audio: UploadFile | None = File(default=None),
    language: str | None = Form(None),
    task: str | None = Form(None),
    conversation_id: str | None = Form(None),
    node_context: NodeContextProvider = Depends(verify_api_key),
) -> dict[str, Any]:
    """Transcribe audio to text.

    Proxies the request to the Whisper service with app-to-app auth
    and context headers. The household_id context is critical for
    Whisper's voice recognition feature - it uses it to filter
    voice profiles to the correct household.

    Args:
        file: Audio file to transcribe
        speaker_audio: Optional separate audio used for the speaker pass
            only (e.g. wake-word + command concat). Improves speaker
            recognition accuracy on short follow-up utterances.
        language: Optional language code (e.g., "en", "es")
        task: Optional task ("transcribe" or "translate")
        node_context: Authenticated node context (from verify_api_key)

    Returns:
        Transcription result dict with text, optional speaker_id, etc.
    """
    # STT gets its own latency trace. Keyed by a unique id — NOT the
    # conversation_id — because the node's warmup (/conversation/start) can
    # still be in flight under the same conversation_id and start_request
    # would clobber that trace. The persisted trace row still carries the
    # conversation_id (when the node provided one) so STT rows join to the
    # voice turn in the trace visualizer.
    trace_key = f"stt-{uuid4().hex}"
    timing = latency_logger.start_request(trace_key, "stt")
    if conversation_id:
        timing.request_id = conversation_id
    timing.source = "node"
    timing.node_id = node_context.node.node_id
    timing.household_id = node_context.household_id

    try:
        client = WhisperClient(
            household_id=node_context.household_id,
            node_id=node_context.node.node_id,
            household_member_ids=node_context.household_member_ids,
        )

        # Read file content
        audio_bytes = await file.read()
        filename = file.filename or "audio.wav"

        speaker_audio_bytes: bytes | None = None
        speaker_audio_filename: str | None = None
        if speaker_audio is not None:
            speaker_audio_bytes = await speaker_audio.read()
            speaker_audio_filename = speaker_audio.filename or "speaker.wav"

        # Build extra params
        params: dict[str, Any] = {}
        if language:
            params["language"] = language
        if task:
            params["task"] = task

        with timing.measure(
            "stt_transcribe",
            service="whisper",
            metadata={
                "audio_bytes": len(audio_bytes),
                "speaker_audio_bytes": (
                    len(speaker_audio_bytes) if speaker_audio_bytes else 0
                ),
            },
        ):
            result = await client.transcribe(
                audio_bytes,
                filename,
                speaker_audio=speaker_audio_bytes,
                speaker_audio_filename=speaker_audio_filename,
                **params,
            )

        # Wake-clip verification: the leading ~2s of speaker_audio is the wake
        # snapshot. Fire-and-forget — the command turn reads the verdict from the
        # conversation cache with a bounded fail-open wait. Only when the node
        # provided a conversation_id (new nodes) and the mode isn't off.
        # Deliberately started AFTER the command transcribe returns: whisper
        # serializes model access, so racing this task against the command
        # transcribe would queue the user's STT behind the wake clip (and
        # before whisper grew its lock, the race crashed it outright —
        # GGML_ASSERT !sched->is_alloc). The command turn's bounded verdict
        # wait (VERDICT_WAIT_SECONDS) still covers the later start.
        if speaker_audio_bytes and conversation_id:
            from app.core.wake_verification import _get_mode, run_wake_verification
            if _get_mode(node_context.household_id, node_context.node.node_id) != "off":
                asyncio.create_task(run_wake_verification(
                    speaker_audio_bytes,
                    conversation_id,
                    node_context.household_id,
                    node_context.node.node_id,
                    node_context.household_member_ids,
                ))

        if isinstance(result, dict):
            timing.user_command = result.get("text")
        return result
    except Exception as exc:
        timing.trace_status = "error"
        timing.error_message = str(exc)
        raise
    finally:
        latency_logger.end_request(trace_key)


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


@router.post("/whisper/voice-profiles/verify")
async def whisper_verify_voice_profile(
    user_id: int,
    household_id: str,
    file: UploadFile = File(...),
    node_context: NodeContextProvider = Depends(verify_api_key),
) -> dict[str, Any]:
    """Verify a voice sample against an enrolled profile.

    Proxies the request to the Whisper service with app-to-app auth.
    """
    client = WhisperClient(
        household_id=household_id,
        node_id=node_context.node.node_id,
    )
    audio_bytes = await file.read()
    filename = file.filename or "verify.wav"
    return await client.verify_voice_profile(user_id, audio_bytes, filename)


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


# The former /tts/generate-wake-response proxy lived here. Removed in favor
# of /api/v0/wake-response (see app/api/wake_response.py) which talks to
# llm-proxy directly and runs the active prompt provider's sanitize_text,
# so model-specific artifacts (Qwen3 <think> blocks, etc.) never reach
# jarvis-tts or the node.

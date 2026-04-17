"""Mobile voice profile endpoints: enrollment, verification, and status.

JWT-authenticated endpoints for the mobile app to manage voice profiles.
Mirrors the node-authenticated media endpoints but uses JWT auth and
proxies through WhisperClient with household context.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.core.clients import WhisperClient
from app.deps import verify_user_jwt, verify_household_role, AuthenticatedUser

logger = logging.getLogger("uvicorn")

router = APIRouter(tags=["mobile-voice-profiles"])


@router.get("/voice-profile/status")
async def voice_profile_status(
    household_id: str,
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict[str, Any]:
    """Check whether the current user has an enrolled voice profile."""
    verify_household_role(user.user_id, household_id, required_role="member")

    client = WhisperClient(household_id=household_id, user_id=user.user_id)
    result = await client.check_voice_profile(user.user_id)
    return {"has_profile": result.get("exists", False)}


@router.post("/voice-profile/enroll")
async def voice_profile_enroll(
    file: UploadFile = File(...),
    household_id: str = Form(...),
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict[str, Any]:
    """Upload a voice sample to enroll (or update) the user's voice profile."""
    verify_household_role(user.user_id, household_id, required_role="member")

    client = WhisperClient(household_id=household_id, user_id=user.user_id)
    audio_bytes = await file.read()
    filename = file.filename or "enrollment.wav"

    result = await client.enroll_voice_profile(user.user_id, audio_bytes, filename)
    logger.info(
        "Voice profile enrolled via mobile",
        extra={"user_id": user.user_id, "household_id": household_id},
    )
    return result


@router.post("/voice-profile/verify")
async def voice_profile_verify(
    file: UploadFile = File(...),
    household_id: str = Form(...),
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict[str, Any]:
    """Test whether an audio sample matches the user's enrolled profile.

    Returns match result with confidence score. Used by the mobile
    enrollment wizard to let users confirm their profile works.
    """
    verify_household_role(user.user_id, household_id, required_role="member")

    client = WhisperClient(household_id=household_id, user_id=user.user_id)
    audio_bytes = await file.read()
    filename = file.filename or "verify.wav"

    result = await client.verify_voice_profile(user.user_id, audio_bytes, filename)
    return {
        "matched": result.get("matched", False),
        "confidence": result.get("confidence", 0.0),
    }


@router.delete("/voice-profile")
async def voice_profile_delete(
    household_id: str,
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict[str, Any]:
    """Delete the current user's voice profile."""
    verify_household_role(user.user_id, household_id, required_role="member")

    client = WhisperClient(household_id=household_id, user_id=user.user_id)
    result = await client.delete_voice_profile(user.user_id)
    logger.info(
        "Voice profile deleted via mobile",
        extra={"user_id": user.user_id, "household_id": household_id},
    )
    return result

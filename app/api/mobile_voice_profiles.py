"""Mobile voice profile endpoints: enrollment, verification, and status.

JWT-authenticated endpoints for the mobile app to manage voice profiles.
Mirrors the node-authenticated media endpoints but uses JWT auth and
proxies through WhisperClient with household context.
"""

import json
import logging
import os
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.clients import WhisperClient
from app.deps import (
    AuthenticatedUser,
    get_db,
    verify_household_role,
    verify_user_jwt,
)
from app.models import Node

logger = logging.getLogger("uvicorn")

router = APIRouter(tags=["mobile-voice-profiles"])

# File-based handoff for the node-mediated enrollment flow. Mirrors the
# pattern in app/api/node_tools.py — node POSTs the result to a file
# keyed by request_id, mobile polls the GET endpoint until the file is
# present. Lives under /tmp because the result is single-use.
_VP_RESULT_DIR = "/tmp/jarvis-voice-profile-results"


@router.get("/voice-profile/status")
async def voice_profile_status(
    household_id: str,
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict[str, Any]:
    """Check whether the current user has an enrolled voice profile.

    Returns sample_count so the mobile UI can show enrollment progress
    (e.g. "1 of 3 samples enrolled") for the multi-take wizard.
    """
    verify_household_role(user.user_id, household_id, required_role="member")

    client = WhisperClient(household_id=household_id, user_id=user.user_id)
    result = await client.check_voice_profile(user.user_id)
    return {
        "has_profile": result.get("exists", False),
        "sample_count": result.get("sample_count", 0),
    }


@router.post("/voice-profile/enroll")
async def voice_profile_enroll(
    file: UploadFile = File(...),
    household_id: str = Form(...),
    sample_index: int | None = Form(None),
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict[str, Any]:
    """Upload a voice sample to enroll (or update) the user's voice profile.

    When ``sample_index`` is omitted the backend auto-allocates the next
    index. The mobile multi-take wizard sends explicit indices so a
    "retake" overwrites the right slot.
    """
    verify_household_role(user.user_id, household_id, required_role="member")

    client = WhisperClient(household_id=household_id, user_id=user.user_id)
    audio_bytes = await file.read()
    filename = file.filename or "enrollment.wav"

    result = await client.enroll_voice_profile(
        user.user_id, audio_bytes, filename, sample_index=sample_index,
    )
    logger.info(
        "Voice profile enrolled via mobile",
        extra={
            "user_id": user.user_id,
            "household_id": household_id,
            "sample_index": result.get("sample_index"),
            "total_samples": result.get("total_samples"),
        },
    )
    return result


@router.get("/voice-profile/samples")
async def voice_profile_samples(
    household_id: str,
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict[str, Any]:
    """List the current user's enrolled voice samples.

    Used by the multi-take enrollment wizard to show progress and let
    the user retake individual samples.
    """
    verify_household_role(user.user_id, household_id, required_role="member")

    client = WhisperClient(household_id=household_id, user_id=user.user_id)
    return await client.list_voice_profile_samples(user.user_id)


@router.delete("/voice-profile/samples/{sample_index}")
async def voice_profile_delete_sample(
    sample_index: int,
    household_id: str,
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict[str, Any]:
    """Delete a single enrollment sample so the user can redo a bad take."""
    verify_household_role(user.user_id, household_id, required_role="member")

    client = WhisperClient(household_id=household_id, user_id=user.user_id)
    result = await client.delete_voice_profile_sample(user.user_id, sample_index)
    logger.info(
        "Voice profile sample deleted via mobile",
        extra={
            "user_id": user.user_id,
            "household_id": household_id,
            "sample_index": sample_index,
            "remaining_samples": result.get("remaining_samples"),
        },
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


# ---------------------------------------------------------------------------
# Node-mediated enrollment & verification
#
# Phone-mic enrollment produces embeddings tied to the phone's acoustics,
# and recognition on the node's mic then scores poorly. These endpoints
# trigger enrollment/verification ON the target node so the same mic
# captures the sample at enrollment time as at runtime.
# ---------------------------------------------------------------------------


class StartNodeEnrollmentBody(BaseModel):
    node_id: str
    prompt_text: str | None = None
    duration_secs: float | None = None


@router.post("/voice-profile/start-node-enrollment")
async def start_node_enrollment(
    body: StartNodeEnrollmentBody,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Tell ``node_id`` to record a voice sample on its own mic and enroll it.

    Returns a ``request_id`` the mobile app should poll on the
    ``/voice-profile-results/{request_id}`` GET below.
    """
    node = db.query(Node).filter(Node.node_id == body.node_id).first()
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    if not node.is_online():
        raise HTTPException(status_code=409, detail="Node is offline")
    if node.household_id:
        verify_household_role(
            user.user_id, node.household_id, required_role="member",
        )

    request_id = str(uuid4())

    # Defer the import: node_command_service depends on the MQTT client
    # which is heavy at import time.
    from app.services.node_command_service import get_node_command_service

    service = get_node_command_service()
    service.publish_command_with_id(
        node.node_id,
        "enroll_voice",
        {
            "request_id": request_id,
            "user_id": user.user_id,
            "household_id": node.household_id,
            "prompt_text": body.prompt_text or "",
            "duration_secs": body.duration_secs or 8.0,
        },
        request_id,
    )
    logger.info(
        "Node enrollment requested",
        extra={
            "request_id": request_id,
            "node_id": node.node_id,
            "user_id": user.user_id,
        },
    )
    return {"request_id": request_id}


class StartNodeVerificationBody(BaseModel):
    node_id: str
    prompt_text: str | None = None
    duration_secs: float | None = None


@router.post("/voice-profile/start-node-verification")
async def start_node_verification(
    body: StartNodeVerificationBody,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Tell ``node_id`` to record a voice sample on its own mic and verify it.

    Same flow as start-node-enrollment but hits the verify endpoint.
    Returns a ``request_id`` the mobile app polls on
    ``/voice-profile-results/{request_id}``.
    """
    node = db.query(Node).filter(Node.node_id == body.node_id).first()
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    if not node.is_online():
        raise HTTPException(status_code=409, detail="Node is offline")
    if node.household_id:
        verify_household_role(
            user.user_id, node.household_id, required_role="member",
        )

    request_id = str(uuid4())

    from app.services.node_command_service import get_node_command_service

    service = get_node_command_service()
    service.publish_command_with_id(
        node.node_id,
        "verify_voice",
        {
            "request_id": request_id,
            "user_id": user.user_id,
            "household_id": node.household_id,
            "prompt_text": body.prompt_text or "",
            "duration_secs": body.duration_secs or 5.0,
        },
        request_id,
    )
    logger.info(
        "Node verification requested",
        extra={
            "request_id": request_id,
            "node_id": node.node_id,
            "user_id": user.user_id,
        },
    )
    return {"request_id": request_id}


@router.post("/voice-profile-results/{request_id}", include_in_schema=False)
def post_voice_profile_result(request_id: str, body: dict) -> dict:
    """Node callback — writes the enrollment outcome to a file for polling.

    Intentionally unauthenticated (matches the node_tools pattern). The
    request_id is single-use and only lives until the mobile app polls.
    """
    os.makedirs(_VP_RESULT_DIR, exist_ok=True)
    result_file = os.path.join(_VP_RESULT_DIR, f"{request_id}.json")
    with open(result_file, "w") as f:
        json.dump(body, f)
    return {"status": "ok"}


@router.get("/voice-profile-results/{request_id}")
async def get_voice_profile_result(
    request_id: str,
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict[str, Any]:
    """Mobile polls this until the node POSTs a result.

    Returns 202 while pending. Result file is consumed (deleted) on read.
    """
    path = os.path.join(_VP_RESULT_DIR, f"{request_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=202, detail="pending")
    try:
        with open(path) as f:
            result = json.load(f)
        try:
            os.unlink(path)
        except OSError:
            pass
        return result
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"result read failed: {e}")

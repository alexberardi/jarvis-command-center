"""Self-scoped account-data endpoints (user JWT).

Mounted at ``/api/v0/me``. The ``user_id`` is taken from the validated Bearer
token (``sub``), never from the URL — a caller can only ever purge their own data.

Part of the cross-service account-deletion contract: jarvis-auth orchestrates a
``DELETE /auth/me`` and fans out to ``DELETE /api/v0/me/data`` here (and to the
notifications service) before deleting the user locally.
"""

import logging

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from app.deps import AuthenticatedUser, get_db, verify_user_jwt
from app.services.user_data_service import purge_user_data

router = APIRouter(prefix="/me", tags=["me"])
logger = logging.getLogger("uvicorn")


async def _purge_voiceprints(user_id: int) -> None:
    """Best-effort delete of the user's biometric voiceprints in whisper.

    Whisper is the canonical proxy target for voice profiles and CC holds the
    app credentials + service discovery, so the biometric purge is driven from
    here. Never block account deletion on it: whisper being down, on an older
    build without the cross-household endpoint, or absent from this install is
    logged and swallowed.
    """
    try:
        from app.core.clients.whisper_client import WhisperClient

        await WhisperClient(household_id="").delete_all_voice_profiles(user_id)
    except Exception as exc:  # noqa: BLE001 — deletion must not fail on a side channel
        logger.warning("Voiceprint purge for user_id=%s failed (continuing): %s", user_id, exc)


@router.delete("/data", status_code=status.HTTP_204_NO_CONTENT)
async def delete_my_data(
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> Response:
    """Purge every command-center row keyed to the authenticated user.

    Deletes the caller's memories, conversation transcripts, request traces,
    user-scoped settings, and OAuth auth-sessions, plus their enrolled voiceprints
    in whisper (best-effort). Idempotent — a user with no data still returns 204.
    Other users' rows and household-/node-scoped rows are never touched.
    """
    purge_user_data(db, user.user_id)
    await _purge_voiceprints(user.user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

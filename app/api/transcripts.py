"""Conversation-transcript feedback API (Phase 1).

Endpoints used by the mobile "Recent Commands" screen to surface recent voice
interactions and let the user rate whether the parse was correct. Ratings feed
the Phase 3 training-data extractor as explicit positive/negative signal.

Auth: user JWT from jarvis-auth. Filtering is scoped by user_id from the token.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.deps import AuthenticatedUser, get_db, verify_user_jwt
from app.services.transcript_service import TranscriptService

router = APIRouter(prefix="/transcripts", tags=["transcripts"])


_RECENT_LIMIT_DEFAULT = 50
_RECENT_LIMIT_MAX = 200


# --------------------------------------------------------------------------
# Response / request models
# --------------------------------------------------------------------------


class TranscriptResponse(BaseModel):
    id: int
    user_id: int
    household_id: str
    conversation_id: str
    user_message: str
    assistant_message: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    created_at: datetime
    user_rating: int | None = None
    rating_notes: str | None = None
    rated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class RatePayload(BaseModel):
    rating: int = Field(..., description="-1 / 0 / +1")
    notes: str | None = Field(default=None, max_length=2000)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _row_to_response(row) -> TranscriptResponse:
    tool_calls = None
    if row.tool_calls_json:
        try:
            tool_calls = json.loads(row.tool_calls_json)
        except (json.JSONDecodeError, ValueError, TypeError):
            tool_calls = None
    return TranscriptResponse(
        id=row.id,
        user_id=row.user_id,
        household_id=row.household_id,
        conversation_id=row.conversation_id,
        user_message=row.user_message,
        assistant_message=row.assistant_message,
        tool_calls=tool_calls,
        created_at=row.created_at,
        user_rating=row.user_rating,
        rating_notes=row.rating_notes,
        rated_at=row.rated_at,
    )


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@router.get("/recent")
def list_recent_transcripts(
    limit: int = Query(_RECENT_LIMIT_DEFAULT, ge=1, le=_RECENT_LIMIT_MAX),
    since: datetime | None = Query(None),
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> list[TranscriptResponse]:
    """Return the caller's most-recent transcripts, newest first."""
    service = TranscriptService(db)
    rows = service.list_recent_for_user(user_id=user.user_id, limit=limit, since=since)
    return [_row_to_response(r) for r in rows]


@router.post("/{transcript_id}/rate")
def rate_transcript(
    transcript_id: int,
    body: RatePayload,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> TranscriptResponse:
    """Record a −1 / 0 / +1 rating on one of the caller's transcripts.

    Returns 404 if the transcript doesn't exist OR doesn't belong to the caller
    (opaque — don't leak existence of other users' rows).
    """
    if body.rating not in (-1, 0, 1):
        raise HTTPException(status_code=400, detail="rating must be -1, 0, or 1")

    service = TranscriptService(db)
    row = service.rate_transcript(
        transcript_id=transcript_id,
        user_id=user.user_id,
        rating=body.rating,
        notes=body.notes,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return _row_to_response(row)

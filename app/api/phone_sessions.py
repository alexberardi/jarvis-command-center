"""Gateway-facing phone-session endpoints (app-to-app auth).

The contract's CC half — the gateway's ``services/session_client.py`` is
the other half; keep the shapes in lockstep (contract tests pin them):

  GET  /internal/phone/sessions/{id}          — session snapshot for the worker
  POST /internal/phone/sessions/{id}/events   — everything the gateway reports:
      {"type": "claim_dial", "worker_url"}    200 ok / 409 = do NOT dial
      {"type": "state", "state", ...extra}    lifecycle transitions
      {"type": "turn", "turn": {...}}         transcript append (+heartbeat)
      {"type": "heartbeat"}                   bare liveness
      {"type": "escalation", "question"}      mid-call question → answer card + push
      {"type": "outcome", "outcome", "audio_key"}  final facts + audio ref

Security requirement 1: the claim CAS here (confirmed→dialing, single
winner) is the ONLY authorization to dial — the Redis job is transport.
"""

import json
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.deps import get_db, require_app_auth
from app.models import PhoneCallSession
from app.services.phone_call_service import (
    ACTIVE_STATES,
    TERMINAL_STATES,
    extract_constraint_envelope,
    post_escalation_card,
    post_outcome_card,
    transition,
    upsert_contact_from_call,
)

router = APIRouter()
logger = logging.getLogger("uvicorn")


class SessionEventBody(BaseModel):
    type: str
    # claim_dial
    worker_url: str | None = None
    # state
    state: str | None = None
    twilio_call_sid: str | None = None
    error: str | None = None
    duration_seconds: int | None = None
    # turn
    turn: dict[str, Any] | None = None
    # escalation
    question: str | None = None
    # outcome
    outcome: dict[str, Any] | None = None
    audio_key: str | None = None


def _get_session_or_404(db: Session, session_id: str) -> PhoneCallSession:
    session = (
        db.query(PhoneCallSession)
        .filter(PhoneCallSession.id == session_id)
        .first()
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get(
    "/internal/phone/sessions/{session_id}",
    dependencies=[Depends(require_app_auth)],
)
async def get_phone_session(session_id: str, db: Session = Depends(get_db)) -> dict:
    """Session snapshot for the claiming worker — everything the call loop
    needs, nothing more (no transcript echo, no audit fields)."""
    s = _get_session_or_404(db, session_id)
    from app.services.phone_call_service import _int_setting, _resolve_initiator_name

    # The disclosure line speaks this ("calling on behalf of {name}") —
    # without it the gateway falls back to "a customer" (heard live).
    initiator_name = await _resolve_initiator_name(s.user_id)

    return {
        "id": s.id,
        "state": s.state,
        "initiator_name": initiator_name,
        "household_id": s.household_id,
        "contact_name": s.contact_name,
        "contact_address": s.contact_address,
        "dialed_number": s.dialed_number,
        "goal": s.goal,
        "details": s.details,
        # Derived from the CONFIRMED brief, so the gateway's
        # "negotiate only within these" section reflects exactly what the
        # user approved on the card — including any edits they made to the
        # times. An explicit stored value still wins if one is ever set.
        "constraints": s.constraints or extract_constraint_envelope(s.details),
        "line_type": s.line_type,
        "max_call_seconds": _int_setting(
            "phone_calls.max_call_seconds", s.household_id, 600
        ),
    }


@router.post(
    "/internal/phone/sessions/{session_id}/events",
    dependencies=[Depends(require_app_auth)],
)
def post_phone_session_event(
    session_id: str,
    body: SessionEventBody,
    db: Session = Depends(get_db),
) -> dict:
    s = _get_session_or_404(db, session_id)
    now = datetime.utcnow()

    if body.type == "claim_dial":
        if not body.worker_url:
            raise HTTPException(status_code=400, detail="worker_url required")
        # DB-level CAS: exactly one worker wins; everyone else 409s and drops
        # the job. Unknown/duplicate/forged jobs die here.
        claimed = (
            db.query(PhoneCallSession)
            .filter(
                PhoneCallSession.id == session_id,
                PhoneCallSession.state == "confirmed",
            )
            .update(
                {
                    "state": "dialing",
                    "worker_url": body.worker_url,
                    "heartbeat_at": now,
                }
            )
        )
        db.commit()
        if claimed != 1:
            logger.info(
                "☎️ claim_dial rejected session=%s state=%s", session_id[:8], s.state
            )
            raise HTTPException(status_code=409, detail=f"Session is {s.state}")
        logger.info(
            "☎️ Session %s claimed by %s", session_id[:8], body.worker_url
        )
        return {"status": "ok", "state": "dialing"}

    if body.type == "state":
        if not body.state:
            raise HTTPException(status_code=400, detail="state required")
        if not transition(s, body.state):
            raise HTTPException(
                status_code=409,
                detail=f"Illegal transition {s.state} -> {body.state}",
            )
        s.heartbeat_at = now
        if body.twilio_call_sid:
            s.twilio_call_sid = body.twilio_call_sid
        if body.duration_seconds is not None:
            s.duration_seconds = body.duration_seconds
        if body.error:
            s.error_message = body.error
        db.commit()
        # An honest failure card when the gateway reports a failed call and
        # no outcome event will follow.
        if body.state == "failed":
            from app.services.phone_call_service import _post_card

            _post_card(
                household_id=s.household_id,
                user_id=s.user_id,
                title=f"⚠️ Call failed: {s.contact_name}",
                summary=body.error or "The call could not be completed.",
                body="",
                metadata={"household_id": s.household_id, "session_id": s.id},
            )
        return {"status": "ok", "state": s.state}

    if body.type == "turn":
        if body.turn is None:
            raise HTTPException(status_code=400, detail="turn required")
        if s.state not in ACTIVE_STATES:
            raise HTTPException(status_code=409, detail=f"Session is {s.state}")
        turns: list = []
        if s.transcript_json:
            try:
                turns = json.loads(s.transcript_json)
            except json.JSONDecodeError:
                turns = []
        turns.append(body.turn)
        s.transcript_json = json.dumps(turns)
        s.heartbeat_at = now
        db.commit()
        return {"status": "ok", "turns": len(turns)}

    if body.type == "heartbeat":
        if s.state in TERMINAL_STATES:
            raise HTTPException(status_code=409, detail=f"Session is {s.state}")
        s.heartbeat_at = now
        db.commit()
        return {"status": "ok"}

    if body.type == "escalation":
        # Gateway hit something outside the brief: build the answer card and
        # push it to the initiating user. The gateway holds its own bounded
        # window (~25 s) — a late tap gets the graceful "call may have ended"
        # path via the escalation_answer callback, never a stuck card.
        question = (body.question or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question required")
        if s.state != "in_call":
            raise HTTPException(status_code=409, detail=f"Session is {s.state}")
        s.heartbeat_at = now
        db.commit()
        post_escalation_card(s, question)
        return {"status": "ok"}

    if body.type == "outcome":
        if body.outcome is None:
            raise HTTPException(status_code=400, detail="outcome required")
        s.outcome_json = json.dumps(body.outcome)
        if body.audio_key:
            s.audio_object_key = body.audio_key
        if body.duration_seconds is not None:
            s.duration_seconds = body.duration_seconds
        # Outcome normally arrives in wrapup; accept from any live state but
        # land terminal exactly once.
        if s.state not in TERMINAL_STATES:
            if s.state in ("dialing", "in_call"):
                transition(s, "wrapup")
            transition(s, "done")
        s.heartbeat_at = now
        db.commit()
        # Remember the business so the next call to it needs no lookup.
        # Best-effort: a phonebook write must never cost us the outcome card.
        try:
            upsert_contact_from_call(db, s)
        except Exception as e:  # noqa: BLE001
            logger.warning("Phonebook auto-save failed for %s: %s", s.id[:8], e)
            db.rollback()
        post_outcome_card(s)
        return {"status": "ok", "state": s.state}

    raise HTTPException(status_code=400, detail=f"Unknown event type {body.type!r}")

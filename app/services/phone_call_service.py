"""Phone-call session service — the CC half of phone-calls P1.

Owns everything stateful about a call: plan creation (resolve → draft →
confirm card), the state machine on ``phone_call_sessions``, the confirm
tap's atomic transitions, the Redis dial hand-off, caps, and the outcome
card. The gateway owns the phone line and reports back through
``app/api/phone_sessions.py``; it never touches this DB.

State machine: draft → confirmed → dialing → in_call → wrapup →
done | failed | declined | expired. Every transition goes through
``transition()`` (single choke point).

Security posture (PRD requirements 1-8):
- The Redis job is transport only — authorization is the confirmed row +
  the claim CAS in the events endpoint.
- Caps are enforced fail-closed at BOTH plan time and confirm time.
- The confirm tap is single-use (state must be draft, not expired).
- resolved_number vs dialed_number + number_edited + confirmed_by are the
  audit trail; edited numbers are re-validated before dialing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import httpx
from sqlalchemy.orm import Session

from app.models import PhoneCallSession, PhoneContact
from app.services.phone_number_search import find_business_number, location_mismatch
from app.services.server_callback_registry import (
    ServerCallbackContext,
    ServerCallbackResult,
    register_server_callback,
)

logger = logging.getLogger("uvicorn")

DIAL_QUEUE_KEY = "phone:dial"

TERMINAL_STATES = ("done", "failed", "declined", "expired")
ACTIVE_STATES = ("dialing", "in_call", "wrapup")

# Legal transitions (PRD call lifecycle). Keys = from-state.
_TRANSITIONS: dict[str, set[str]] = {
    "draft": {"confirmed", "declined", "expired", "failed"},
    "confirmed": {"dialing", "failed", "expired"},
    "dialing": {"in_call", "wrapup", "failed"},
    "in_call": {"wrapup", "failed"},
    "wrapup": {"done", "failed"},
}


class PhoneCallError(Exception):
    """User-presentable failure — message is safe to speak/show."""


class NumberValidationError(PhoneCallError):
    pass


# =============================================================================
# Settings (fail-closed)
# =============================================================================


def phone_calls_enabled(household_id: str | None) -> bool:
    """Fail-closed master gate — clone of the web_search pattern.

    Any settings error → disabled. This backs both the tool's execute-time
    refusal and the confirm tap.
    """
    if not household_id:
        return False
    try:
        from app.services.settings_service import get_settings_service

        val = get_settings_service().get(
            "phone_calls.enabled", household_id=str(household_id)
        )
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return bool(val)
    except Exception as e:  # noqa: BLE001 — fail closed
        logger.warning("phone_calls gate check failed, defaulting DISABLED: %s", e)
        return False


def _int_setting(key: str, household_id: str | None, default: int) -> int:
    try:
        from app.services.settings_service import get_settings_service

        kwargs: dict[str, Any] = {}
        if household_id:
            kwargs["household_id"] = str(household_id)
        raw = get_settings_service().get(key, **kwargs)
        if raw is None:
            return default
        if isinstance(raw, bool):
            return default
        return int(str(raw).strip())
    except Exception:  # noqa: BLE001
        return default


# =============================================================================
# Number validation (decision 10)
# =============================================================================

# US-only in P1 (country allowlist is a provisioning-level posture; the
# Twilio account is additionally geo-locked). Deny: emergency/short codes,
# premium-rate, anything that isn't a plain 10-digit NANP number.
_PREMIUM_AREA_CODES = {"900", "976"}
_EMERGENCY_NUMBERS = {"911", "112", "933", "988"}


def normalize_us_number(raw: str) -> str:
    """Normalize to E.164 +1XXXXXXXXXX or raise NumberValidationError."""
    if not raw or not raw.strip():
        raise NumberValidationError("No phone number provided.")
    digits = re.sub(r"[^\d+]", "", raw.strip())
    if digits in _EMERGENCY_NUMBERS or digits.lstrip("+") in _EMERGENCY_NUMBERS:
        raise NumberValidationError("Emergency and service numbers can't be called.")
    d = digits.lstrip("+")
    if len(d) == 11 and d.startswith("1"):
        d = d[1:]
    if len(d) != 10:
        if len(d) < 7:
            raise NumberValidationError(
                "Short codes and service numbers can't be called."
            )
        raise NumberValidationError(
            "Only US numbers are supported right now (10 digits)."
        )
    area = d[:3]
    if area in _PREMIUM_AREA_CODES:
        raise NumberValidationError("Premium-rate numbers can't be called.")
    if area.startswith("0") or area.startswith("1"):
        raise NumberValidationError("That doesn't look like a valid US number.")
    return f"+1{d}"


# =============================================================================
# Contact resolution
# =============================================================================


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def resolve_contact(
    db: Session, household_id: str, business_name: str
) -> PhoneContact | None:
    """Fuzzy phonebook lookup within the household. DNC is checked by the
    caller so the refusal message can be specific."""
    contacts = (
        db.query(PhoneContact)
        .filter(PhoneContact.household_id == household_id)
        .all()
    )
    if not contacts:
        return None
    try:
        from rapidfuzz import fuzz, process

        target = _normalize_name(business_name)
        match = process.extractOne(
            target,
            {c.id: c.normalized_name for c in contacts},
            scorer=fuzz.WRatio,
            score_cutoff=80,
        )
        if match is None:
            return None
        _matched_name, _score, contact_id = match
        return next((c for c in contacts if c.id == contact_id), None)
    except Exception as e:  # noqa: BLE001 — resolution is best-effort
        logger.warning("Contact fuzzy match failed: %s", e)
        return None


def _search_miss_note(reason: str | None) -> str:
    """Honest, specific reason the card has no number to show.

    "Web search is off" and "I looked and couldn't find it" are very
    different facts for the user; collapsing them into one message would
    hide a setting they can change.
    """
    if reason == "web_search_disabled":
        return (
            "This business isn't in your phonebook, and web search is off "
            "for your household, so I couldn't look it up — enter the "
            "number to call."
        )
    if reason in ("no_results", "no_number_found"):
        return (
            "This business isn't in your phonebook and I couldn't find a "
            "number for it online — enter the number to call."
        )
    if reason == "search_failed":
        return (
            "This business isn't in your phonebook and the lookup failed — "
            "enter the number to call."
        )
    return (
        "I couldn't find this business in the phonebook — "
        "enter the number to call."
    )


async def lookup_line_type(number: str) -> str:
    """Line-type via the gateway (it holds the Twilio creds — CC never does).

    Contract: POST {gateway}/internal/lookup/line-type {"number": e164}
    -> {"line_type": "mobile"|"landline"|"voip"|"unknown"}.
    Graceful degradation: any failure → "unknown".
    """
    try:
        from app.core import service_config

        base = service_config.get_phone_gateway_url().rstrip("/")
        headers = {
            "X-Jarvis-App-Id": os.getenv("JARVIS_APP_ID", ""),
            "X-Jarvis-App-Key": os.getenv("JARVIS_APP_KEY", ""),
        }
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.post(
                f"{base}/internal/lookup/line-type",
                json={"number": number},
                headers=headers,
            )
            if r.status_code == 200:
                lt = r.json().get("line_type")
                if lt in ("mobile", "landline", "voip"):
                    return lt
    except Exception as e:  # noqa: BLE001 — optional service, degrade
        logger.info("Line-type lookup unavailable (%s) — using 'unknown'", e)
    return "unknown"


# =============================================================================
# Caps (fail-closed, checked at plan AND confirm time)
# =============================================================================


def check_caps(db: Session, household_id: str) -> str | None:
    """Return a user-presentable refusal string when a cap blocks a new
    call/plan, else None. Errors count as blocked (fail-closed)."""
    try:
        now = datetime.utcnow()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        per_day = _int_setting("phone_calls.calls_per_day", household_id, 10)
        today = (
            db.query(PhoneCallSession)
            .filter(
                PhoneCallSession.household_id == household_id,
                PhoneCallSession.created_at >= day_start,
            )
            .count()
        )
        if today >= per_day:
            return "The daily call limit for this household has been reached."

        concurrent_cap = _int_setting(
            "phone_calls.max_concurrent_calls", household_id, 1
        )
        active = (
            db.query(PhoneCallSession)
            .filter(
                PhoneCallSession.household_id == household_id,
                PhoneCallSession.state.in_(ACTIVE_STATES),
            )
            .count()
        )
        if active >= concurrent_cap:
            return "There's already a call in progress for this household."

        minutes_cap = _int_setting(
            "phone_calls.monthly_minutes_cap", household_id, 60
        )
        used_seconds = sum(
            s.duration_seconds or 0
            for s in db.query(PhoneCallSession)
            .filter(
                PhoneCallSession.household_id == household_id,
                PhoneCallSession.created_at >= month_start,
                PhoneCallSession.duration_seconds.isnot(None),
            )
            .all()
        )
        if used_seconds >= minutes_cap * 60:
            return "The monthly call-minutes limit for this household has been reached."
        return None
    except Exception as e:  # noqa: BLE001 — fail closed
        logger.error("Phone caps check failed — blocking (fail-closed): %s", e)
        return "Phone calls are temporarily unavailable."


# =============================================================================
# State machine
# =============================================================================


def transition(session: PhoneCallSession, to_state: str) -> bool:
    """Apply a state transition if legal. Returns False (no mutation) when
    illegal — callers decide whether that's a 409 or a no-op."""
    allowed = _TRANSITIONS.get(session.state, set())
    if to_state not in allowed:
        return False
    session.state = to_state
    if to_state == "confirmed":
        session.confirmed_at = datetime.utcnow()
    if to_state in TERMINAL_STATES:
        session.ended_at = datetime.utcnow()
    return True


# =============================================================================
# Plan creation (tool → resolve → draft → confirm card)
# =============================================================================


async def create_call_plan(
    *,
    business: str,
    goal: str,
    household_id: str,
    user_id: int | None,
) -> None:
    """Background flow behind the tool's spoken ack.

    Failures land as an inbox card, never silence (anti-vanishing rule).
    """
    from app.db import get_session_local

    db = get_session_local()()
    try:
        cap_refusal = check_caps(db, household_id)
        if cap_refusal:
            _post_card(
                household_id=household_id,
                user_id=user_id,
                title="📵 Call not started",
                summary=cap_refusal,
                body="",
                metadata={"household_id": household_id},
            )
            return

        contact = resolve_contact(db, household_id, business)
        resolved_number: str | None = None
        line_type = "unknown"
        if contact is not None:
            if contact.do_not_call:
                _post_card(
                    household_id=household_id,
                    user_id=user_id,
                    title="📵 Call refused",
                    summary=(
                        f"{contact.name} is marked do-not-call for this household."
                    ),
                    body="Remove the do-not-call flag from the phonebook to call them again.",
                    metadata={"household_id": household_id},
                )
                return
            try:
                resolved_number = normalize_us_number(contact.number)
            except NumberValidationError:
                resolved_number = None
            line_type = contact.line_type or "unknown"

        # Where the number came from drives the card's note: the user must
        # always know whether they're looking at a number they curated or one
        # a search engine guessed at (PRD stale-search-results mitigation).
        number_source = "phonebook" if resolved_number else None
        search_address: str | None = None
        search_url: str | None = None
        searched_near: str | None = None
        location_warning: str | None = None
        resolution_note = ""

        if contact is None:
            # Phonebook miss → try the web, gated per household. Runs in a
            # thread: the search + scrape machinery is synchronous.
            search = await asyncio.to_thread(
                find_business_number, business, household_id
            )
            if search.found:
                resolved_number = search.number
                number_source = "web"
                search_address = search.address
                search_url = search.source_url
                searched_near = search.searched_near
                # A wrong-state result is the failure mode that actually bit
                # us (a Maryland pizzeria for a New Jersey household). Warn
                # loudly on the card; never block — the tap is the checkpoint.
                location_warning = location_mismatch(
                    search.address, search.searched_near
                )
                near_note = (
                    f" (searched near {searched_near})" if searched_near else ""
                )
                resolution_note = (
                    f"I found this number via web search{near_note} — check it "
                    "before calling."
                )
            else:
                resolution_note = _search_miss_note(search.reason)

        if number_source == "phonebook":
            resolution_note = "This number is from your phonebook."

        if resolved_number and line_type == "unknown":
            line_type = await lookup_line_type(resolved_number)

        initiator = await _resolve_initiator_name(user_id)
        details = await _draft_details(
            business=business, goal=goal, initiator=initiator
        )
        # Plan-time context: real availability from the node's calendar
        # command, if one is installed and reachable (PRD cross-agent-context).
        details = await apply_availability_envelope(
            household_id=household_id, goal=goal, details=details
        )

        ttl_minutes = _int_setting("phone_calls.plan_ttl_minutes", household_id, 20)
        now = datetime.utcnow()
        session_row = PhoneCallSession(
            id=str(uuid4()),
            household_id=household_id,
            user_id=user_id,
            contact_id=contact.id if contact else None,
            contact_name=contact.name if contact else business,
            goal=goal,
            details=details,
            resolved_number=resolved_number,
            dialed_number=None,
            contact_address=search_address,
            line_type=line_type,
            state="draft",
            created_at=now,
            expires_at=now + timedelta(minutes=ttl_minutes),
        )
        db.add(session_row)
        db.commit()

        expires_iso = session_row.expires_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        mobile_note = (
            " Note: this appears to be a mobile number." if line_type == "mobile" else ""
        )
        source_line = f" Source: {search_url}" if search_url else ""
        warning_line = f"{location_warning}\n\n" if location_warning else ""
        metadata = {
            "household_id": household_id,
            "session_id": session_row.id,
            "number_source": number_source or "none",
            "editor_schema": 2,
            "editable_fields": [
                {
                    "label": "Phone number",
                    "initial": resolved_number or "",
                    "data_key": "dialed_number",
                    "input_type": "tel",
                    "required": True,
                },
                {
                    "label": "Details",
                    "initial": details,
                    "data_key": "details",
                    "input_type": "multiline",
                    "required": True,
                },
            ],
            "expires_at": expires_iso,
            "interactive_elements": [
                {
                    "id": "confirm-call",
                    "label": "Call now",
                    "command": "make_phone_call",
                    "callback": "confirm_call",
                    "target": "server",
                    "data": {
                        "session_id": session_row.id,
                        "dialed_number": "",
                        "details": "",
                    },
                },
                {
                    "id": "cancel-call",
                    "label": "Cancel",
                    "command": "make_phone_call",
                    "callback": "cancel_call",
                    "target": "server",
                    "data": {"session_id": session_row.id},
                },
            ],
        }
        _post_card(
            household_id=household_id,
            user_id=user_id,
            title=f"📞 Call plan: {session_row.contact_name}",
            summary=f"{goal}{mobile_note}",
            body=(
                f"{warning_line}"
                f"Review the number and details, then tap **Call now**. "
                f"{resolution_note}{source_line} The call will open with an "
                f"AI + recording disclosure.{mobile_note}"
            ),
            metadata=metadata,
        )
    except Exception as e:  # noqa: BLE001 — never vanish
        logger.exception("Call plan creation failed")
        _post_card(
            household_id=household_id,
            user_id=user_id,
            title="📵 Call plan failed",
            summary=f"Couldn't prepare the call: {e}",
            body="",
            metadata={"household_id": household_id},
        )
    finally:
        db.close()


_SCHEDULING_HINTS = (
    "appointment", "book", "schedule", "reservation", "reserve", "visit",
)


def is_scheduling_goal(goal: str, details: str = "") -> bool:
    """Does this call need times negotiated on it?"""
    lowered = f"{goal} {details}".lower()
    return any(h in lowered for h in _SCHEDULING_HINTS)


def _ensure_times_section(goal: str, details: str) -> str:
    """Deterministic belt over the prompt: scheduling goals ALWAYS carry an
    Acceptable-times section the user can fill on the confirm card — the
    model complying with the drafting prompt is hoped-for, not guaranteed
    (live 2026-07-19: a card shipped without it)."""
    if "acceptable times" in details.lower():
        return details
    if not is_scheduling_goal(goal, details):
        return details
    return (
        f"{details.rstrip()}\n"
        "Acceptable times: (fill in your availability before calling)"
    )


def _format_availability(data: dict[str, Any]) -> str | None:
    """Render a calendar answer as the brief's constraint envelope.

    Kept compact and human-editable: the card is still the guardrail
    boundary, so the user must be able to read and correct this at a
    glance. Returns None when the answer carries nothing usable.
    """
    free = [str(w) for w in (data.get("free") or []) if str(w).strip()]
    busy = [str(w) for w in (data.get("busy") or []) if str(w).strip()]
    if not free and not busy:
        return None
    parts: list[str] = []
    if free:
        parts.append("Acceptable times: " + "; ".join(free[:6]))
    else:
        parts.append("Acceptable times: (calendar shows no free windows — edit me)")
    if busy:
        parts.append("Do not book: " + "; ".join(busy[:6]))
    return "\n".join(parts)


async def apply_availability_envelope(
    *, household_id: str, goal: str, details: str
) -> str:
    """Bake real calendar availability into the brief at PLAN time.

    Context enters here and only here (phone-calls PRD, cross-agent-context):
    the callee is untrusted input, so the call loop never gets a live
    calendar tool — it only ever sees this envelope, which the user reviews
    and can edit on the confirm card.

    Degrades in every direction: no scheduling goal, no node, no calendar
    command, node offline, malformed answer — all fall back to the
    fill-me-in placeholder rather than blocking or inventing times.
    """
    if not is_scheduling_goal(goal, details):
        return details
    if "acceptable times" in details.lower() and "(fill in" not in details.lower():
        return details  # the user or the model already supplied real times

    try:
        from app.services.context_provider_client import query_context

        start = datetime.utcnow().date()
        answer = await query_context(
            household_id,
            "availability",
            {"start": start.isoformat(), "end": (start + timedelta(days=7)).isoformat()},
        )
    except Exception as e:  # noqa: BLE001 — context is an enhancement, never a gate
        logger.warning("Availability lookup failed: %s", e)
        return _ensure_times_section(goal, details)

    if not answer.ok:
        logger.info("Availability unavailable (%s) — using placeholder", answer.error)
        base = _strip_times_placeholder(details)
        return (
            f"{base.rstrip()}\n"
            "Acceptable times: (calendar unavailable — fill in your availability)"
        )

    envelope = _format_availability(answer.data)
    if not envelope:
        return _ensure_times_section(goal, details)

    return f"{_strip_times_placeholder(details).rstrip()}\n{envelope}"


def _strip_times_placeholder(details: str) -> str:
    """Drop a fill-me-in line so a real envelope can replace it."""
    kept = [
        line
        for line in details.splitlines()
        if not (
            line.lower().startswith("acceptable times:")
            and ("fill in" in line.lower() or "unavailable" in line.lower())
        )
    ]
    return "\n".join(kept)


async def _resolve_initiator_name(user_id: int | None) -> str | None:
    """Display name for the disclosure line + brief; None degrades gracefully."""
    if user_id is None:
        return None
    try:
        from app.core.utils.speaker_resolver import resolve_speaker_name
        from app.deps import _get_auth_base_url

        return await resolve_speaker_name(_get_auth_base_url(), user_id)
    except Exception as e:  # noqa: BLE001 — name is a nicety
        logger.debug("Initiator name resolution failed: %s", e)
        return None


async def _draft_details(
    *, business: str, goal: str, initiator: str | None = None
) -> str:
    """Background-model draft of the details brief. Degrades to the raw goal."""
    try:
        from app.core.llm_proxy_client import LLMProxyClient

        client = LLMProxyClient()
        result = await client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You draft briefs for an assistant that places phone "
                        "calls to businesses. The assistant can use ONLY what "
                        "the brief contains — it cannot look anything up "
                        "mid-call — so gather everything the business will "
                        "plausibly need up front. Format: 1-3 plain sentences "
                        "stating exactly what to accomplish, then short "
                        "'If asked:' lines anticipating the business's "
                        "questions (name for the order/booking, quantities, "
                        "sizes, timing). Use the caller's name when given. "
                        "For appointments or anything needing scheduling, end "
                        "with an 'Acceptable times:' line — copy times from "
                        "the goal if stated, otherwise write exactly "
                        "'Acceptable times: (fill in your availability "
                        "before calling)'. Never include payment details. "
                        "Stay strictly consistent with the goal: never mix "
                        "pickup and delivery; only include an address for a "
                        "delivery; invent no facts. /no_think"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Business: {business}\nGoal: {goal}"
                        + (f"\nCaller name: {initiator}" if initiator else "")
                    ),
                },
            ],
            model="background",
            temperature=0.3,
            include_date_context=True,
            max_tokens=200,
        )
        content = (result.get("content") or result.get("message") or "").strip()
        return _ensure_times_section(goal, content or goal)
    except Exception as e:  # noqa: BLE001 — draft is a nicety
        logger.warning("Details draft failed, using raw goal: %s", e)
        return _ensure_times_section(goal, goal)


# =============================================================================
# Confirm / cancel / escalation callbacks (server plane)
# =============================================================================


def _handle_confirm_call(ctx: ServerCallbackContext) -> ServerCallbackResult:
    from app.db import get_session_local

    db = get_session_local()()
    try:
        session_id = ctx.data.get("session_id")
        if not session_id:
            return ServerCallbackResult(success=False, error="Missing session_id")
        session = (
            db.query(PhoneCallSession)
            .filter(
                PhoneCallSession.id == session_id,
                PhoneCallSession.household_id == ctx.household_id,
            )
            .first()
        )
        if session is None:
            return ServerCallbackResult(success=False, error="Call plan not found")

        # Gate re-check at the moment of authorization (toggle-off mid-plan
        # invalidates pending plans).
        if not phone_calls_enabled(ctx.household_id):
            transition(session, "declined")
            session.error_message = "phone_calls.enabled turned off"
            db.commit()
            return ServerCallbackResult(
                success=False, error="Phone calls are disabled for this household."
            )

        if session.state != "draft":
            # Single-use: a second tap is a friendly no-op.
            return ServerCallbackResult(
                success=True,
                context_data={
                    "inbox": {
                        "title": "📞 Already handled",
                        "summary": f"This call plan is already {session.state}.",
                        "metadata": {"household_id": ctx.household_id},
                    }
                },
            )
        if session.expires_at and datetime.utcnow() > session.expires_at:
            transition(session, "expired")
            db.commit()
            return ServerCallbackResult(
                success=False,
                error="This call plan expired. Ask Jarvis again to get a fresh one.",
            )

        cap_refusal = check_caps(db, ctx.household_id)
        if cap_refusal:
            transition(session, "declined")
            session.error_message = cap_refusal
            db.commit()
            return ServerCallbackResult(success=False, error=cap_refusal)

        # The edited number is the highest-risk moment — re-validate it.
        raw_number = (ctx.data.get("dialed_number") or "").strip()
        try:
            dialed = normalize_us_number(raw_number)
        except NumberValidationError as e:
            return ServerCallbackResult(success=False, error=str(e))

        details = (ctx.data.get("details") or "").strip()
        if not details:
            return ServerCallbackResult(
                success=False, error="The call details can't be empty."
            )

        session.dialed_number = dialed
        session.number_edited = dialed != (session.resolved_number or "")
        session.details = details
        session.confirmed_by = ctx.user_id
        transition(session, "confirmed")
        db.commit()

        try:
            enqueue_dial(session.id, ctx.household_id)
        except Exception as e:  # noqa: BLE001 — honest failure, never stuck
            logger.error("Dial enqueue failed for %s: %s", session.id[:8], e)
            transition(session, "failed")
            session.error_message = f"dial enqueue failed: {e}"
            db.commit()
            return ServerCallbackResult(
                success=False,
                error="Couldn't start the call — try again in a minute.",
            )

        logger.info(
            "📞 Call confirmed session=%s number=%s edited=%s by user=%s",
            session.id[:8], dialed, session.number_edited, ctx.user_id,
        )
        return ServerCallbackResult(
            success=True,
            context_data={
                "inbox": {
                    "title": f"📞 Calling {session.contact_name}…",
                    "summary": "You'll get a summary card when the call ends.",
                    "metadata": {
                        "household_id": ctx.household_id,
                        "session_id": session.id,
                    },
                }
            },
        )
    finally:
        db.close()


def _handle_cancel_call(ctx: ServerCallbackContext) -> ServerCallbackResult:
    from app.db import get_session_local

    db = get_session_local()()
    try:
        session_id = ctx.data.get("session_id")
        session = (
            db.query(PhoneCallSession)
            .filter(
                PhoneCallSession.id == session_id,
                PhoneCallSession.household_id == ctx.household_id,
            )
            .first()
        )
        if session is None:
            return ServerCallbackResult(success=False, error="Call plan not found")
        if session.state == "draft":
            transition(session, "declined")
            db.commit()
        return ServerCallbackResult(
            success=True,
            context_data={
                "inbox": {
                    "title": "🚫 Call cancelled",
                    "summary": f"Won't call {session.contact_name}.",
                    "metadata": {"household_id": ctx.household_id},
                }
            },
        )
    finally:
        db.close()


async def _handle_escalation_answer(
    ctx: ServerCallbackContext,
) -> ServerCallbackResult:
    """Mid-call escalation answer → forward to the owning gateway worker."""
    from app.db import get_session_local

    db = get_session_local()()
    try:
        session_id = ctx.data.get("session_id")
        answer = (ctx.data.get("answer") or "").strip()
        session = (
            db.query(PhoneCallSession)
            .filter(
                PhoneCallSession.id == session_id,
                PhoneCallSession.household_id == ctx.household_id,
            )
            .first()
        )
        if session is None or not session.worker_url:
            return ServerCallbackResult(
                success=False, error="This call is no longer active."
            )
        if session.state not in ACTIVE_STATES:
            return ServerCallbackResult(
                success=False, error="The call already ended."
            )
        worker = session.worker_url.rstrip("/")
    finally:
        db.close()

    try:
        headers = {
            "X-Jarvis-App-Id": os.getenv("JARVIS_APP_ID", ""),
            "X-Jarvis-App-Key": os.getenv("JARVIS_APP_KEY", ""),
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{worker}/internal/call/{session_id}/escalation-answer",
                json={"answer": answer, "answered_by": ctx.user_id},
                headers=headers,
            )
        if r.status_code != 200:
            return ServerCallbackResult(
                success=False,
                error="Couldn't reach the call — it may have just ended.",
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("Escalation forward failed for %s: %s", session_id, e)
        return ServerCallbackResult(
            success=False, error="Couldn't reach the call — it may have just ended."
        )
    return ServerCallbackResult(success=True)


def register_phone_callbacks() -> None:
    register_server_callback("make_phone_call", "confirm_call", _handle_confirm_call)
    register_server_callback("make_phone_call", "cancel_call", _handle_cancel_call)
    register_server_callback(
        "make_phone_call", "escalation_answer", _handle_escalation_answer
    )
    logger.info("📞 Phone-call server callbacks registered")


# =============================================================================
# Redis dial queue (transport only — security requirement 1)
# =============================================================================


def enqueue_dial(session_id: str, household_id: str) -> None:
    """LPUSH the dial job. Raises on any failure — the caller marks the
    session failed with an honest card (never a silent stuck 'confirmed')."""
    import redis

    url = os.getenv("REDIS_URL")
    if not url:
        raise PhoneCallError("REDIS_URL is not configured")
    client = redis.Redis.from_url(url, socket_timeout=5, socket_connect_timeout=5)
    try:
        client.lpush(
            DIAL_QUEUE_KEY,
            json.dumps({"session_id": session_id, "household_id": household_id}),
        )
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


# =============================================================================
# Outcome card + inbox helper
# =============================================================================


def post_outcome_card(session: PhoneCallSession) -> None:
    """Outcome inbox card. Callee-derived content is ATTRIBUTED (security
    requirement 3) — the summary reads as a report about the call, and
    structured facts are labeled as coming from the callee."""
    outcome: dict[str, Any] = {}
    if session.outcome_json:
        try:
            outcome = json.loads(session.outcome_json)
        except json.JSONDecodeError:
            outcome = {}

    achieved = outcome.get("goal_achieved")
    summary = outcome.get("summary") or "The call ended."
    facts = outcome.get("facts") or {}
    fact_lines = "\n".join(f"- **{k}**: {v}" for k, v in facts.items() if v)
    facts_block = (
        f"\n\nWhat the business said:\n{fact_lines}" if fact_lines else ""
    )
    title_icon = "✅" if achieved else ("📞" if achieved is None else "⚠️")

    _post_card(
        household_id=session.household_id,
        user_id=session.user_id,
        title=f"{title_icon} Call finished: {session.contact_name}",
        summary=summary,
        body=(
            f"{summary}{facts_block}\n\n"
            f"_Summary generated from the call transcript; statements above "
            f"are the business's, not Jarvis's._"
        ),
        metadata={
            "household_id": session.household_id,
            "session_id": session.id,
            **(
                {"audio_object_key": session.audio_object_key}
                if session.audio_object_key
                else {}
            ),
        },
    )


def upsert_contact_from_call(db: Session, session: PhoneCallSession) -> PhoneContact | None:
    """Remember a business after a call that actually worked.

    Without this the phonebook can never fill: nothing else writes to it, so
    every call would keep asking the user for a number they already gave.

    Rules:
    - Only completed calls. A failed or declined attempt proves nothing about
      the number.
    - The **dialed** number wins, not the resolved one. If the user corrected
      it on the confirm card, that correction is the most trustworthy signal
      we have about this business.
    - ``do_not_call`` is never cleared. A household that blocked a business
      does not un-block it by calling once.
    - Idempotent: same business, same household → one row, freshly verified.
    """
    if session.state != "done":
        return None
    number = session.dialed_number or session.resolved_number
    name = (session.contact_name or "").strip()
    if not number or not name:
        return None
    try:
        number = normalize_us_number(number)
    except NumberValidationError:
        return None

    normalized = _normalize_name(name)
    existing = (
        db.query(PhoneContact)
        .filter(
            PhoneContact.household_id == session.household_id,
            PhoneContact.normalized_name == normalized,
        )
        .first()
    )
    now = datetime.utcnow()

    if existing is not None:
        existing.number = number          # a user correction supersedes
        existing.verified_at = now
        if session.line_type and session.line_type != "unknown":
            existing.line_type = session.line_type
        if session.contact_address and not existing.address:
            existing.address = session.contact_address
        if existing.source == "web":
            existing.source = "call"      # a completed call outranks a guess
        db.commit()
        return existing

    contact = PhoneContact(
        id=str(uuid4()),
        household_id=session.household_id,
        name=name,
        normalized_name=normalized,
        number=number,
        address=session.contact_address,
        source="call",
        line_type=(session.line_type if session.line_type != "unknown" else None),
        do_not_call=False,
        verified_at=now,
        created_at=now,
    )
    db.add(contact)
    db.commit()
    return contact


def post_escalation_card(session: PhoneCallSession, question: str) -> None:
    """Mid-call escalation: push the answer card to the initiating user.

    The callee's question is UNTRUSTED input (security requirement 3): it is
    rendered attributed — "the business asked" — never in Jarvis's voice, and
    never as tappable actions; the only actions are our own fixed chips. The
    gateway's answer window is short (~25 s), so a late tap degrades through
    the escalation_answer callback's "call may have ended" path.
    """
    metadata: dict[str, Any] = {
        "household_id": session.household_id,
        "session_id": session.id,
        "editor_schema": 2,
        "editable_fields": [
            {
                "label": "Your answer",
                "initial": "",
                "data_key": "answer",
                "input_type": "multiline",
                "required": True,
            },
        ],
        # Bounds stale cards; the real window is the gateway's, not this TTL.
        "expires_at": (datetime.utcnow() + timedelta(minutes=10)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "interactive_elements": [
            {
                "id": "send-answer",
                "label": "Send answer",
                "command": "make_phone_call",
                "callback": "escalation_answer",
                "target": "server",
                "data": {"session_id": session.id, "answer": ""},
            },
            {
                "id": "end-call",
                "label": "End the call",
                "command": "make_phone_call",
                "callback": "cancel_call",
                "target": "server",
                "data": {"session_id": session.id},
            },
        ],
    }
    _post_card(
        household_id=session.household_id,
        user_id=session.user_id,
        title="📞 The call needs your input",
        summary=f'They asked: "{question}"',
        body=(
            f'On the call to **{session.contact_name or session.dialed_number}**, '
            f'the person asked:\n\n> {question}\n\n'
            "Answer quickly — Jarvis is holding the line and will offer a "
            "call-back if this window passes."
        ),
        metadata=metadata,
    )


def _post_card(
    *,
    household_id: str,
    user_id: int | None,
    title: str,
    summary: str,
    body: str,
    metadata: dict[str, Any],
) -> None:
    try:
        from app.services.inbox_notification_service import post_inbox_item_sync

        post_inbox_item_sync(
            household_id=household_id,
            user_id=user_id,
            title=title,
            summary=summary,
            body=body,
            category="phone_call",
            metadata=metadata,
            push=True,
            target_type="user" if user_id is not None else "household",
        )
    except Exception as e:  # noqa: BLE001 — inbox is best-effort
        logger.warning("Phone-call inbox card failed: %s", e)


# =============================================================================
# Reaper (call lifecycle)
# =============================================================================

HEARTBEAT_STALE_SECONDS = 60


async def reap_phone_sessions() -> int:
    """One reaper pass. Returns number of sessions reaped.

    - in_call/dialing/wrapup with stale heartbeat or over max_call_seconds
      → failed + honest notify + best-effort worker cancel.
    - expired drafts → expired (cards render the expired state client-side).
    """
    from app.db import get_session_local

    db = get_session_local()()
    reaped = 0
    try:
        now = datetime.utcnow()

        stale_cutoff = now - timedelta(seconds=HEARTBEAT_STALE_SECONDS)
        active = (
            db.query(PhoneCallSession)
            .filter(PhoneCallSession.state.in_(ACTIVE_STATES))
            .all()
        )
        for s in active:
            max_seconds = _int_setting(
                "phone_calls.max_call_seconds", s.household_id, 600
            )
            started = s.confirmed_at or s.created_at
            over_time = started and (now - started).total_seconds() > max_seconds + 120
            stale = (s.heartbeat_at or started or now) < stale_cutoff
            if not (stale or over_time):
                continue
            reason = "call exceeded the time limit" if over_time else "lost contact with the call"
            s.error_message = reason
            s.state = "failed"
            s.ended_at = now
            db.commit()
            reaped += 1
            logger.warning("☎️ Reaped session %s (%s)", s.id[:8], reason)
            if s.worker_url:
                try:
                    headers = {
                        "X-Jarvis-App-Id": os.getenv("JARVIS_APP_ID", ""),
                        "X-Jarvis-App-Key": os.getenv("JARVIS_APP_KEY", ""),
                    }
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        await client.post(
                            f"{s.worker_url.rstrip('/')}/internal/call/{s.id}/cancel",
                            headers=headers,
                        )
                except Exception:  # noqa: BLE001 — best-effort
                    pass
            _post_card(
                household_id=s.household_id,
                user_id=s.user_id,
                title=f"⚠️ Call ended: {s.contact_name}",
                summary=f"The call was ended because Jarvis {reason}.",
                body="",
                metadata={"household_id": s.household_id, "session_id": s.id},
            )

        expired_drafts = (
            db.query(PhoneCallSession)
            .filter(
                PhoneCallSession.state == "draft",
                PhoneCallSession.expires_at.isnot(None),
                PhoneCallSession.expires_at < now,
            )
            .all()
        )
        for s in expired_drafts:
            s.state = "expired"
            s.ended_at = now
            reaped += 1
        if expired_drafts:
            db.commit()
        return reaped
    except Exception:  # noqa: BLE001 — reaper must never die
        logger.exception("Phone-session reaper pass failed")
        return reaped
    finally:
        db.close()

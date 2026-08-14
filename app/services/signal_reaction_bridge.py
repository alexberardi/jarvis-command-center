"""SignalReactionBridge — the ``appt.upcoming`` → leave-by CARD reaction.

An ``appt.upcoming`` Signal (emitted by the calendar_alerts agent when a located
event enters the leave-by runway) triggers a DETERMINISTIC reaction here:

  1. run ``get_drive_time`` on the node (strict resolution) — deterministic facts
  2. compute the absolute leave-by instant  (event_start − drive − buffer)
  3. persist a ``leave_by.suggested`` Signal (observability / reactive render)
  4. emit a ``reminder.set_at`` tap-to-confirm CARD via the hardened
     directed-proposal path (resolve the reminder-owning node + emit_proposal_card)

No LLM is in the loop — the ``leave_by.suggested`` → ``reminder.set_at`` mapping is
1:1 — so there is zero nag risk. The user taps to confirm; the confirm routes to the
shipped ``jarvis.proposable_action.execute`` dispatcher (which re-validates opt-in,
typed params, blast tier, idempotency, then runs ``reminder.set_at`` on the node).

Gated on ``proposals.enabled`` (the same fail-closed gate the confirm honors, so a
household with proposals off does no work) and the per-category "Never suggest this"
suppression (one tap silences the whole leave-by category).

The proactive situation matcher (``proposals.proactive_enabled``) stays OFF; this
reaction does not use it. If it is ever enabled, the persisted
``leave_by.suggested`` Signal must be EXCLUDED from its bundle — otherwise it would
propose a second reminder card on top of the one this bridge already proposes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from app.services.signal_reaction_registry import ReactionContext, register_signal_reaction

logger = logging.getLogger("uvicorn")

DEFAULT_LEAVE_BUFFER_MINUTES = 5
# Stable category key the card's "Never suggest this" writes; the reaction checks
# it up front so one suppression silences ALL future leave-by cards (not per-event).
_LEAVE_BY_SUPPRESS_SOURCE = "leaveby"
# get_drive_time (Directions round-trip) + the MQTT node dispatch. Runs as an async
# task on the main loop (not a sync threadpool worker), so a longer await is free.
_DRIVE_TIME_TIMEOUT_S = 20.0

# In-process dedup so the calendar agent's per-cycle re-emit of the same upcoming
# event doesn't spawn duplicate reactions/cards. Keyed on (household, event); resets
# on restart (the emitted card's idempotency_key + the signal upsert back it up).
_fired: set[str] = set()


# ── gates + helpers ───────────────────────────────────────────────────────────
def _proposals_enabled(household_id: str) -> bool:
    """Fail-closed read of ``proposals.enabled`` — the same gate the confirm
    dispatcher honors. Off ⇒ the card could never be confirmed, so skip the whole
    reaction (and its node work)."""
    try:
        from app.services.proposable_action_service import _proposals_enabled as _pe

        return _pe(household_id)
    except Exception:  # noqa: BLE001 — fail closed
        return False


def _is_suppressed(household_id: str, user_id: int | None) -> bool:
    """True once the user tapped 'Never suggest this' on a leave-by card."""
    try:
        from app.services.proposal_suppressions import suppression_signals

        sig = suppression_signals(
            household_id=household_id, user_id=user_id, command="reminder"
        )
        return _LEAVE_BY_SUPPRESS_SOURCE in (sig.get("source_keys") or [])
    except Exception:  # noqa: BLE001 — a bookkeeping read must not block the card
        return False


def _claim(household_id: str, event_id: str) -> bool:
    """Claim (household, event) for reaction; False if already claimed."""
    key = f"{household_id}:{event_id}"
    if key in _fired:
        return False
    _fired.add(key)
    return True


def _parse_iso(raw: Any) -> datetime | None:
    """Parse an ISO-8601 string to a tz-aware datetime. A naive value (producers
    should send tz-aware — the calendar command attaches the node's local tz) is
    treated as UTC so it can be compared against a tz-aware ``now``."""
    if not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


async def _node_command_names(node_id: str) -> set[str]:
    """The command names the node currently advertises (short-timeout probe)."""
    from app.api.node_tools import _request_tools_from_node

    report = await _request_tools_from_node(node_id, timeout=4.0)
    if not report:
        return set()
    return {
        c.get("command_name")
        for c in (report.get("available_commands") or [])
        if c.get("command_name")
    }


async def _dispatch_drive_time(node_id: str, location: str, user_id: int | None) -> dict[str, Any]:
    """Run ``get_drive_time`` on the node in STRICT resolution mode and return its
    output. Strict mode skips un-pinnable generic locations ("Dentist") — those come
    back ``success=False`` with ``reason='generic_location'`` and yield no card."""
    from app.services.node_command_service import dispatch_node_command

    return await dispatch_node_command(
        node_id,
        "get_drive_time",
        {"destination": location, "resolution": "strict"},
        user_id=user_id,
        timeout=_DRIVE_TIME_TIMEOUT_S,
    )


def _save_leave_by_signal(
    *,
    household_id: str,
    node_id: str | None,
    user_id: int | None,
    event_id: str,
    title: str,
    due_at_iso: str,
    drive_minutes: int,
    address: str,
    start: datetime,
) -> None:
    """Persist ``leave_by.suggested`` for observability + reactive render. Best-effort
    — a failure here must not block the card. NOT wired to the situation matcher (no
    ``signal_situation_edge``): the card is proposed deterministically below, so
    routing this signal through the matcher too would double-propose."""
    try:
        from app.db import get_session_local
        from app.services.signal_service import SignalService

        db = get_session_local()()
        try:
            ttl = max(
                300, int((start - datetime.now(timezone.utc)).total_seconds()) + 300
            )
            SignalService(db).save_signal(
                household_id=household_id,
                source_key=f"leaveby:{event_id}",
                kind="leave_by.suggested",
                summary=f"Leave-by reminder suggested for {title}",
                facts={
                    "title": title,
                    "due_at_iso": due_at_iso,
                    "drive_minutes": drive_minutes,
                    "address": address,
                    "event_id": event_id,
                },
                user_id=user_id,
                node_id=node_id,
                ttl_seconds=ttl,
                cacheable=False,
                source_agent="leave_by_bridge",
            )
        finally:
            db.close()
    except Exception:  # noqa: BLE001 — observability write, never blocks the card
        logger.debug("leave-by: save_signal failed", exc_info=True)


async def _emit_leave_by_card(
    *,
    household_id: str,
    user_id: int | None,
    event_id: str,
    title: str,
    due_at_iso: str,
    address: str,
    start_display: str,
    duration_text: str,
) -> bool:
    """Resolve the reminder-owning node and emit the ``reminder.set_at`` confirm card.

    Reuses the hardened directed-proposal node resolution + the SDK-shaped card. The
    crux: ``idempotency_key`` rides IN ``params`` (it's a declared REQUIRED param of
    set_at) so the confirm-time ``validate_against_params`` passes; the same value
    keys the dispatcher's dedup via ``_action.idempotency_key``. Returns True if a
    card was posted."""
    from app.services.capability_registry import resolve_proposable_action
    from app.services.proposable_action_service import resolve_household_node_for_command
    from app.services.proposal_card import emit_proposal_card

    # The reminder-owning node may differ from the drive-time node — resolve it.
    target_node = await resolve_household_node_for_command(
        household_id, "reminder", "set_at"
    )
    if not target_node:
        logger.info("leave-by: no household node advertises reminder.set_at — no card")
        return False
    action = await resolve_proposable_action(target_node, "reminder", "set_at")
    if not action:
        return False

    idem = f"leaveby:{event_id}"
    when = f"at {start_display}" if start_display else "soon"
    params = {
        "text": f"Leave for {title}",
        "due_at_iso": due_at_iso,
        "idempotency_key": idem,
    }
    summary = f"{title} {when} — {duration_text} drive from home"
    body = (
        f"{title} starts {when} at {address}. It's about a {duration_text} drive, "
        f"so I can remind you when it's time to leave."
    )
    return emit_proposal_card(
        household_id=household_id,
        node_id=target_node,
        command="reminder",
        callback="set_at",
        params=params,
        idempotency_key=idem,
        card_title=action.get("card_title") or "Set a reminder to leave on time?",
        confirm_label=action.get("confirm_label") or "Set reminder",
        dismiss_label=action.get("dismiss_label") or "Dismiss",
        blast_tier=action.get("blast_tier") or "reversible",
        summary=summary,
        body=body,
        user_id=user_id,
        # Stable category source → "Never suggest this" blocks all future leave-by.
        source=_LEAVE_BY_SUPPRESS_SOURCE,
        descriptor="leave-by reminders for calendar events",
    )


# ── the reaction ──────────────────────────────────────────────────────────────
async def react_to_appt_upcoming(
    ctx: ReactionContext,
    *,
    enabled_check: Callable[[str], bool] | None = None,
    menu_fetch: Callable[[str], Awaitable[set[str]]] | None = None,
    dispatch: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    emit_card: Callable[..., Awaitable[bool]] | None = None,
) -> str:
    """React to an ``appt.upcoming`` signal by proposing a leave-by reminder card.

    A registered handler on the generic signal-reaction registry (dispatched by
    kind — this function names no wiring). Deterministic: drive time (facts) is
    computed on the node, then the card is proposed directly (no LLM). Returns a
    short status string (for logging/tests): one of ``disabled``,
    ``no_node``/``no_location``/``no_start``/``no_user``, ``suppressed``,
    ``duplicate``, ``no_drive_time``, ``no_route``, ``departure_passed``,
    ``not_advertised``, ``proposed``, ``error``. Never raises.
    """
    household_id = ctx.household_id
    facts = ctx.facts or {}
    node_id = ctx.node_id
    user_id = ctx.user_id

    enabled_check = enabled_check or _proposals_enabled
    if not enabled_check(household_id):
        return "disabled"
    if not node_id:
        return "no_node"
    location = facts.get("location")
    if not location:
        return "no_location"  # leave-by is meaningless without a place
    start_iso = facts.get("start_iso") or facts.get("start")
    if not start_iso:
        return "no_start"
    if user_id is None:
        return "no_user"  # reminder.set_at needs a concrete speaker

    event_id = str(facts.get("event_id") or facts.get("id") or start_iso)
    if _is_suppressed(household_id, user_id):
        return "suppressed"
    if not _claim(household_id, event_id):
        return "duplicate"

    try:
        # The node must advertise get_drive_time so we can compute the facts.
        menu_fetch = menu_fetch or _node_command_names
        names = await menu_fetch(node_id)
        if "get_drive_time" not in names:
            logger.info("leave-by: node %s lacks get_drive_time — skipping", node_id)
            return "no_drive_time"

        # (1) drive time on the node (strict resolution → generic locations skipped)
        dispatch = dispatch or _dispatch_drive_time
        result = await dispatch(node_id, location, user_id)
        if not isinstance(result, dict) or not result.get("success"):
            logger.info(
                "leave-by: get_drive_time no route (%s)",
                (result or {}).get("reason") or (result or {}).get("error"),
            )
            return "no_route"
        try:
            drive_minutes = int(result.get("duration_minutes"))
        except (TypeError, ValueError):
            return "no_route"
        address = result.get("destination") or location
        duration_text = result.get("duration_text") or f"{drive_minutes} min"

        # (2) the absolute leave-by instant
        start = _parse_iso(start_iso)
        if start is None:
            return "bad_start"
        now = datetime.now(timezone.utc)
        due_at = start - timedelta(minutes=drive_minutes + DEFAULT_LEAVE_BUFFER_MINUTES)
        if due_at <= now:
            return "departure_passed"
        due_at_iso = due_at.isoformat()
        title = facts.get("title") or "your appointment"
        start_display = facts.get("start_display") or ""

        # (3) persist leave_by.suggested (observability only — see helper docstring)
        _save_leave_by_signal(
            household_id=household_id,
            node_id=node_id,
            user_id=user_id,
            event_id=event_id,
            title=title,
            due_at_iso=due_at_iso,
            drive_minutes=drive_minutes,
            address=address,
            start=start,
        )

        # (4) the reminder.set_at CARD
        emit_card = emit_card or _emit_leave_by_card
        emitted = await emit_card(
            household_id=household_id,
            user_id=user_id,
            event_id=event_id,
            title=title,
            due_at_iso=due_at_iso,
            address=address,
            start_display=start_display,
            duration_text=duration_text,
        )
        return "proposed" if emitted else "not_advertised"
    except Exception:  # noqa: BLE001 — a reaction must never break signal ingest
        logger.warning("leave-by reaction failed for %s", household_id, exc_info=True)
        return "error"


# ── registration ──────────────────────────────────────────────────────────────
def register_leave_by_reaction() -> None:
    """Register the leave-by reaction for ``appt.upcoming`` (called once at startup).

    Dispatch + scheduling live in the generic :mod:`signal_reaction_registry` — this
    module owns only the reaction's logic, and names its trigger kind here in one
    line. Adding another reaction (any kind) needs no ingest or dispatcher change."""
    register_signal_reaction("appt.upcoming", "leave_by", react_to_appt_upcoming)

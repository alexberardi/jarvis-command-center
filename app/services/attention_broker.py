"""Attention Broker — deterministic, LLM-free routing for proactive output.

Every proactive notification (interposed /node/* endpoints, or direct
POST /api/v0/attention/events) is recorded as an AttentionEvent and routed
down the ladder journal -> inbox -> push through four gates: consent ceiling,
trust tier, budget, context (quiet hours). Failing a gate demotes one rung —
never drops; the floor is the journal. Safety-class categories bypass gates
entirely (only exact-dedupe applies). See prds/attention-broker.md.

No LLM anywhere in this module, by design. Any future LLM consumer proposes
a rung; this code only demotes (the demote-only law).
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.models import AttentionConsent, AttentionDelivery, AttentionEvent, AttentionSourceTier
from app.services.settings_service import get_settings_service

# Rung order, least to most intrusive. led_invite/speak are reserved for the
# verified in-room rung (separate work) — the phase-1 ladder tops out at push.
RUNGS = ["journal", "inbox", "push", "led_invite", "speak"]

_TIER_TO_MAX_RUNG = {0: "journal", 1: "inbox", 2: "push", 3: "push"}

_TRUTHY = ("true", "1", "yes")


def _is_truthy(value) -> bool:
    return value is True or str(value).lower() in _TRUTHY


def _rung_index(rung: str) -> int:
    return RUNGS.index(rung)


def _demote(rung: str) -> str:
    return RUNGS[max(0, _rung_index(rung) - 1)]


def _cap(rung: str, ceiling: str) -> str:
    return rung if _rung_index(rung) <= _rung_index(ceiling) else ceiling


def fallback_dedupe_key(title: str) -> str:
    """Stable identity for legacy producers that set no dedupe_key."""
    normalized = " ".join(title.strip().lower().split())
    return "title:" + hashlib.sha256(normalized.encode()).hexdigest()[:32]


@dataclass
class BrokerDecision:
    deliver: bool
    rung: str
    event_id: str
    delivery_id: str
    withheld_by: str | None = None
    gate_trail: list[dict] = field(default_factory=list)


def attention_enabled(household_id: str) -> bool:
    settings = get_settings_service()
    return _is_truthy(settings.get("attention.enabled", household_id=household_id))


def _household_now(settings, household_id: str) -> datetime:
    tz_name = settings.get("attention.timezone", household_id=household_id) or "UTC"
    try:
        tz = ZoneInfo(str(tz_name))
    except Exception:
        tz = timezone.utc
    return datetime.now(tz)


def _local_day_start_utc_naive(local_now: datetime) -> datetime:
    """Start of the household-local day, as the naive-UTC datetime the DB stores."""
    day_start_local = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    return day_start_local.astimezone(timezone.utc).replace(tzinfo=None)


def _in_quiet_hours(spec: str, local_now: datetime) -> bool:
    """spec is "HH:MM-HH:MM" in household-local time; may cross midnight."""
    try:
        start_s, end_s = str(spec).split("-")
        sh, sm = start_s.strip().split(":")[:2]
        eh, em = end_s.strip().split(":")[:2]
        start = dt_time(int(sh), int(sm))
        end = dt_time(int(eh), int(em))
    except Exception:
        return False  # malformed spec never silences anything
    now_t = local_now.time()
    if start <= end:
        return start <= now_t < end
    return now_t >= start or now_t < end  # crosses midnight


def _safety_categories(settings, household_id: str) -> set[str]:
    raw = settings.get("attention.safety_categories", household_id=household_id) or ""
    return {part.strip().lower() for part in str(raw).split(",") if part.strip()}


def _int_setting(settings, key: str, household_id: str, default: int) -> int:
    try:
        return int(settings.get(key, household_id=household_id))
    except (TypeError, ValueError):
        return default


def record_and_route(
    db: Session,
    *,
    household_id: str,
    source: str,
    category: str,
    title: str,
    summary: str = "",
    requested_rung: str = "push",
    dedupe_key: str | None = None,
    target_user_id: int | None = None,
    origin_node_id: str | None = None,
    payload: dict | None = None,
) -> BrokerDecision:
    """Record the event and run the gates. Commits the event + delivery rows.

    Callers perform the actual delivery for deliver=True decisions (through
    the same downstream services as today), then stamp the outcome via
    mark_outcome(). The broker never raises for policy reasons; callers wrap
    this in try/except and fall through to legacy delivery on error
    (fail-open — see the PRD's Security section).
    """
    settings = get_settings_service()
    trail: list[dict] = []

    # notifications hard caps: title String(500) errors (not truncates) on
    # overflow; category is String(50).
    title = (title or "").strip()[:500] or "(untitled)"
    category = (category or "general").strip()[:50] or "general"
    source = (source or category).strip()[:100]
    key = dedupe_key or fallback_dedupe_key(title)

    event = AttentionEvent(
        household_id=household_id,
        source=source,
        category=category,
        title=title,
        summary=summary or "",
        dedupe_key=key,
        target_user_id=target_user_id,
        origin_node_id=origin_node_id,
        payload_json=json.dumps(payload) if payload else None,
    )
    db.add(event)
    db.flush()

    local_now = _household_now(settings, household_id)
    day_start = _local_day_start_utc_naive(local_now)
    safety = category.lower() in _safety_categories(settings, household_id)
    rung = requested_rung if requested_rung in RUNGS else "push"
    # Phase-1 ladder tops out at push regardless of what was requested.
    rung = _cap(rung, "push")
    withheld_by: str | None = None

    # Gate 0 — dedup. Applies to safety categories too (exact key only).
    window_h = _int_setting(settings, "attention.dedupe_window_hours", household_id, 24)
    window_start = datetime.utcnow() - timedelta(hours=window_h)
    dup = (
        db.query(AttentionEvent.id)
        .filter(
            AttentionEvent.household_id == household_id,
            AttentionEvent.source == source,
            AttentionEvent.dedupe_key == key,
            AttentionEvent.created_at >= window_start,
            AttentionEvent.id != event.id,
        )
        .first()
    )
    if dup is not None:
        trail.append({"gate": "dedupe", "result": "duplicate", "detail": f"of event {dup.id} within {window_h}h"})
        rung, withheld_by = "journal", "dedupe"
    elif safety:
        trail.append({"gate": "safety_class", "result": "bypass", "detail": f"category '{category}' is safety-class"})
    else:
        # Gate 1 — consent ceiling.
        consent = (
            db.query(AttentionConsent)
            .filter_by(household_id=household_id, source=source, category=category)
            .first()
        )
        ceiling = consent.max_rung if consent else "push"
        if ceiling == "never":
            trail.append({"gate": "consent", "result": "withheld", "detail": "ceiling is 'never'"})
            rung, withheld_by = "journal", "consent"
        else:
            capped = _cap(rung, ceiling if ceiling in RUNGS else "push")
            trail.append({"gate": "consent", "result": "pass" if capped == rung else "demoted", "detail": f"ceiling {ceiling}"})
            rung = capped

        # Gate 2 — trust tier (rows only exist once phase-2 scoring writes them).
        if withheld_by is None:
            tier_row = (
                db.query(AttentionSourceTier)
                .filter_by(household_id=household_id, source=source, category=category)
                .first()
            )
            if tier_row is not None:
                tier_ceiling = _TIER_TO_MAX_RUNG.get(tier_row.tier, "push")
                capped = _cap(rung, tier_ceiling)
                trail.append({
                    "gate": "tier",
                    "result": "pass" if capped == rung else "demoted",
                    "detail": f"T{tier_row.tier}" + (f" ({tier_row.state_reason})" if tier_row.state_reason else ""),
                })
                rung = capped
                if rung == "journal":
                    withheld_by = "tier"
            else:
                trail.append({"gate": "tier", "result": "pass", "detail": "no tier row (default)"})

        # Gate 3 — budgets. Exhausted budget demotes one rung, never drops.
        if withheld_by is None:
            source_cap = _int_setting(settings, "attention.source_daily_cap", household_id, 4)
            source_today = (
                db.query(AttentionDelivery)
                .join(AttentionEvent, AttentionDelivery.event_id == AttentionEvent.id)
                .filter(
                    AttentionDelivery.household_id == household_id,
                    AttentionEvent.source == source,
                    AttentionDelivery.rung != "journal",
                    AttentionDelivery.created_at >= day_start,
                )
                .count()
            )
            if source_today >= source_cap:
                trail.append({"gate": "budget", "result": "withheld", "detail": f"source cap {source_cap}/day reached"})
                rung, withheld_by = "journal", "budget"
            else:
                budget_keys = {"push": "attention.daily_push_budget", "inbox": "attention.daily_inbox_budget"}
                defaults = {"push": 8, "inbox": 30}
                while rung in budget_keys:
                    budget = _int_setting(settings, budget_keys[rung], household_id, defaults[rung])
                    used = (
                        db.query(AttentionDelivery)
                        .filter(
                            AttentionDelivery.household_id == household_id,
                            AttentionDelivery.rung == rung,
                            AttentionDelivery.created_at >= day_start,
                        )
                        .count()
                    )
                    if used < budget:
                        trail.append({"gate": "budget", "result": "pass", "detail": f"{rung} {used}/{budget}"})
                        break
                    demoted = _demote(rung)
                    trail.append({"gate": "budget", "result": "demoted", "detail": f"{rung} budget {budget}/day exhausted -> {demoted}"})
                    rung = demoted
                if rung == "journal" and withheld_by is None:
                    withheld_by = "budget"

        # Gate 4 — context (quiet hours): push demotes to inbox.
        if withheld_by is None and rung == "push":
            quiet = settings.get("attention.quiet_hours", household_id=household_id) or ""
            if quiet and _in_quiet_hours(str(quiet), local_now):
                trail.append({"gate": "context", "result": "demoted", "detail": f"quiet hours {quiet} -> inbox"})
                rung = "inbox"
            else:
                trail.append({"gate": "context", "result": "pass", "detail": "outside quiet hours"})

    delivery = AttentionDelivery(
        event_id=event.id,
        household_id=household_id,
        rung=rung,
        gate_trail_json=json.dumps(trail),
        withheld_by=withheld_by,
        outcome="duplicate" if withheld_by == "dedupe" else None,
    )
    db.add(delivery)
    db.commit()

    return BrokerDecision(
        deliver=rung != "journal",
        rung=rung,
        event_id=event.id,
        delivery_id=delivery.id,
        withheld_by=withheld_by,
        gate_trail=trail,
    )


def mark_outcome(db: Session, delivery_id: str, *, outcome: str, inbox_item_id: str | None = None) -> None:
    delivery = db.query(AttentionDelivery).filter_by(id=delivery_id).first()
    if delivery is None:
        return
    delivery.outcome = outcome
    if inbox_item_id:
        delivery.inbox_item_id = inbox_item_id
    db.commit()

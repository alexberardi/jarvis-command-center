"""Daily attention-journal card + TTL cleanup (prds/attention-broker.md).

The card is the transparency surface: what was delivered at which rung, and —
the actual product — what was withheld and by which gate. Suppression must be
visible or the broker "feels dead" (the weather-agent burn, inverted).

Rendered as a markdown body on a plain inbox item (category
"attention_journal"): unknown categories fall back to mobile's generic
InboxDetail screen, and content_format is a backend phantom so the body
always renders as markdown. View-only in phase 1 — Useful/Mute buttons need
a node-hosted @callback command (phase 2).
"""
import json
import logging
from collections import Counter
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.models import AttentionDelivery, AttentionEvent

logger = logging.getLogger(__name__)

JOURNAL_CATEGORY = "attention_journal"


def households_with_activity(db: Session, *, hours: int = 24) -> list[str]:
    """Households with broker events in the window — the broker enumerates
    from its own rows (settings cascade cannot enumerate; see PRD)."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    rows = (
        db.query(AttentionEvent.household_id)
        .filter(AttentionEvent.created_at >= cutoff)
        .distinct()
        .all()
    )
    return [r.household_id for r in rows]


def compose_journal_card(db: Session, household_id: str, *, hours: int = 24) -> tuple[str, str] | None:
    """Returns (summary, markdown_body), or None when there was no activity."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    rows = (
        db.query(AttentionDelivery, AttentionEvent)
        .join(AttentionEvent, AttentionDelivery.event_id == AttentionEvent.id)
        .filter(
            AttentionDelivery.household_id == household_id,
            AttentionDelivery.created_at >= cutoff,
        )
        .order_by(AttentionDelivery.created_at.desc())
        .all()
    )
    if not rows:
        return None

    delivered = [(d, e) for d, e in rows if d.rung != "journal"]
    withheld = [(d, e) for d, e in rows if d.rung == "journal"]

    lines: list[str] = []
    if delivered:
        lines.append(f"**Delivered ({len(delivered)})**")
        for d, e in delivered[:20]:
            lines.append(f"- {e.title} — {e.source} → {d.rung}")
        if len(delivered) > 20:
            lines.append(f"- …and {len(delivered) - 20} more")
        lines.append("")
    if withheld:
        gate_counts = Counter(d.withheld_by or "unknown" for d, _ in withheld)
        gates = ", ".join(f"{n} {g}" for g, n in gate_counts.most_common())
        lines.append(f"**Withheld ({len(withheld)})** — {gates}")
        for d, e in withheld[:20]:
            reason = d.withheld_by or "unknown"
            detail = ""
            try:
                trail = json.loads(d.gate_trail_json or "[]")
                detail = next(
                    (g.get("detail", "") for g in trail if g.get("gate") == reason),
                    "",
                )
            except (ValueError, AttributeError):
                pass
            lines.append(f"- {e.title} — {e.source} ({reason}{': ' + detail if detail else ''})")
        if len(withheld) > 20:
            lines.append(f"- …and {len(withheld) - 20} more")

    summary = f"Delivered {len(delivered)}, withheld {len(withheld)} in the last {hours}h"
    return summary, "\n".join(lines)


def post_journal_card(db: Session, household_id: str) -> str | None:
    """Compose + post the card. Sync (blocking) — callers thread it."""
    from app.services.inbox_notification_service import post_inbox_item_sync

    composed = compose_journal_card(db, household_id)
    if composed is None:
        return None
    summary, body = composed
    return post_inbox_item_sync(
        household_id=household_id,
        title=f"Attention journal — {datetime.utcnow().strftime('%Y-%m-%d')}",
        summary=summary,
        body=body,
        category=JOURNAL_CATEGORY,
        metadata=None,
        push=False,  # the journal never spends the attention it accounts for
        target_type="household",
    )


def cleanup_expired(db: Session, *, ttl_days: int) -> int:
    """Delete events older than the TTL; deliveries/feedback cascade."""
    cutoff = datetime.utcnow() - timedelta(days=ttl_days)
    count = (
        db.query(AttentionEvent)
        .filter(AttentionEvent.created_at < cutoff)
        .delete(synchronize_session=False)
    )
    db.commit()
    return count

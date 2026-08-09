"""Storage helpers for the 'never suggest this again' blocklist.

Written by the suppress dispatcher handler on a card tap; read by the detector
agent (deterministic source_key skip + descriptors injected into its extraction
prompt) via the node-auth endpoint, and by the mobile management screen (list /
delete). Scoped per (household, user, command).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger("uvicorn")


def record_suppression(
    *,
    household_id: str,
    user_id: int | None,
    command: str,
    source_key: str | None,
    descriptor: str | None,
) -> str | None:
    """Persist a suppression. Deduplicates on (household, user, command,
    source_key) when a source_key is present (re-tapping refreshes the row
    rather than piling up). Returns the row id, or None on error."""
    try:
        from app.db import get_session_local
        from app.models import ProposalSuppression

        db = get_session_local()()
        try:
            existing = None
            if source_key:
                existing = (
                    db.query(ProposalSuppression)
                    .filter(
                        ProposalSuppression.household_id == household_id,
                        ProposalSuppression.user_id == user_id,
                        ProposalSuppression.command == command,
                        ProposalSuppression.source_key == source_key,
                    )
                    .first()
                )
            if existing is not None:
                existing.descriptor = descriptor or existing.descriptor
                existing.created_at = datetime.utcnow()
                db.commit()
                return existing.id
            row = ProposalSuppression(
                household_id=household_id,
                user_id=user_id,
                command=command,
                source_key=source_key or None,
                descriptor=descriptor or None,
                created_at=datetime.utcnow(),
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return row.id
        finally:
            db.close()
    except Exception:  # noqa: BLE001 — never fail a tap on a bookkeeping write
        logger.warning("record_suppression failed", exc_info=True)
        return None


def list_suppressions(
    *, household_id: str, user_id: int | None = None, command: str | None = None
) -> list[dict[str, Any]]:
    """List suppression rows (newest first) for the household, optionally scoped
    to a user and/or command."""
    from app.db import get_session_local
    from app.models import ProposalSuppression

    db = get_session_local()()
    try:
        q = db.query(ProposalSuppression).filter(
            ProposalSuppression.household_id == household_id
        )
        if user_id is not None:
            q = q.filter(ProposalSuppression.user_id == user_id)
        if command is not None:
            q = q.filter(ProposalSuppression.command == command)
        rows = q.order_by(ProposalSuppression.created_at.desc()).all()
        return [
            {
                "id": r.id,
                "command": r.command,
                "source_key": r.source_key,
                "descriptor": r.descriptor,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    finally:
        db.close()


def suppression_signals(
    *, household_id: str, user_id: int | None, command: str
) -> dict[str, list[str]]:
    """The agent's view: the distinct source_keys (deterministic skips) and
    descriptors (negative examples for the prompt) for one (user, command)."""
    rows = list_suppressions(household_id=household_id, user_id=user_id, command=command)
    sources = sorted({r["source_key"] for r in rows if r["source_key"]})
    descriptors = [r["descriptor"] for r in rows if r["descriptor"]]
    return {"source_keys": sources, "descriptors": descriptors}


def delete_suppression(
    *, suppression_id: str, household_id: str, user_id: int | None = None
) -> bool:
    """Un-block: delete one suppression the caller owns. Returns True if a row
    was removed (household — and user when provided — must match)."""
    from app.db import get_session_local
    from app.models import ProposalSuppression

    db = get_session_local()()
    try:
        q = db.query(ProposalSuppression).filter(
            ProposalSuppression.id == suppression_id,
            ProposalSuppression.household_id == household_id,
        )
        if user_id is not None:
            q = q.filter(ProposalSuppression.user_id == user_id)
        row = q.first()
        if row is None:
            return False
        db.delete(row)
        db.commit()
        return True
    finally:
        db.close()

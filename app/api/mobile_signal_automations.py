"""Mobile signal-automations — surface the signal kinds a household can react to,
and let a household admin attach a free-text instruction to each.

    GET  /household/{id}/signal-automations
        The catalog (:mod:`app.services.signal_catalog`), each kind annotated with
        whether it's been *observed* firing for this household + the household's
        current instruction/enabled.
    PUT  /household/{id}/signal-automations/{kind}
        Set {instruction, enabled} for one kind. An empty instruction clears it.

Storage: a single per-household ``signals.automations`` JSON-string setting (a map
``kind -> {"instruction": str, "enabled": bool}``) — no new table. The free-text
instruction is interpreted when the signal fires (execution lives elsewhere); this
router only authors it.

Auth mirrors the household-settings + presence routers: JWT + household role
(member reads, admin writes). Mounted at ``/api/v0/mobile`` (see main.py).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path
from pydantic import BaseModel, Field

from app.deps import AuthenticatedUser, verify_household_role, verify_user_jwt
from app.services import signal_catalog
from app.services.signal_automation_store import load_rules, save_rules

logger = logging.getLogger("uvicorn")

router = APIRouter(tags=["mobile-signal-automations"])

_READ_ROLE = "member"
_WRITE_ROLE = "admin"
# Bounds the free text (it's read at fire time into a prompt) — generous but finite.
_MAX_INSTRUCTION_CHARS = 500


def _observed_kinds(household_id: str) -> set[str]:
    """The distinct signal kinds this household has actually produced (active,
    unexpired) — reuses the ``/signals`` bundle filter. Best-effort: a query error
    just yields an empty set (everything shows as not-yet-observed)."""
    from app.db import get_session_local
    from app.models import Signal

    db = get_session_local()()
    try:
        now = datetime.utcnow()
        rows = (
            db.query(Signal.kind)
            .filter(
                Signal.household_id == household_id,
                Signal.is_active == True,  # noqa: E712
                (Signal.expires_at.is_(None)) | (Signal.expires_at > now),
            )
            .distinct()
            .all()
        )
        return {r[0] for r in rows}
    except Exception:  # noqa: BLE001 — annotation only, never break the listing
        logger.warning("household %s observed-kinds query failed", household_id, exc_info=True)
        return set()
    finally:
        db.close()


class RuleIn(BaseModel):
    instruction: str = Field("", max_length=_MAX_INSTRUCTION_CHARS)
    enabled: bool = Field(True)


@router.get("/household/{household_id}/signal-automations")
async def list_signal_automations(
    household_id: str = Path(..., description="Household UUID"),
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict[str, Any]:
    """The authorable signal catalog + this household's current instruction per kind.

    Any member may read. ``observed`` marks kinds that have actually fired for the
    household (vs. merely possible), so the UI can surface "active" ones first.
    """
    verify_household_role(user.user_id, household_id, required_role=_READ_ROLE)

    rules = load_rules(household_id)
    observed = _observed_kinds(household_id)
    items: list[dict[str, Any]] = []
    for s in signal_catalog.catalog():
        rule = rules.get(s.kind) or {}
        items.append(
            {
                "kind": s.kind,
                "label": s.label,
                "description": s.description,
                "facts": s.facts,
                "example": s.example,
                "source": s.source,
                "observed": s.kind in observed,
                "instruction": rule.get("instruction", ""),
                "enabled": bool(rule.get("enabled", False)),
            }
        )
    return {"household_id": household_id, "automations": items}


@router.put("/household/{household_id}/signal-automations/{kind}")
async def set_signal_automation(
    household_id: str = Path(..., description="Household UUID"),
    kind: str = Path(..., description="Signal kind, e.g. presence.left"),
    body: RuleIn = Body(...),
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> dict[str, Any]:
    """Set (or clear) the free-text instruction for one signal kind. Admin only.

    A blank instruction removes the rule for that kind. Rejects any kind not in the
    catalog (404) so this can't stash arbitrary keys in the setting.
    """
    if not signal_catalog.is_authorable(kind):
        raise HTTPException(
            status_code=404, detail=f"Unknown or non-authorable signal kind: {kind}"
        )
    verify_household_role(user.user_id, household_id, required_role=_WRITE_ROLE)

    instruction = body.instruction.strip()
    rules = load_rules(household_id)
    if not instruction:
        rules.pop(kind, None)  # empty clears the rule
    else:
        rules[kind] = {"instruction": instruction, "enabled": bool(body.enabled)}

    if not save_rules(household_id, rules):
        raise HTTPException(status_code=500, detail="Failed to save automation")

    logger.info(
        "household %s set automation for %s (enabled=%s, %d chars)",
        household_id, kind, body.enabled, len(instruction),
    )
    return {
        "success": True,
        "kind": kind,
        "instruction": instruction,
        "enabled": bool(body.enabled) if instruction else False,
        "cleared": not instruction,
    }

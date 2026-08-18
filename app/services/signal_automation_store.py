"""Shared read/write of the per-household ``signals.automations`` setting.

The setting is a JSON object mapping a signal kind to
``{"instruction": str, "enabled": bool}`` — the free-text rule a household admin
authored on the signal-automations screen. Both the mobile authoring endpoint
(read + write) and the execution reaction (read the enabled instruction) go
through here so the storage shape lives in exactly one place.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("uvicorn")

SETTING_KEY = "signals.automations"


def load_rules(household_id: str) -> dict[str, dict[str, Any]]:
    """Parse the household's rules map. Never raises — a missing/garbage value
    reads as no rules (the caller must still render / stay a no-op)."""
    from app.services.settings_service import get_settings_service

    raw = get_settings_service().get(SETTING_KEY, household_id=str(household_id))
    if not raw:
        return {}
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        logger.warning("household %s has an unparseable %s", household_id, SETTING_KEY)
        return {}


def save_rules(household_id: str, rules: dict[str, dict[str, Any]]) -> bool:
    """Persist the full rules map (JSON-encoded). Returns the settings write result."""
    from app.services.settings_service import get_settings_service

    return bool(
        get_settings_service().set(
            SETTING_KEY, json.dumps(rules), household_id=str(household_id)
        )
    )


def get_enabled_instruction(household_id: str, kind: str) -> str | None:
    """The household's ENABLED, non-blank instruction for ``kind`` — or None if
    there is no rule, it's disabled, or the text is empty. This is the execution
    gate: no enabled instruction ⇒ the automation reaction does nothing."""
    rule = load_rules(household_id).get(kind) or {}
    instruction = (rule.get("instruction") or "").strip()
    return instruction if instruction and rule.get("enabled") else None

"""Shared read/write of the per-household ``signals.automations`` setting.

The setting is a JSON object mapping a signal kind to
``{"instruction": str, "enabled": bool, "delivery": "automatic"|"notification"}``
— the free-text rule a household admin authored on the signal-automations screen,
plus how they want it delivered. Both the mobile authoring endpoint (read + write)
and the execution reaction (read the enabled rule) go through here so the storage
shape lives in exactly one place.

``delivery`` is the USER'S per-rule choice — "automatic" runs the action
immediately, "notification" proposes it as a tap-to-confirm card. It defaults to
the safe "notification" when unset (legacy rules), so nothing runs unattended
unless the user opts into it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("uvicorn")

SETTING_KEY = "signals.automations"

DELIVERY_MODES = ("automatic", "notification")
DEFAULT_DELIVERY = "notification"


def normalize_delivery(value: Any) -> str:
    """Coerce a stored/received delivery value to a valid mode (safe default)."""
    return value if value in DELIVERY_MODES else DEFAULT_DELIVERY


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


def get_enabled_rule(household_id: str, kind: str) -> dict[str, str] | None:
    """The household's ENABLED rule for ``kind`` as
    ``{"instruction": str, "delivery": "automatic"|"notification"}`` — or None if
    there is no rule, it's disabled, or the instruction is blank. This is the
    execution gate: no enabled rule ⇒ the automation reaction does nothing.
    Delivery is normalized to a valid mode (safe default)."""
    rule = load_rules(household_id).get(kind) or {}
    instruction = (rule.get("instruction") or "").strip()
    if not (instruction and rule.get("enabled")):
        return None
    return {"instruction": instruction, "delivery": normalize_delivery(rule.get("delivery"))}

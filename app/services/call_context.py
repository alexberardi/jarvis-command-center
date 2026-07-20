"""Per-user details the call agent may use — "global call context".

A user stores facts once (name, address, callback number, insurance details)
instead of typing them into every call brief. Two independent controls decide
what actually reaches a call, and they answer different questions:

  CATEGORY  — is this field loaded into the brief AT ALL?
              General is always loaded; everything else only when that
              category applies to the call. This is blast radius: an
              insurance policy number is simply absent from a pizza order,
              so no prompt rule has to hold it back.

  TIER      — once loaded, may it be VOLUNTEERED or only given if asked?
              A name is fine to state; a policy number is not, even on a
              call where it is legitimately available.

Both are needed. Category alone would mean selecting Medical lets the agent
open with your policy number; tier alone would mean every secret is one
persuasive callee away on every call.

Storage is a single JSON blob in the user-scoped `phone_calls.call_context`
setting rather than a table: the shape is a short list the user edits as a
unit, and the well-known fields want to change without a migration.

**This is PII.** Anything that reaches a call can be spoken aloud, which puts
it in the transcript and the recording. That is why the default tier for
anything the user adds themselves is "only if asked", and why the eventual
spoken-output guard reuses `restricted_values()` below.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("uvicorn")

SETTING_KEY = "phone_calls.call_context"

# Fixed set, deliberately. If users could invent categories, something would
# have to map an arbitrary label onto a call, which is more classification and
# more ways to be wrong. GENERAL is always included, so the common case needs
# no classification at all.
GENERAL = "general"
MEDICAL = "medical"
AUTO = "auto"
HOME = "home"
FINANCIAL = "financial"
DINING = "dining"

CATEGORIES: tuple[str, ...] = (GENERAL, MEDICAL, AUTO, HOME, FINANCIAL, DINING)

CATEGORY_LABELS: dict[str, str] = {
    GENERAL: "General",
    MEDICAL: "Medical",
    AUTO: "Auto",
    HOME: "Home & trades",
    FINANCIAL: "Financial",
    DINING: "Dining & reservations",
}

# Tiers.
STATE = "state"        # may be said freely
IF_ASKED = "if_asked"  # only when they ask AND it is needed for the task
TIERS: tuple[str, ...] = (STATE, IF_ASKED)


@dataclass(frozen=True)
class FieldSpec:
    """A well-known field: stable key, human label, and its two controls."""

    key: str
    label: str
    category: str
    tier: str


# Well-known fields get fixed labels so the brief reads the same every time
# and the prompt can refer to them. Anything else the user adds is a free
# key/value pair and defaults to IF_ASKED.
WELL_KNOWN_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("full_name", "Full name", GENERAL, STATE),
    FieldSpec("address", "Address", GENERAL, IF_ASKED),
    FieldSpec("callback_number", "Callback number", GENERAL, IF_ASKED),
    FieldSpec("date_of_birth", "Date of birth", MEDICAL, IF_ASKED),
    FieldSpec("insurance_provider", "Insurance provider", MEDICAL, IF_ASKED),
    FieldSpec("insurance_member_id", "Insurance member ID", MEDICAL, IF_ASKED),
    FieldSpec("pharmacy", "Preferred pharmacy", MEDICAL, STATE),
    FieldSpec("vehicle", "Vehicle", AUTO, STATE),
    FieldSpec("license_plate", "License plate", AUTO, IF_ASKED),
)

_WELL_KNOWN_BY_KEY: dict[str, FieldSpec] = {f.key: f for f in WELL_KNOWN_FIELDS}


@dataclass(frozen=True)
class ContextField:
    """One resolved field: a spec plus the user's value."""

    key: str
    label: str
    value: str
    category: str
    tier: str


def _coerce_entry(raw: Any) -> ContextField | None:
    """One stored entry -> ContextField, or None if unusable.

    Deliberately forgiving. This blob is user-edited through an API and a
    mobile grid; one malformed row must not cost the user every other field
    on the call.
    """
    if not isinstance(raw, dict):
        return None
    key = str(raw.get("key") or "").strip()
    value = str(raw.get("value") or "").strip()
    if not key or not value:
        return None

    known = _WELL_KNOWN_BY_KEY.get(key)
    label = str(raw.get("label") or "").strip() or (known.label if known else key)

    category = str(raw.get("category") or "").strip().lower()
    if category not in CATEGORIES:
        category = known.category if known else GENERAL

    tier = str(raw.get("tier") or "").strip().lower()
    if tier not in TIERS:
        # Unknown/missing tier falls to the private side on purpose: a custom
        # field is far more likely to be an account number than a pleasantry.
        tier = known.tier if known else IF_ASKED

    return ContextField(key=key, label=label, value=value, category=category, tier=tier)


def parse_call_context(raw: Any) -> list[ContextField]:
    """Parse the stored blob into fields, dropping anything unusable.

    Accepts the stored JSON as a string or an already-decoded object, since
    settings backends differ on whether they deserialize.
    """
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("call context is not valid JSON — ignoring")
            return []

    entries = raw.get("fields") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return []

    out: list[ContextField] = []
    seen: set[str] = set()
    for entry in entries:
        field = _coerce_entry(entry)
        if field is None or field.key in seen:
            continue
        seen.add(field.key)
        out.append(field)
    return out


def load_call_context(user_id: int | None) -> list[ContextField]:
    """Read a user's stored context. Any failure degrades to no context."""
    if user_id is None:
        return []
    try:
        from app.services.settings_service import get_settings_service

        return parse_call_context(
            get_settings_service().get(SETTING_KEY, user_id=int(user_id))
        )
    except Exception as e:  # noqa: BLE001 — context is an enhancement, never a gate
        logger.warning("call context lookup failed for user %s: %s", user_id, e)
        return []


def select_fields(
    fields: list[ContextField], categories: list[str] | None
) -> list[ContextField]:
    """Fields whose category is General or one of ``categories``.

    Unknown category names are ignored rather than rejected — a caller
    passing junk gets General, not an exception on the plan path.
    """
    wanted = {GENERAL}
    for c in categories or []:
        c = str(c).strip().lower()
        if c in CATEGORIES:
            wanted.add(c)
    return [f for f in fields if f.category in wanted]


def restricted_values(fields: list[ContextField]) -> list[str]:
    """Values that must never be volunteered.

    The spoken-output guard checks candidate speech against these before it
    reaches TTS: the prompt asks the model not to offer them, this is how we
    stop relying on that.
    """
    return [f.value for f in fields if f.tier == IF_ASKED and f.value]


def build_context_block(fields: list[ContextField]) -> str | None:
    """Render selected fields as the brief's caller-details section.

    Two labelled groups, because the prompt's rules refer to them: details it
    may state, and details it may give only if asked. Returns None when there
    is nothing to say, so the brief gains no empty scaffolding.
    """
    statable = [f for f in fields if f.tier == STATE]
    private = [f for f in fields if f.tier == IF_ASKED]
    if not statable and not private:
        return None

    parts: list[str] = []
    if statable:
        lines = "\n".join(f"- {f.label}: {f.value}" for f in statable)
        parts.append(f"About the caller — you may state these:\n{lines}")
    if private:
        lines = "\n".join(f"- {f.label}: {f.value}" for f in private)
        parts.append(
            "Give ONLY if they ask and it is needed to finish this task — "
            f"never volunteer these:\n{lines}"
        )
    return "\n\n".join(parts)

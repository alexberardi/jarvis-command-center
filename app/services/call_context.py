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
import re
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

    Not used on the call path yet — see ``select_for_call``. It is the
    per-call filter that ships once the confirm card can choose categories.
    """
    wanted = {GENERAL}
    for c in categories or []:
        c = str(c).strip().lower()
        if c in CATEGORIES:
            wanted.add(c)
    return [f for f in fields if f.category in wanted]


def select_for_call(fields: list[ContextField]) -> list[ContextField]:
    """What a call loads TODAY: everything, regardless of category.

    The category is captured when the user adds a detail, but it does not
    filter a call yet — per-call category selection (the confirm-card toggle)
    is future work. Loading everything until then is deliberate (Alex,
    2026-07-23): it means categorizing a detail — or the grid defaulting a new
    one to a non-General category — can never silently drop it from a call in
    the meantime. When filtering ships, the call path swaps this for
    ``select_fields(fields, chosen_categories)``; the category data captured
    now is what makes that possible without a backfill.
    """
    return list(fields)


def restricted_fields(fields: list[ContextField]) -> list[ContextField]:
    """Fields that must never be volunteered — the guard's denylist.

    The gateway's spoken-output guard checks candidate speech against these
    before it reaches TTS: the prompt asks the model not to offer them, this
    is how we stop relying on that.

    Returns whole fields, not bare values, because the guard needs the LABEL
    too: deciding whether the callee actually asked for something is a
    question about "Insurance member ID", and the guard is deliberately built
    so that judgement can be made from the label alone — the value itself
    never has to reach the classifier.
    """
    return [f for f in fields if f.tier == IF_ASKED and f.value]


# ---------------------------------------------------------------- write path
#
# Reads parse a stored blob whose keys we wrote. Writes take rows straight from
# the mobile grid, where a custom field is just a label the user typed and a
# value — no key. The key is derived here so `parse_call_context` can stay
# strict (a keyless row is genuinely unusable; a keyless row from the grid just
# needs a key minted first).

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(label: str) -> str:
    return _SLUG_RE.sub("_", label.strip().lower()).strip("_")


def prepare_for_storage(rows: Any) -> list[ContextField]:
    """Grid rows -> canonical fields, ready to serialize and store.

    A row without a key gets one derived from its label; everything else runs
    through the same coercion and first-wins dedup as a read, so what a caller
    gets back from a write is exactly what a later call will see. Returns the
    fields so the endpoint can both persist and echo the canonical result.
    """
    prepared: list[dict] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        row = dict(row)
        if not str(row.get("key") or "").strip():
            row["key"] = _slugify(str(row.get("label") or ""))
        prepared.append(row)
    return parse_call_context({"fields": prepared})


def serialize_fields(fields: list[ContextField]) -> str:
    """Canonical stored blob for a list of fields."""
    return json.dumps(
        {
            "fields": [
                {
                    "key": f.key,
                    "label": f.label,
                    "value": f.value,
                    "category": f.category,
                    "tier": f.tier,
                }
                for f in fields
            ]
        }
    )


def field_to_dict(f: ContextField) -> dict[str, str]:
    """One field as the mobile grid consumes it."""
    return {
        "key": f.key,
        "label": f.label,
        "value": f.value,
        "category": f.category,
        "tier": f.tier,
    }


def catalog() -> dict[str, Any]:
    """The static vocabulary the grid renders, served rather than duplicated.

    The well-known fields it offers as presets, and the full category and tier
    option lists. Kept here so the app cannot drift from the brief: a new
    well-known field, a relabelled category, or a changed default tier reaches
    the grid without an app release.
    """
    return {
        "well_known": [
            {"key": f.key, "label": f.label, "category": f.category, "tier": f.tier}
            for f in WELL_KNOWN_FIELDS
        ],
        "categories": [
            {"value": c, "label": CATEGORY_LABELS[c]} for c in CATEGORIES
        ],
        "tiers": [
            {"value": STATE, "label": "May be said freely"},
            {"value": IF_ASKED, "label": "Only if they ask"},
        ],
    }


def build_context_block(fields: list[ContextField]) -> str | None:
    """Render selected fields as the brief's caller-details section.

    Framed so the model can never read these as its OWN identity. The old
    header ("About the caller — you may state these: Full Name: Alex Berardi")
    made the agent introduce itself AS the caller ("Hi, I'm Alex Berardi")
    on 5/5 openings against the box model — it collapsed "calling on behalf of
    Alex" into "I'm Alex". Two fixes, together taking that to 0/6:

    - ``full_name`` is never PROACTIVELY stated: with it in the "you may state
      these" block the model opened "Hi, I'm Alex Berardi" (5/5). But it IS
      needed for identity-heavy calls (a pharmacy authorization needs the
      patient's full legal name), so it is routed to the give-WHEN-ASKED section
      — the model provides it reactively if the callee asks, never in the
      opening — under the on-behalf-of framing below.
    - the block opens by stating whose details these are and that the model is
      the caller's ASSISTANT, not the caller.

    Returns None when there is nothing to say, so the brief gains no empty
    scaffolding.
    """
    # full_name is the worst identity offender ONLY when stated proactively, so it
    # is forced out of the "state freely" group and into give-when-asked (regardless
    # of its configured tier); everything else follows its own tier.
    statable = [f for f in fields if f.tier == STATE and f.key != "full_name"]
    private = [f for f in fields if f.tier == IF_ASKED or f.key == "full_name"]
    if not statable and not private:
        return None

    # "on behalf of" + "never ask to speak to them": a generic "the person you
    # are calling for" was ambiguous — the model opened with "Can I speak to
    # Alex Berardi?", reading it as calling to REACH the person. This wording
    # holds identity, the who-am-I-calling confusion, and give-when-asked all
    # at once (0/8, 0/8, 5/5 against the box model).
    parts: list[str] = [
        "The details below belong to the person you are calling ON BEHALF OF. "
        "You are their assistant: never claim to be them, never introduce "
        "yourself with their name, and never ask to speak to them."
    ]
    if statable:
        lines = "\n".join(f"- {f.label}: {f.value}" for f in statable)
        parts.append(f"You may state these if it helps:\n{lines}")
    if private:
        lines = "\n".join(f"- {f.label}: {f.value}" for f in private)
        # "Give when asked" framing (not "give ONLY if they ask — never
        # volunteer"): the refusal-forward version made the model refuse a
        # value the callee asked for by name. The guard enforces the "only
        # when asked" half, so the brief safely emphasises giving.
        parts.append(
            "If the business asks for one of these details, give it directly "
            "and accurately — refusing one listed here fails the call. Never "
            f"volunteer them otherwise:\n{lines}"
        )
    return "\n\n".join(parts)

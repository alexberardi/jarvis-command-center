"""The ``<exchange_complete/>`` marker — the model closing its own exchange.

After every reply the node holds the mic open (the follow-up window) to
catch a continuation. That's right for open-ended answers, but for a
terminal reply — "Timer set", "Goodnight!" — the window just sits there
listening to the room, and whatever it hears next has to be argued away
by the not_for_me check.

This marker lets the model say "this exchange is finished": appended as
the last thing in a final reply, stripped before TTS, surfaced to the
node as ``end_of_exchange`` so it skips the follow-up window entirely.
Sitting quietly, by design.

The cost asymmetry is deliberate: a wrongly-closed window costs one
re-wake; the instruction (``core_rules.EXCHANGE_COMPLETE_INSTRUCTION``)
still biases toward omission because cutting off a live exchange is the
worse experience.

``contains_marker`` is lenient (case, separators, missing self-close) for
the same reason the not_for_me detector is — small models deviate.
"""

from __future__ import annotations

import re

MARKER_LITERAL: str = "<exchange_complete/>"

# Case-insensitive; tolerates space/underscore/hyphen separators, optional
# self-close slash, and stray whitespace inside the tag.
_MARKER_RE = re.compile(r"<\s*exchange[\s_\-]*complete\s*/?\s*>", re.IGNORECASE)


def contains_marker(text: str | None) -> bool:
    """True when ``text`` carries the exchange-complete marker."""
    if not text:
        return False
    return _MARKER_RE.search(text) is not None


def strip_marker(text: str | None) -> str:
    """Return ``text`` with every marker occurrence removed and trimmed."""
    if not text:
        return ""
    return _MARKER_RE.sub("", text).strip()


def apply_to_result(result: dict) -> dict:
    """Detect + strip the marker on a final result dict, in place.

    Sets ``result["end_of_exchange"] = True`` and removes the marker from
    ``assistant_message`` so TTS never speaks it. No-op for non-final
    results (``not_for_me`` is already terminal on the node side) and for
    messages without the marker.
    """
    if result.get("stop_reason") == "not_for_me":
        return result
    message = result.get("assistant_message")
    if not contains_marker(message):
        return result
    result["assistant_message"] = strip_marker(message)
    result["end_of_exchange"] = True
    return result

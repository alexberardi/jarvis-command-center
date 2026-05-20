"""Detect the ``<not_for_me/>`` sentinel the LLM emits for ambient speech.

The wake word fires on transcripts that aren't always addressed to Jarvis —
overheard conversation between people, audio from a TV, a phone call in the
room. We instruct the LLM (in ``core_rules.NOT_FOR_ME_INSTRUCTION``) to emit
the ``<not_for_me/>`` marker as its entire response when it concludes the
input wasn't directed at it. ``contains_sentinel`` is the lenient detector
that handles the inevitable small deviations (case, hyphens, whitespace,
or the LLM adding a stray period/quote around it).
"""

import re

SENTINEL_LITERAL: str = "<not_for_me/>"

# Lenient: matches case variations, with/without trailing slash, and
# underscore | dash | space between words. Anchored to angle brackets so we
# don't suppress speech that merely contains the phrase "not for me" in
# prose (which the LLM would say in a real reply).
_SENTINEL_RE: re.Pattern[str] = re.compile(
    r"<\s*not[\s_\-]+for[\s_\-]+me\s*/?\s*>",
    re.IGNORECASE,
)


def contains_sentinel(text: str | None) -> bool:
    """Return True if ``text`` contains the not-for-me sentinel anywhere."""
    if not text:
        return False
    return _SENTINEL_RE.search(text) is not None


def strip_sentinel(text: str | None) -> str:
    """Remove the sentinel from ``text`` and return the rest, stripped."""
    if not text:
        return ""
    return _SENTINEL_RE.sub("", text).strip()

"""Restate profile lines the utterance mentions, inside the user turn.

2026-07-27 prod, "Who is Leo?": the User Profile block sat directly
adjacent to the question — "Has golden doodle dogs named Leo and Groot
who are brothers" — and the quantized no-think 14B still answered "I
don't have any information about someone named Leo in your profile."
Its /think re-check even narrated "let me check the profile" without
reading it. Against a ~38KB system prompt the model's attention on
mid-context facts is unreliable; the only dependable position is inside
the LAST message.

So: deterministic micro-retrieval over the (tiny) profile. When a
content word of the utterance appears in a profile line, restate that
line as a bracketed hint appended to the user message — maximum
recency, one line of token cost, impossible to miss. False positives
are harmless (an extra known-true fact in context); false negatives
just leave today's behavior.
"""

from __future__ import annotations

import re

# Words too common to indicate a real reference to a profile line. Small
# and curated on purpose — over-filtering creates silent no-matches.
_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "who", "what", "whats", "when", "where", "why", "how",
    "is", "are", "was", "were", "does", "did", "can", "could", "will",
    "would", "should", "you", "your", "yours", "know", "tell", "about",
    "for", "with", "that", "this", "these", "those", "have", "has", "had",
    "get", "his", "her", "hers", "their", "our", "one", "name", "named",
    "call", "called", "like", "likes", "any", "some", "all", "not",
    "please", "hey", "okay",
})

_WORD_RE = re.compile(r"[a-z0-9]+")

# Cap how many profile lines a single hint may restate — a broad word
# hitting many lines must not paste the whole profile back into the turn.
MAX_MATCHED_LINES: int = 3


def _content_words(text: str) -> set[str]:
    return {
        w for w in _WORD_RE.findall(text.lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }


def build_profile_match_hint(
    utterance: str | None,
    speaker_block: str | None,
) -> str | None:
    """Return a one-line ``[profile match: ...]`` hint, or None.

    Args:
        utterance: The user's transcribed/typed message.
        speaker_block: The per-turn speaker system message (the output of
            ``build_speaker_block`` — profile items are its "- " lines).

    Returns:
        A bracketed single-line hint restating the matched profile
        line(s), or ``None`` when nothing matches (or inputs are empty).
    """
    if not utterance or not speaker_block:
        return None

    query_words = _content_words(utterance)
    if not query_words:
        return None

    matched: list[str] = []
    for raw_line in speaker_block.splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        item = line[2:].strip()
        if query_words & _content_words(item):
            matched.append(item)
            if len(matched) >= MAX_MATCHED_LINES:
                break

    if not matched:
        return None

    facts = "; ".join(matched)
    return (
        f"[profile match: {facts} — this is stored about this speaker; "
        f"answer from it directly]"
    )

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

# Utterance leads that mean the speaker is ASKING ABOUT a profile fact
# ("who is Leo?", "tell me about Kaitlyn"). Only these keep the strong
# "answer from it directly" instruction — the original 2026-07-27 fix for a
# model that wouldn't read a mid-context profile. Everything else — notably a
# first-person action statement like "I gave Leo his medication" — is a
# request that may need a TOOL, and telling the model to "answer directly"
# there made the native-tools model narrate a fake confirmation instead of
# calling the tool (prod 2026-08: medication doses silently never marked).
_ASK_LEADS: frozenset[str] = frozenset({
    "who", "whos", "whose", "what", "whats", "which", "when", "where", "why",
    "how", "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "will", "would", "should", "have", "has", "had", "am", "may",
    "tell", "describe", "list", "show", "remind", "explain",
})


def _content_words(text: str) -> set[str]:
    return {
        w for w in _WORD_RE.findall(text.lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }


def _asks_about_profile(utterance: str) -> bool:
    """True when the utterance reads as a question about a profile fact.

    Lenient because STT routinely drops the question mark: a leading
    interrogative/info-request word counts too ("who is leo", "tell me …").
    A first-person action statement ("I gave …", "I took …") is neither and
    must not be told to answer from the profile.
    """
    text = utterance.strip()
    if text.endswith("?"):
        return True
    words = _WORD_RE.findall(text.lower())
    return bool(words) and words[0] in _ASK_LEADS


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
    if _asks_about_profile(utterance):
        return (
            f"[profile match: {facts} — this is stored about this speaker; "
            f"answer from it directly]"
        )
    # A statement/command that merely mentions a profile entity ("I gave Leo
    # his medication"): restate the fact for recency (helps pick the right
    # item), but NEVER tell the model to answer from it — it must still run the
    # tool the request calls for, not narrate that it did.
    return (
        f"[profile match: {facts} — this is stored about this speaker; use it "
        f"as context, but still call whatever tool the request needs — don't "
        f"just say you did it]"
    )

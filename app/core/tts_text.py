"""Pre-TTS text sanitation.

Strips characters that don't belong in spoken output regardless of which
LLM produced them — currently emoji, but the same shape works for any
future "looks fine on screen, terrible spoken aloud" cases.

Distinct from `IJarvisPromptProvider.sanitize_text`, which removes
model-specific scaffolding (Qwen3 ``<think>...</think>``, etc.). That
runs FIRST; this runs SECOND on the result. Keeping them separate so
neither concern leaks into the other:

  raw LLM text
    → provider.sanitize_text   # strips model artifacts
    → clean_for_tts            # strips presentation-only chars
    → TTS engine
"""

from __future__ import annotations

import re


# Targeted ranges for the unicode blocks where most emoji live, plus the
# zero-width joiner and variation selector that combine modifiers into
# compound emoji. Avoids broad "category=Symbol" stripping so currency,
# math operators, and similar text-symbols stay intact.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"   # misc symbols & pictographs
    "\U0001F600-\U0001F64F"   # emoticons (😄 lives here at U+1F604)
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F700-\U0001F77F"   # alchemical
    "\U0001F780-\U0001F7FF"   # geometric shapes extended
    "\U0001F800-\U0001F8FF"   # supplemental arrows-C
    "\U0001F900-\U0001F9FF"   # supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"   # chess symbols
    "\U0001FA70-\U0001FAFF"   # symbols & pictographs extended-A
    "\U00002600-\U000026FF"   # misc symbols (☀, ☁, ★, etc.)
    "\U00002700-\U000027BF"   # dingbats (✓, ✗, ✔, ✘, etc.)
    "\U00002B00-\U00002BFF"   # misc symbols & arrows (⭐, ⬆, ⬇, etc.)
    "\U00002300-\U000023FF"   # misc technical (⌚, ⌛, ⏰, ⏱, ⏲, etc.)
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flag halves)
    "\U0000FE0F"              # variation selector-16 (emoji presentation)
    "\U0000200D"              # zero-width joiner (compound emojis)
    "\U000020E3"              # combining enclosing keycap
    "]+",
    flags=re.UNICODE,
)


def clean_for_tts(text: str) -> str:
    """Return ``text`` stripped of emoji + collapsed whitespace.

    Safe to call on any string. Returns the input unchanged when there's
    nothing to strip.
    """
    if not text:
        return text
    cleaned = _EMOJI_RE.sub("", text)
    if cleaned == text:
        return text
    # Collapse any double-spaces an emoji removal may have left behind
    # (e.g. "you 😄 there" → "you  there"). Also trim leading/trailing
    # space introduced if an emoji was at a string boundary.
    cleaned = re.sub(r"  +", " ", cleaned).strip()
    return cleaned

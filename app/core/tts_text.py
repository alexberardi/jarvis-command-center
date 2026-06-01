"""Pre-TTS text sanitation.

Strips characters that don't belong in spoken output regardless of which
LLM produced them — emoji + markdown presentation syntax that TTS would
otherwise read aloud literally ("asterisk asterisk hello asterisk
asterisk").

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


# Markdown patterns applied in order. Each strips presentation syntax
# while preserving the wrapped prose so the TTS still hears the words.
#
# Order matters: triple-fence code blocks before inline backticks;
# ``***bold-italic***`` before ``**bold**`` before ``*italic*`` (otherwise
# the shorter pattern consumes the outer asterisks and orphans the inner
# ones).
_MARKDOWN_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Fenced code blocks ```lang\n...\n``` — DOTALL to span lines, also
    # eats an optional language hint on the opening fence.
    (re.compile(r"```[a-zA-Z0-9_-]*\n?(.*?)```", flags=re.DOTALL), r"\1"),
    # Inline code `...`
    (re.compile(r"`([^`\n]+)`"), r"\1"),
    # Image ![alt](url) — drop entirely (alt text is rarely useful spoken)
    (re.compile(r"!\[[^\]]*\]\([^)]+\)"), ""),
    # Link [text](url) → text
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),
    # Bold-italic ***text***
    (re.compile(r"\*\*\*([^\*\n]+?)\*\*\*"), r"\1"),
    # Bold **text**
    (re.compile(r"\*\*([^\*\n]+?)\*\*"), r"\1"),
    # Italic *text* — single asterisks wrapping a span
    (re.compile(r"\*([^\*\n]+?)\*"), r"\1"),
    # Bold __text__
    (re.compile(r"__([^_\n]+?)__"), r"\1"),
    # Italic _text_ — only when the underscores are at word boundaries,
    # so identifiers like ``user_id`` are left alone.
    (re.compile(r"(?<![A-Za-z0-9])_([^_\n]+?)_(?![A-Za-z0-9])"), r"\1"),
    # Strikethrough ~~text~~
    (re.compile(r"~~([^~\n]+?)~~"), r"\1"),
    # Headings at line start: ``# ``, ``## ``, ...
    (re.compile(r"^[ \t]*#{1,6}[ \t]+", flags=re.MULTILINE), ""),
    # Blockquote markers at line start
    (re.compile(r"^[ \t]*>[ \t]?", flags=re.MULTILINE), ""),
    # Unordered list markers (-, *, +) at line start
    (re.compile(r"^[ \t]*[-\*\+][ \t]+", flags=re.MULTILINE), ""),
    # Ordered list markers (1., 2., ...) at line start
    (re.compile(r"^[ \t]*\d+\.[ \t]+", flags=re.MULTILINE), ""),
    # Horizontal rules — a whole line of ---, ***, or ___
    (re.compile(r"^[ \t]*[-\*_]{3,}[ \t]*$", flags=re.MULTILINE), ""),
)

# Final sweep for orphaned formatting characters the paired patterns above
# didn't catch (e.g. an unclosed ``*hello`` or a stray trailing ``*``).
# Matches a run of ``*``/``~``/`` ` `` glyphs that has a non-word char on
# at least one side, so they're clearly punctuation rather than embedded
# in something like ``5*3`` (which we leave for the TTS to interpret).
_ORPHAN_MARKDOWN_RE = re.compile(r"(?<!\w)[\*~`]+|[\*~`]+(?!\w)")


def clean_for_tts(text: str) -> str:
    """Return ``text`` with emoji, markdown syntax, and double-spaces removed.

    Safe to call on any string. Returns the input unchanged when there's
    nothing to strip.
    """
    if not text:
        return text
    cleaned = _EMOJI_RE.sub("", text)
    for pattern, replacement in _MARKDOWN_SUBS:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = _ORPHAN_MARKDOWN_RE.sub("", cleaned)
    if cleaned == text:
        return text
    # Collapse double-spaces left by removals and trim boundaries.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned

"""Pre-LLM filter for transcripts that aren't actually addressed speech.

Whisper occasionally surfaces non-speech as bracketed action notation —
``*sniff*``, ``*sad noises*``, ``[laughter]``, ``(coughing)``, etc. The
node's own ``_is_non_speech`` catches ``[...]`` and ``(...)`` annotations
but historically let asterisk-bracketed forms through. When they reach the
LLM, the model treats them as text and hallucinates concern ("I smell
something burning" off ``*sniff*`` — see the 2026-06-02 prod incident).

``is_stt_noise`` is the server-side belt + suspenders. It runs before any
LLM call in the voice-command path and short-circuits to ``not_for_me`` so
we (a) don't waste an inference round-trip and (b) deny the model the
opportunity to fabricate a response from a non-utterance.

This is intentionally narrow — it only matches transcripts that consist
ENTIRELY of empty space, fillers, or bracketed annotations. A real
utterance containing the word "sniff" or "[brackets]" mid-sentence is
not affected.
"""

from __future__ import annotations

import re

# A single bracketed-annotation token: ``*sniff*``, ``[laughter]``,
# ``(coughing)``, ``<inaudible>``. Allows whitespace and a single trailing
# punctuation char (``.``/``,``/``!``/``?``/``-``) inside the bracket pair
# because Whisper sometimes emits ``*sad noises.*``.
_BRACKETED_TOKEN = r"[\*\[\(\<][^\*\]\)\>]+[\*\]\)\>]"

# Entire transcript is one or more bracketed tokens separated by whitespace
# or simple punctuation. Allows a trailing period.
_BRACKETED_ONLY_RE: re.Pattern[str] = re.compile(
    rf"^\s*(?:{_BRACKETED_TOKEN}\s*[\.,!?\-]?\s*)+\.?\s*$",
)

# Whisper non-speech fillers that aren't bracketed but still aren't
# commands. Kept very tight on purpose — single-word fillers like "yeah"
# or "okay" are legitimate follow-ups and must NOT match here.
_PURE_FILLER_RE: re.Pattern[str] = re.compile(
    r"^\s*(?:\.{2,}|-{2,}|\?{2,}|!{2,}|…+)\s*$",
)


def is_stt_noise(text: str | None) -> bool:
    """True when ``text`` is a non-utterance the LLM cannot answer.

    Matches:

    - Empty / whitespace-only strings.
    - Transcripts consisting entirely of bracketed action notation:
      ``*sniff*``, ``*sad noises*``, ``[laughter]``, ``(coughing)``,
      ``<inaudible>``. Multiple consecutive tokens count too:
      ``*sniff* *cough*``.
    - Pure punctuation fillers: ``...``, ``--``, ``???``, ``!!!``, ``…``.

    Does NOT match real utterances that happen to contain a bracketed
    segment mid-sentence (``open the *kitchen* light``) — only fully
    bracketed transcripts are noise.
    """
    if not text:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    if _PURE_FILLER_RE.match(stripped):
        return True
    if _BRACKETED_ONLY_RE.match(stripped):
        return True
    return False

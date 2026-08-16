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

This module also hosts the transcript SHAPE detectors the false-wake
defenses share (``is_device_command_shaped``, ``has_multi_speaker_markers``,
``is_short_non_command_fragment``). They classify the transcript's form,
never its meaning, and feed the direction/turn hints — a device-shaped
imperative must never be talked out of by acoustic-side evidence alone
(prod 2026-08-15: two real "turn on the ... lights" commands were
suppressed because wake verification read garbage clips).
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


# Whisper renders multi-speaker dialogue (or TV audio) as dash-prefixed
# speaker turns: ``- Uh-huh. - Eat it.``. A dash counts only when it opens
# the transcript or follows whitespace AND is itself followed by a space, so
# hyphenated words ("Uh-huh") and mid-sentence dashes never match.
_SPEAKER_DASH_RE: re.Pattern[str] = re.compile(r"(?:^|\s)-\s+\S")

# Imperative verbs that open a device-control command. Deliberately the
# household-device vocabulary only — generic verbs ("eat", "get", "take")
# stay out so overheard human imperatives don't earn the guard.
_DEVICE_COMMAND_VERBS = (
    "turn", "switch", "flip", "toggle", "dim", "brighten",
    "lock", "unlock", "set", "start", "stop", "pause", "resume",
    "open", "close", "shut",
)

# Optional politeness/wake-word prefix before the verb: "please ...",
# "jarvis, ...", "hey jarvis ...". Auxiliary-question forms ("can you
# turn ...") are NOT matched — they read as questions and the guard only
# covers bare imperatives.
_DEVICE_COMMAND_RE: re.Pattern[str] = re.compile(
    rf"^\s*(?:hey\s+)?(?:\w+[,.!]\s+)?(?:please,?\s+)?"
    rf"(?:{'|'.join(_DEVICE_COMMAND_VERBS)})\b\s+\S+",
    re.IGNORECASE,
)

_WORD_RE: re.Pattern[str] = re.compile(r"[\w']+")


def has_multi_speaker_markers(text: str | None) -> bool:
    """True when the transcript carries whisper's multi-speaker dash notation.

    ``- Uh-huh. - Eat it.`` → True (two speaker turns). A single leading
    dash (whisper sometimes prefixes lone utterances) is NOT enough — one
    dash is formatting, two is a dialogue.
    """
    if not text:
        return False
    return len(_SPEAKER_DASH_RE.findall(text)) >= 2


def is_device_command_shaped(text: str | None) -> bool:
    """True when the transcript reads as an imperative device command.

    Shape: an imperative device verb ("turn on/off", "switch", "dim",
    "lock/unlock", "set ... to ...", "start/stop", ...) followed by an
    object, single-speaker (no dash markers), and not a question. Matching
    is deliberately generous — this signal only ever PREVENTS a
    suppression-leaning hint (fail-open), so over-matching costs a missed
    hint while under-matching risks silencing a real command (the worst
    outcome; see the 2026-08-15 suppressed lights commands).
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped or "?" in stripped:
        return False
    if has_multi_speaker_markers(stripped):
        return False
    return _DEVICE_COMMAND_RE.match(stripped) is not None


def is_short_non_command_fragment(text: str | None) -> bool:
    """True for ≤2-word fragments that aren't device commands.

    Wake-turn fragments like ``Eat it.`` or ``Oh.`` are the STT-fragment /
    cross-talk signature. Device-shaped two-worders ("stop music") are
    excluded — the imperative guard is senior to junk-shape signals.
    """
    if not text:
        return False
    words = _WORD_RE.findall(text)
    if not words or len(words) > 2:
        return False
    return not is_device_command_shaped(text)


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

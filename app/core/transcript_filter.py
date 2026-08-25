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
``is_short_non_command_fragment``, ``is_music_control_shaped``). They
classify the transcript's form,
never its meaning, and feed the direction/turn hints — a device-shaped
imperative must never be talked out of by acoustic-side evidence alone
(prod 2026-08-15: two real "turn on the ... lights" commands were
suppressed because wake verification read garbage clips).

``response_claims_action`` is the one detector here that reads the MODEL's
reply instead of the user's transcript — the signal that survives when STT
truncates the utterance out from under the shape gates (prod 2026-08-25).
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

# Music-control commands as spoken over the node's OWN music playback. Unlike
# the device-command guard, bare verbs count ("pause", "skip", "next") — with
# music audibly playing there is no ambiguity about the object, and users
# routinely bark the one-word form. Matching stays anchored to a control verb
# opening the utterance (after an optional politeness/wake prefix) so ordinary
# speech that merely CONTAINS "play"/"stop" doesn't earn the directed posture.
# Only consulted when the node reported self-playback (see is_music_control_
# shaped's callers) — it never fires on normal turns.
_MUSIC_CONTROL_RE: re.Pattern[str] = re.compile(
    r"^\s*(?:hey\s+)?(?:\w+[,.!]\s+)?(?:please,?\s+)?"
    r"(?:"
    r"(?:stop|pause|resume|skip|mute|unmute)\b"
    r"|next\b"
    r"|play\b"
    r"|(?:turn|crank)\s+(?:it|that|this|the\s+\w+)\s+(?:up|down)\b"
    r"|turn\s+(?:up|down|off)\s+the\s+\w+"
    r"|volume\s+(?:up|down)\b"
    r"|(?:louder|quieter|softer)\b"
    r"|(?:go\s+back|previous)\b"
    r")",
    re.IGNORECASE,
)


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


# --- Action / report / question shapes (force-tool-calls gate) -------------
#
# DOCTRINE: the model decides when to talk and how to answer; guards force
# tools ONLY where not-acting is a correctness failure — imperative ACTIONS
# ("turn off the lights", "set a timer") and REPORTS that trigger logging
# tools ("Leo took his medicine") — NEVER on questions or conversation.
# Live incident (2026-08-17): "What should I do with Miles today?"
# keyword-matched a calendar command, the force-tool-calls guard popped the
# model's prose answer and [MUST_CALL_RETRY]-ed it into get_calendar_events —
# the whole calendar read aloud in answer to an open question. These
# detectors give the guard a SHAPE requirement on top of its keyword
# requirement. Like every detector in this module they classify FORM, not
# meaning.

# Imperative verbs that open an actionable command. Extends the household-
# device vocabulary with the wider "do something for me" verbs that map to
# tools (timers, reminders, logging, media, communication). Generic
# human-directed verbs ("eat", "get", "take") stay out — overheard family
# imperatives must not earn a forced tool call.
_ACTION_COMMAND_VERBS = _DEVICE_COMMAND_VERBS + (
    "play", "log", "add", "remind", "cancel", "mark", "record", "create",
    "delete", "remove", "schedule", "send", "text", "call", "snooze",
    "skip", "mute", "unmute", "check", "save", "note", "clear",
)

_ACTION_COMMAND_RE: re.Pattern[str] = re.compile(
    rf"^\s*(?:hey\s+)?(?:\w+[,.!]\s+)?(?:please,?\s+)?"
    rf"(?:{'|'.join(_ACTION_COMMAND_VERBS)})\b\s+\S+",
    re.IGNORECASE,
)

# Leading words that mark an utterance as a QUESTION (interrogative or
# auxiliary-fronted). Question shapes never qualify for tool-forcing — the
# model owns how to answer a question.
_QUESTION_LEAD_WORDS = (
    "what", "how", "why", "when", "where", "who", "whose", "which",
    "should", "could", "can", "would", "will", "shall", "may", "might",
    "do", "does", "did", "is", "are", "am", "was", "were", "have", "has",
    "isn't", "aren't", "don't", "doesn't", "didn't", "won't", "wouldn't",
    "couldn't", "shouldn't",
)

_QUESTION_LEAD_RE: re.Pattern[str] = re.compile(
    rf"^\s*(?:hey\s+)?(?:\w+[,.!]\s+)?(?:please,?\s+)?"
    rf"(?:{'|'.join(_QUESTION_LEAD_WORDS)})\b",
    re.IGNORECASE,
)

# Report shapes that feed logging tools: "<subject> took/gave/administered
# <possessive/object> ..." — "Leo took his medicine", "I took my pills",
# "she gave the dog its meds". The verb vocabulary mirrors the medication
# command's declared keywords (took/take my, gave, administered). Kept
# moderately generous on purpose: the force-tool gate ALSO requires a
# command-keyword match, so "I took a walk" never forces anything.
_REPORT_SHAPE_RE: re.Pattern[str] = re.compile(
    r"^\s*(?:hey\s+)?(?:\w+[,.!]\s+)?"
    r"(?:the\s+)?[\w']+\s+"
    r"(?:just\s+|already\s+|finally\s+)?"
    r"(?:took|takes|taken|gave|given|administered|got)\s+"
    r"(?:his|her|their|my|our|its|the|a|an|some|him|them)\b",
    re.IGNORECASE,
)


def is_question_shaped(text: str | None) -> bool:
    """True when the transcript reads as a question.

    A trailing/embedded ``?`` or a leading interrogative / fronted-auxiliary
    word ("what", "how", "should", "can", "is", "do", ...) after an optional
    politeness/wake prefix. Question shapes are EXEMPT from every
    tool-forcing guard: the model decides how to answer a question, even
    when the words keyword-match a command (the 2026-08-17 "What should I
    do with Miles today?" calendar incident).
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if "?" in stripped:
        return True
    return _QUESTION_LEAD_RE.match(stripped) is not None


def is_action_command_shaped(text: str | None) -> bool:
    """True when the transcript reads as an imperative actionable command.

    Generalizes ``is_device_command_shaped`` beyond household-device verbs:
    an imperative action verb ("set", "start", "stop", "turn", "play",
    "pause", "lock", "log", "add", "remind", "cancel", ...) followed by an
    object, single-speaker (no dash markers), and not a question. This is
    the ONLY imperative shape allowed to arm the force-tool-calls guard —
    not-acting on "turn off the living room lights" is a correctness
    failure; not-calling-a-tool on chatter never is.
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if is_question_shaped(stripped):
        return False
    if has_multi_speaker_markers(stripped):
        return False
    return _ACTION_COMMAND_RE.match(stripped) is not None


def is_report_shaped(text: str | None) -> bool:
    """True when the transcript reads as a done-that report for a logger.

    Shape: ``<subject> took/gave/administered <possessive/object> ...`` —
    "Leo took his medicine", "I took my pills". Reports are the second (and
    last) utterance family where a forced tool call is legitimate: failing
    to log a reported dose is a correctness failure. Single-speaker, never
    a question.
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if is_question_shaped(stripped):
        return False
    if has_multi_speaker_markers(stripped):
        return False
    return _REPORT_SHAPE_RE.match(stripped) is not None


# Verb vocabulary for ``response_claims_action`` — the work a TOOL does, in the
# three tenses the model uses to talk about it. Enumerated per-form rather than
# stem+suffix because the useful ones double their final consonant ("logging",
# "setting", "putting") or go irregular ("sent", "put"). Conversational verbs
# ("know", "keep", "think", "hope") are deliberately absent: "Let me know if you
# need anything" and "I'll keep that in mind" must never read as tool work.
_CLAIM_BASE_VERBS = (
    "check", "look", "mark", "log", "record", "add", "set", "remind",
    "schedule", "cancel", "delete", "remove", "turn", "play", "pause",
    "start", "send", "update", "save", "note", "create", "put", "track",
    "book", "order", "text", "message",
)

_CLAIM_PAST_VERBS = (
    "checked", "looked", "marked", "logged", "recorded", "added", "set",
    "reminded", "scheduled", "cancelled", "canceled", "deleted", "removed",
    "turned", "played", "paused", "started", "sent", "updated", "saved",
    "noted", "created", "put", "tracked", "booked", "ordered", "texted",
    "messaged",
)

_CLAIM_ING_VERBS = (
    "checking", "looking", "marking", "logging", "recording", "adding",
    "setting", "reminding", "scheduling", "cancelling", "canceling",
    "deleting", "removing", "turning", "playing", "pausing", "starting",
    "sending", "updating", "saving", "noting", "creating", "putting",
    "tracking", "booking", "ordering", "texting", "messaging",
)

# "I'll check on Leo's meds", "Let me look that up", "I'm going to set a timer".
_CLAIM_INTENT_RE: re.Pattern[str] = re.compile(
    rf"\b(?:i'?ll|i\s+will|i'?m\s+going\s+to|i\s+am\s+going\s+to|let\s+me|i\s+can)\s+"
    rf"(?:go\s+ahead\s+and\s+)?(?:just\s+|now\s+)?"
    rf"(?:{'|'.join(_CLAIM_BASE_VERBS)})\b",
    re.IGNORECASE,
)

# "I've marked it as taken", "Done — I added that", "I just logged it".
_CLAIM_DONE_RE: re.Pattern[str] = re.compile(
    rf"\b(?:i'?ve|i\s+have|i)\s+"
    rf"(?:just\s+|already\s+)?"
    rf"(?:{'|'.join(_CLAIM_PAST_VERBS)})\b",
    re.IGNORECASE,
)

# "Leo's medicine is marked as taken", "That's been scheduled" — the claim
# stated about the thing rather than about the assistant.
_CLAIM_PASSIVE_RE: re.Pattern[str] = re.compile(
    rf"\b(?:is|are|was|were|'s|'re|has|have|had)\s+(?:been\s+)?"
    rf"(?:now\s+|already\s+)?"
    rf"(?:{'|'.join(_CLAIM_PAST_VERBS)})\b",
    re.IGNORECASE,
)

# "I'm marking that now", or a bare gerund opener: "Setting a reminder for 8".
_CLAIM_PROGRESS_RE: re.Pattern[str] = re.compile(
    rf"(?:\bi'?m\s+(?:just\s+|now\s+)?|^\s*)"
    rf"(?:{'|'.join(_CLAIM_ING_VERBS)})\b",
    re.IGNORECASE,
)


def response_claims_action(text: str | None) -> bool:
    """True when the MODEL's own reply promises or claims a tool action.

    Note the inversion: every other detector here classifies the user's
    transcript. This one reads the assistant's response, because that is where
    the signal survives when the transcript does not.

    2026-08-25 prod incident: STT truncated "Leo took his medicine" to "his
    medicine." — the head of the utterance, verb included, was lost. The
    force-tool-calls guard gates on utterance SHAPE, so a bare noun phrase
    scored as neither action- nor report-shaped, the guard stood down, and the
    model's "I'll check on Leo's meds for you." shipped with ``tool_calls: []``.
    Nothing ran and the dose went unlogged. The identical model failure on the
    typed path was caught, because "Leo took his medicine" reads as a report.

    A reply that says it acted, or is about to, while calling no tool is a
    correctness failure no matter how the transcript reads — so this arms the
    guard as a third qualifying shape. It does NOT widen the guard on its own:
    the keyword gate still has to agree, and question-shaped utterances stay
    exempt (the 2026-08-17 calendar incident), so a plain answer to a plain
    question is untouched.
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    return bool(
        _CLAIM_INTENT_RE.search(stripped)
        or _CLAIM_DONE_RE.search(stripped)
        or _CLAIM_PASSIVE_RE.search(stripped)
        or _CLAIM_PROGRESS_RE.search(stripped)
    )

def is_music_control_shaped(text: str | None) -> bool:
    """True when the transcript reads as a music-control command.

    Shape: a music-control verb ("stop", "pause", "skip", "next", "play
    ...", "volume up/down", "turn it up/down", "louder") opening the
    utterance, single-speaker, not a question. Meant for turns where the
    node reported SELF-PLAYBACK — with music coming out of the node's own
    speaker, these shapes are how users talk to Jarvis over the music and
    deserve the directed posture explicitly. Like the device-command guard
    this signal only ever PREVENTS a suppression-leaning hint (fail-open),
    so it errs generous; unlike it, bare verbs ("pause") count because the
    playing music supplies the object.
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped or "?" in stripped:
        return False
    if has_multi_speaker_markers(stripped):
        return False
    return _MUSIC_CONTROL_RE.match(stripped) is not None


# --- Named-person addressing (2026-08-17 prod incident) -------------------
#
# A follow-up window captured a parent calling their child: "already done.
# Wow. Miles, come here." Miles is a KNOWN household member, so the utterance
# is near-certainly meant for him, not Jarvis — shape evidence the turn hint
# can lean on for ANY wake verdict (the incident had wake_verdict=none, so
# the doubt machinery never engaged). Like every shape detector here this
# classifies FORM, not meaning, and only ever feeds a lean hint (fail-open).

# Names that must never count as "another person being addressed": the
# assistant itself and the wake-phrase tokens ("hey jarvis" — "hey" can be a
# member nickname collision only in theory, but a false negative here just
# skips a hint).
_ADDRESSING_EXCLUDED_NAMES = frozenset({"jarvis", "hey"})

# Sentence boundary for per-sentence vocative matching — the incident
# transcript carries the vocative in its THIRD sentence.
_SENTENCE_SPLIT_RE: re.Pattern[str] = re.compile(r"[.!?;]+")

# Human-directed imperative verbs that follow a bare leading name when STT
# drops the vocative comma ("Miles come here"). Deliberately people-verbs —
# overlap with device verbs ("stop") is fine because the device-command guard
# is senior at every call site.
_HUMAN_IMPERATIVE_VERBS = (
    "come", "go", "stop", "wait", "get", "put", "grab", "bring", "take",
    "look", "listen", "sit", "stand", "stay", "hold", "leave", "hurry",
    "let", "help", "eat", "finish", "clean", "pick",
)
_HUMAN_IMPERATIVE_RE_FRAGMENT = "|".join(_HUMAN_IMPERATIVE_VERBS)


def _addressable_name_tokens(member_names: list[str] | None) -> dict[str, str]:
    """Map lowercase first-name tokens → display token for matching.

    Members are stored as display names / usernames ("Miles", "Jess B.");
    the spoken vocative is the first name token. Tokens shorter than 2 chars
    or on the exclusion list (jarvis / wake phrase) are dropped.
    """
    tokens: dict[str, str] = {}
    for name in member_names or []:
        if not isinstance(name, str):
            continue
        words = _WORD_RE.findall(name)
        if not words:
            continue
        token = words[0]
        lowered = token.lower()
        if len(lowered) < 2 or lowered in _ADDRESSING_EXCLUDED_NAMES:
            continue
        tokens.setdefault(lowered, token)
    return tokens


def addressed_household_member(
    text: str | None,
    member_names: list[str] | None,
) -> str | None:
    """Return the household member a sentence in ``text`` directly addresses.

    Vocative shapes, per sentence, case-insensitive, word-boundary:

    - ``NAME, ...`` / ``NAME! ...`` — name opens the sentence with a
      vocative separator ("Miles, come here", "Jess, can you grab that").
    - ``NAME <imperative>`` — leading name straight into a human-directed
      imperative, for STT-dropped commas ("Miles come here").
    - ``..., NAME`` — trailing comma vocative ("come here, Miles").
    - ``<imperative> ... NAME`` — imperative sentence ending on the name
      ("come here Miles").

    Speech merely ABOUT a member ("Miles said he wants pizza") matches no
    shape. Returns the member's first-name token, or None. This is a hint
    signal only — callers must fail open on None.
    """
    if not text:
        return None
    tokens = _addressable_name_tokens(member_names)
    if not tokens:
        return None
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        stripped = sentence.strip()
        if not stripped:
            continue
        for lowered, display in tokens.items():
            escaped = re.escape(lowered)
            if re.match(rf"^{escaped}\s*[,!:]", stripped, re.IGNORECASE):
                return display
            if re.match(
                rf"^{escaped}\s+(?:{_HUMAN_IMPERATIVE_RE_FRAGMENT})\b",
                stripped,
                re.IGNORECASE,
            ):
                return display
            if re.search(rf",\s*{escaped}\s*$", stripped, re.IGNORECASE):
                return display
            if re.match(
                rf"^(?:{_HUMAN_IMPERATIVE_RE_FRAGMENT})\b.*\s{escaped}\s*$",
                stripped,
                re.IGNORECASE,
            ):
                return display
    return None


def is_addressed_to_other_person(
    text: str | None,
    member_names: list[str] | None,
) -> bool:
    """True when ``text`` directly addresses a household member by name.

    Boolean wrapper over ``addressed_household_member`` — see it for the
    matched shapes and the jarvis / wake-phrase exclusion.
    """
    return addressed_household_member(text, member_names) is not None


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

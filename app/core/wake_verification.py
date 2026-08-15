"""Wake-clip verification — did the audio that fired the wake word actually
contain the wake word?

openWakeWord misfires are the one false-wake class no downstream defense can
catch: a high-confidence score (0.95+) in a quiet room looks EXACTLY like a
deliberate wake to every acoustic signal we have, so the not_for_me postures
correctly lean toward answering — and the node ends up acting on speech that
was never addressed to it (prod 2026-08-15: "Leo took his medicine", spoken
family-to-family, marked a medication as taken; "Oh." got a full greeting).

The ground truth exists and was already being shipped: the node sends a
``speaker_audio`` file with every transcription — the ~2s wake-clip snapshot
concatenated with the command audio — for whisper's speaker pass. The wake
word, if genuinely spoken, is in the LEADING seconds of that file. So:

- The media proxy (app/api/media.py) kicks off a background task per wake
  turn: slice the leading seconds, transcribe them (speaker pass off), fuzzy
  check for the wake phrase, and store a verdict in the conversation cache.
- The command turn resolves the verdict with a short bounded wait and either
  short-circuits to not_for_me (mode=enforce) or injects a strong prompt
  hint (mode=bias).

FAIL OPEN everywhere: a missing, pending, or errored verdict means the turn
proceeds as verified. A false silence on a real command is the documented
worse failure (see turn_context.py) — this feature only ever *adds* evidence
that a wake was ambient, never blocks on its own machinery.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import time
import wave
from typing import Any, Optional

from app.core.conversation_cache import conversation_cache

logger = logging.getLogger("uvicorn")

# How much of the leading speaker_audio to transcribe. The node's wake
# snapshot is ~2.0s (core/wake_transcription.py _WAKE_SNAPSHOT_SECONDS); a
# small margin tolerates clock/concat slop without swallowing much command.
WAKE_SLICE_SECONDS = 2.2

# How long the command turn will wait for a pending verdict before failing
# open. The verify transcription is ~2s of audio (a few hundred ms on the
# GPU); the command turn arrives ~50ms after the transcribe response, so the
# verdict is usually pending exactly when we need it.
VERDICT_WAIT_SECONDS = 1.2
_VERDICT_POLL_SECONDS = 0.1


def _get_mode(household_id: str | None = None, node_id: str | None = None) -> str:
    """voice.wake_verification_mode: off | bias | enforce. Fail SAFE to off."""
    try:
        from app.services.settings_service import get_settings_service
        raw = get_settings_service().get(
            "voice.wake_verification_mode", household_id=household_id, node_id=node_id
        )
        mode = str(raw or "").strip().lower()
        return mode if mode in ("off", "bias", "enforce") else "bias"
    except Exception:
        return "off"


def _get_phrase(household_id: str | None = None) -> str:
    try:
        from app.services.settings_service import get_settings_service
        raw = get_settings_service().get(
            "voice.wake_verification_phrase", household_id=household_id
        )
        return str(raw).strip().lower() or "jarvis"
    except Exception:
        return "jarvis"


def slice_leading_wav(wav_bytes: bytes, seconds: float = WAKE_SLICE_SECONDS) -> bytes | None:
    """Return a WAV containing only the leading ``seconds`` of ``wav_bytes``.

    None on any parse failure — the caller fails open.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as src:
            params = src.getparams()
            n_frames = min(src.getnframes(), int(params.framerate * seconds))
            frames = src.readframes(n_frames)
        out = io.BytesIO()
        with wave.open(out, "wb") as dst:
            dst.setnchannels(params.nchannels)
            dst.setsampwidth(params.sampwidth)
            dst.setframerate(params.framerate)
            dst.writeframes(frames)
        return out.getvalue()
    except Exception as e:
        logger.warning("wake-verify: could not slice speaker audio: %s", e)
        return None


def _edit_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein (adjacent transpositions count as ONE edit).

    Plain Levenshtein scores whisper's favorite mangling — a transposed
    syllable like "jarvis"→"travis" (j→t plus ar↔ra) — at 3 and fails the
    match; with transpositions it's the 2 it perceptually is.
    """
    if abs(len(a) - len(b)) > 2:
        return 3  # early out — anything >2 is "no match" to us
    rows: list[list[int]] = [list(range(len(b) + 1))]
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = min(
                rows[i - 1][j] + 1,
                cur[j - 1] + 1,
                rows[i - 1][j - 1] + (ca != cb),
            )
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cost = min(cost, rows[i - 2][j - 2] + 1)
            cur.append(cost)
        rows.append(cur)
    return rows[-1][-1]


def wake_phrase_present(transcript: str | None, phrase: str = "jarvis") -> bool:
    """Fuzzy: does the transcript contain something wake-phrase-shaped?

    Whisper mangles proper nouns on 2s clips ("jarvis" → "travis"/"jervis"/
    "service"), so exact matching would manufacture false silences. A token
    within edit distance 2 of the phrase (or containing it) counts.
    """
    if not transcript:
        return False
    target = phrase.strip().lower()
    if not target:
        return True  # no phrase configured — nothing to verify against
    text = transcript.lower()
    if target in text:
        return True
    for token in re.findall(r"[a-z']+", text):
        if _edit_distance(token, target) <= 2:
            return True
    return False


async def run_wake_verification(
    speaker_audio_bytes: bytes,
    conversation_id: str,
    household_id: str | None,
    node_id: str | None,
    household_member_ids: list | None,
) -> None:
    """Background task: transcribe the leading wake slice and store a verdict.

    Never raises — every failure path stores nothing (downstream fails open).
    """
    try:
        phrase = _get_phrase(household_id)
        clip = slice_leading_wav(speaker_audio_bytes)
        if clip is None:
            return
        from app.core.clients.whisper_client import WhisperClient
        client = WhisperClient(
            household_id=household_id,
            node_id=node_id,
            household_member_ids=household_member_ids,
        )
        t0 = time.time()
        result = await client.transcribe(
            clip, "wake_clip.wav", speaker_recognition=False, timeout=10
        )
        transcript = (result or {}).get("text") or ""
        verified = wake_phrase_present(transcript, phrase)
        verdict = {
            "verified": verified,
            "transcript": transcript.strip(),
            "phrase": phrase,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
        conversation_cache.set_wake_verification(conversation_id, verdict)
        logger.info(
            "🔎 wake-verify | conversation_id=%s verified=%s transcript=%r phrase=%r elapsed_ms=%d",
            conversation_id[:8], verified, verdict["transcript"], phrase,
            verdict["elapsed_ms"],
        )
    except Exception as e:
        logger.warning("wake-verify task failed (failing open): %s", e)


async def resolve_wake_verification(
    conversation_id: str,
    turn_context: Optional[dict],
    household_id: str | None = None,
    node_id: str | None = None,
) -> Optional[dict[str, Any]]:
    """Resolve the wake-clip verdict for THIS turn, with a bounded wait.

    Returns the verdict dict (with an added ``mode`` key) only for wake turns
    with verification enabled AND a resolved verdict; None in every other
    case — including timeout — so callers fail open by treating None as
    verified.
    """
    if (turn_context or {}).get("source") != "wake":
        return None
    mode = _get_mode(household_id, node_id)
    if mode == "off":
        return None

    deadline = time.monotonic() + VERDICT_WAIT_SECONDS
    while True:
        verdict = conversation_cache.get_wake_verification(conversation_id)
        if verdict is not None:
            return {**verdict, "mode": mode}
        if time.monotonic() >= deadline:
            logger.info(
                "🔎 wake-verify | verdict still pending after %.1fs — failing open "
                "(conversation_id=%s)", VERDICT_WAIT_SECONDS, conversation_id[:8],
            )
            return None
        await asyncio.sleep(_VERDICT_POLL_SECONDS)

"""Build a ``[turn context: ...]`` line from the node's turn provenance.

Two very different situations open the mic, and they have opposite
failure costs for the ``<not_for_me/>`` decision:

* **Fresh wake** — the user said the wake word. The wake word itself is
  the addressing signal; OWW's detection score says how clean the fire
  was. A false silence here means the user summoned Jarvis by name and
  got ignored — the worst outcome the system can produce (2026-07-25
  prod incident: "what's my son's name?" → ``<not_for_me/>`` despite the
  memory existing and the recall tool being offered).
* **Follow-up window** — no wake word at all. The node keeps the mic
  open after TTS to catch a continuation of the exchange. A false answer
  here means Jarvis butts into the room's resumed conversation and keeps
  going; a false silence costs one re-wake.

``NOT_FOR_ME_INSTRUCTION`` (in the cached system-prompt prefix) defines a
per-mode posture; this hint line is what selects the mode for the turn.
It rides on the user message — like the direction hint — so the cached
prefix stays byte-identical across turns and the warmup KV cache is
never invalidated.

Older node clients don't send ``turn_source``. Only the wake path
measures ``pre_wake_speech_seconds``, so its presence implies a fresh
wake; with neither signal we emit nothing and behavior is unchanged.
"""

from __future__ import annotations

# OWW detection scores at or above this are treated as a clean, deliberate
# wake ("the user said your wake word"). Below it, the wake may be a false
# fire and the transcript itself has to carry the addressing evidence.
# Prod wakes on real requests routinely score 0.9+; marginal/false fires
# cluster near the node's trigger threshold (~0.5).
WAKE_CONFIDENT_THRESHOLD: float = 0.75

# From this follow-up iteration onward the window has been open long enough
# that the room's conversation has likely resumed — require explicit
# engagement instead of inferring continuation.
FOLLOW_UP_STRICT_ITERATION: int = 3


def build_turn_hint(
    turn_source: str | None,
    wake_confidence: float | None = None,
    follow_up_iteration: int | None = None,
    pre_wake_speech_seconds: float | None = None,
) -> str | None:
    """Return a one-line ``[turn context: ...]`` hint, or None.

    Args:
        turn_source: ``"wake"`` or ``"follow_up"`` when the node reports
            how the mic came to be open. Unrecognized values degrade to
            the inference path (treated as absent).
        wake_confidence: OWW detection score for the wake fire (0-1).
        follow_up_iteration: 1-based iteration of the follow-up window.
        pre_wake_speech_seconds: Pre-wake VAD signal — used only to infer
            wake mode for old clients that don't send ``turn_source``.

    Returns:
        The bracketed hint to append to the user message, or ``None``
        when there is no provenance signal at all (old client, alternate
        entry path) — in which case behavior is identical to before this
        module existed.
    """
    if turn_source == "follow_up":
        return _follow_up_hint(follow_up_iteration)
    if turn_source == "wake":
        return _wake_hint(wake_confidence)
    # No (recognized) source: only the wake path measures pre-wake VAD,
    # so a reported value implies a fresh wake. Never claim a confidence
    # we weren't given.
    if pre_wake_speech_seconds is not None:
        return _wake_hint(None)
    return None


def _wake_hint(wake_confidence: float | None) -> str:
    if wake_confidence is not None and wake_confidence < WAKE_CONFIDENT_THRESHOLD:
        return (
            f"[turn context: wake word fired at low confidence "
            f"({wake_confidence:.2f}) — possibly a false wake. Judge by the "
            f"transcript: a coherent command or question is still for you; "
            f"fragments, half-sentences, or noise are not.]"
        )
    scored = (
        f" (detection confidence {wake_confidence:.2f})"
        if wake_confidence is not None
        else ""
    )
    return (
        f"[turn context: fresh wake — the user said your wake word{scored}. "
        f"This turn is addressed to you: answer it or run the tool it calls "
        f"for. If it asks about the speaker's own life and you don't see the "
        f"fact, call recall — do not go silent. Reserve <not_for_me/> for "
        f"STT artifacts or speech explicitly aimed at another person.]"
    )


def _follow_up_hint(follow_up_iteration: int | None) -> str:
    iteration = max(1, follow_up_iteration or 1)
    escalation = (
        " This window has stayed open across several turns; the room has "
        "likely moved on — respond only to explicit engagement (your name, "
        "a direct question to you, an unmistakable continuation)."
        if iteration >= FOLLOW_UP_STRICT_ITERATION
        else ""
    )
    return (
        f"[turn context: follow-up window, iteration {iteration} — there "
        f"was no wake word; the mic stayed open after your last reply to "
        f"catch a continuation. If this clearly continues your exchange, "
        f"answer. If the room's conversation has moved on without you, emit "
        f"<not_for_me/> — your exchange is over, and going quiet is the "
        f"designed ending, not a failure.{escalation}]"
    )

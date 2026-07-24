"""Build a per-turn ``[voice: ...]`` tone hint from whisper's acoustic affect read.

jarvis-whisper-api (when ``voice.emotion_enabled``) returns an ``affect`` block
describing HOW an utterance was spoken — its arousal/energy and a short
descriptor — which the node forwards on the voice command. We surface it to the
LLM as a single bracketed line appended to the user message, so the model can
fuse "how they sound" with "what they said" and attune its tone. Sibling of
:mod:`app.core.direction_hint` — same shape, same append site, same silent-band
philosophy.

Two guardrails, both non-negotiable:

- **Shape, never announce.** The hint tells the model to let the read colour its
  TONE; it must never say "you sound tired". A false read that quietly warms the
  reply costs nothing; one spoken aloud is creepy and breaks trust.
- **Bias to silence.** Only a confident, actionable read (low or high arousal
  above ``MIN_CONFIDENCE``) emits a hint. Neutral / low-confidence / malformed
  reads emit nothing, exactly like direction_hint's silent middle band.
"""
from __future__ import annotations

from typing import Any

# Independent of whisper's own surface floor (``voice.emotion_min_confidence``):
# whisper decides what to put on the wire, this decides what actually shapes the
# prompt. Kept a hair stricter, to bias toward silence.
MIN_CONFIDENCE: float = 0.5


def build_affect_hint(affect: dict[str, Any] | None) -> str | None:
    """Return a one-line tone hint, or ``None`` when the read isn't actionable.

    Args:
        affect: whisper's affect block ``{read, arousal, confidence}`` forwarded
            by the node, or ``None``.

    Returns:
        A bracketed ``[voice: ...]`` hint to append to the user message, or
        ``None`` (missing / malformed, low confidence, or a neutral read).
    """
    if not isinstance(affect, dict):
        return None

    try:
        confidence = float(affect.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    if confidence < MIN_CONFIDENCE:
        return None

    arousal = affect.get("arousal")
    read_raw = affect.get("read")
    read = read_raw.strip() if isinstance(read_raw, str) else ""
    if not read:
        return None

    if arousal == "low":
        return (
            f"[voice: they sound {read}. Meet them there — a little warmer, "
            f"gentler, and less hurried. Do NOT mention their mood or how they "
            f"sound; just let it shape your tone.]"
        )
    if arousal == "high":
        return (
            f"[voice: they sound {read}. Match their energy — a bit more lively "
            f"and engaged. Do NOT mention their mood or how they sound; just let "
            f"it shape your tone.]"
        )
    # neutral / unknown → no actionable tone adjustment
    return None

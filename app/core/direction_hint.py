"""Build a `[direction hint: ...]` line from the node's pre-wake VAD signal.

The node maintains a rolling RMS ring buffer alongside its wake-detection
loop and reports ``pre_wake_speech_seconds`` — how many seconds of
speech-like audio occurred in the fixed window immediately before the
wake fired. This is a strong external signal about whether the wake was
directed (room was quiet → user just walked up and spoke) vs. ambient
(continuous conversation → wake fired mid-sentence).

We surface it to the LLM as a single bracketed hint appended to the user
message, ONLY when the signal is strong — with one exception: in the
borderline middle band, a LOW-confidence wake fire upgrades the combined
signal to actionable (intermittent room speech + a barely-over-threshold
score is the kitchen-conversation signature; prod 2026-08-15). A middle
band wake at normal confidence stays silent as before.
"""

from __future__ import annotations

# Below this: "quiet before wake" — actively boost confidence in "directed".
# Raised from 0.5s after the 2026-06-02 prod incident where side-conversation
# wakes measured 0.00s and 0.32s (both under the old 0.5s floor) and so got
# the "directed" hint, which biases the prompt toward ANSWER on borderline.
# At 1.5s we still trust truly-quiet rooms (a clear walk-up addressing
# Jarvis) but stop over-claiming "directed" when speech-like audio briefly
# dips below the VAD threshold in the middle of an ongoing conversation.
# The middle band (1.5–4.5s) emits no hint, which is the correct posture
# when the acoustic signal isn't decisive.
QUIET_THRESHOLD_S: float = 1.5
# Above this: "ongoing conversation" — actively boost confidence in not_for_me.
# Set near the top of the window: until we have per-mic calibration, only
# treat near-saturated windows as a strong "ambient" signal. A 2.0s default
# was over-firing because dev-Pi mic baseline RMS exceeds the static node
# VAD threshold even in a quiet room — every wake then got the "ambient"
# hint and the LLM dutifully emitted <not_for_me/>.
ACTIVE_THRESHOLD_S: float = 4.5
# Window the node measures over. Kept here so the hint text matches reality
# even if the node changes its window (we'd update both sides together).
WINDOW_SECONDS: float = 5.0
# Below this OWW score a middle-band wake reads as "marginal fire during
# room speech" and gets the combined-signal hint. Mirrors
# turn_context.WAKE_CONFIDENT_THRESHOLD (not imported — turn_context
# imports from this module; keep the values in sync by hand).
BORDERLINE_CONFIDENCE: float = 0.75


def build_direction_hint(
    pre_wake_speech_seconds: float | None,
    wake_confidence: float | None = None,
    turn_source: str | None = None,
) -> str | None:
    """Return a one-line hint, or None when the signal isn't actionable.

    Args:
        pre_wake_speech_seconds: Seconds of speech-like audio detected in
            the WINDOW_SECONDS window before wake. ``None`` when the node
            didn't report it.
        wake_confidence: OWW score of the wake fire, when known. Only used
            in the ambiguous middle band, where a marginal score upgrades
            the combined signal to an ambient-leaning hint.
        turn_source: Turn provenance ("wake" / "follow_up" / "chat"). The
            combined-signal branch applies to wake turns only.

    Returns:
        A short bracketed hint to append to the user message, or ``None``
        when no hint should be added (signal missing, or ambiguous middle
        band without a marginal wake score).
    """
    if pre_wake_speech_seconds is None:
        return None
    if pre_wake_speech_seconds < QUIET_THRESHOLD_S:
        return (
            f"[direction hint: room was quiet "
            f"({pre_wake_speech_seconds:.1f}s of speech in the "
            f"{WINDOW_SECONDS:.0f}s before wake) — strong signal this is directed at you]"
        )
    if pre_wake_speech_seconds > ACTIVE_THRESHOLD_S:
        return (
            f"[direction hint: continuous speech detected "
            f"({pre_wake_speech_seconds:.1f}s in the {WINDOW_SECONDS:.0f}s before wake) — "
            f"wake may have fired during a conversation between people; "
            f"emit <not_for_me/> unless the transcript is clearly addressed to you]"
        )
    # Ambiguous middle band: the VAD cue alone isn't actionable, but paired
    # with a marginal wake score it is — intermittent room speech + a
    # barely-over-threshold fire is how kitchen conversations wake the node
    # (prod 2026-08-15). Coherence is NOT the discriminator on these turns
    # (overheard speech is usually coherent); ADDRESSING is.
    if (
        turn_source == "wake"
        and wake_confidence is not None
        and wake_confidence < BORDERLINE_CONFIDENCE
    ):
        return (
            f"[direction hint: intermittent speech before wake "
            f"({pre_wake_speech_seconds:.1f}s in the {WINDOW_SECONDS:.0f}s window) "
            f"AND the wake score was marginal ({wake_confidence:.2f}) — this "
            f"pattern usually means the wake fired inside a conversation "
            f"between people. A coherent sentence is NOT evidence it is for "
            f"you: if the transcript reads like people talking to each other "
            f"(replies to something you didn't say, third-person references, "
            f"mid-story fragments, 'we/let's' plans), emit <not_for_me/>. "
            f"Answer only if it plausibly addresses you.]"
        )
    return None

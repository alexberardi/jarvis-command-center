"""Build a `[direction hint: ...]` line from the node's pre-wake VAD signal.

The node maintains a rolling RMS ring buffer alongside its wake-detection
loop and reports ``pre_wake_speech_seconds`` — how many seconds of
speech-like audio occurred in the fixed window immediately before the
wake fired. This is a strong external signal about whether the wake was
directed (room was quiet → user just walked up and spoke) vs. ambient
(continuous conversation → wake fired mid-sentence).

We surface it to the LLM as a single bracketed hint appended to the user
message, ONLY when the signal is strong. The borderline middle band is
deliberately silent so we don't add noise to the prompt for cases where
the cue isn't actionable.
"""

from __future__ import annotations

# Below this: "quiet before wake" — actively boost confidence in "directed".
QUIET_THRESHOLD_S: float = 0.5
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


def build_direction_hint(pre_wake_speech_seconds: float | None) -> str | None:
    """Return a one-line hint, or None when the signal isn't actionable.

    Args:
        pre_wake_speech_seconds: Seconds of speech-like audio detected in
            the WINDOW_SECONDS window before wake. ``None`` when the node
            didn't report it.

    Returns:
        A short bracketed hint to append to the user message, or ``None``
        when no hint should be added (signal missing or in the ambiguous
        middle band).
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
    return None

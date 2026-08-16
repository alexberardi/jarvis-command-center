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

Beyond the VAD cue, two transcript-shape signals feed this hint on wake
turns (both ambient-LEANING only, never a hard suppression instruction):
multi-speaker dash notation ("- Yeah. - Eat it.") and ≤2-word non-command
fragments from an unrecognized speaker. Senior to both is the imperative
device-command guard: a transcript shaped like "turn on the ... lights"
never receives an ambient-leaning hint from acoustic/shape evidence alone
(prod 2026-08-15: two real lights commands were suppressed that way).
"""

from __future__ import annotations

from app.core.transcript_filter import (
    has_multi_speaker_markers,
    is_device_command_shaped,
    is_short_non_command_fragment,
)

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
# room speech" and gets the combined-signal hint. SINGLE SOURCE for the
# confident/marginal wake-score boundary — turn_context imports this as
# WAKE_CONFIDENT_THRESHOLD (turn_context already imports from this module,
# so the dependency direction is fixed).
BORDERLINE_CONFIDENCE: float = 0.75


# Appended to ambient-leaning hints when the speaker's voice matched a
# known household member — a recognized voice is directed-leaning evidence
# that must temper any acoustic ambient signal.
_SPEAKER_KNOWN_NOTE = (
    " Note: the speaker's voice matched a known household member, which "
    "leans toward this being directed at you."
)


def build_direction_hint(
    pre_wake_speech_seconds: float | None,
    wake_confidence: float | None = None,
    turn_source: str | None = None,
    transcript: str | None = None,
    speaker_known: bool | None = None,
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
            combined-signal and transcript-shape branches apply to wake
            turns only.
        transcript: The command transcript, when available. Feeds the
            imperative device-command guard (senior: suppresses every
            ambient-leaning branch) and the junk-shape signals (dash
            markers, short fragments).
        speaker_known: True when the speaker was voice-matched to a
            household member — a directed-leaning input that tempers
            ambient-leaning hints and disables the fragment signal.

    Returns:
        A short bracketed hint to append to the user message, or ``None``
        when no hint should be added (signal missing, or ambiguous middle
        band without a marginal wake score).
    """
    quiet_hint = (
        (
            f"[direction hint: room was quiet "
            f"({pre_wake_speech_seconds:.1f}s of speech in the "
            f"{WINDOW_SECONDS:.0f}s before wake) — strong signal this is directed at you]"
        )
        if (
            pre_wake_speech_seconds is not None
            and pre_wake_speech_seconds < QUIET_THRESHOLD_S
        )
        else None
    )

    # IMPERATIVE DEVICE-COMMAND GUARD (senior to every ambient-leaning
    # signal below): "turn on the living room lights" must never receive a
    # suppression-leaning hint from acoustic-side evidence alone. Directed
    # evidence (quiet room) still surfaces; everything ambient-leaning is
    # muted (prod 2026-08-15: two real lights commands suppressed by wake
    # verification reading garbage clips).
    if transcript is not None and is_device_command_shaped(transcript):
        return quiet_hint

    ambient_note = _SPEAKER_KNOWN_NOTE if speaker_known else ""

    # Transcript-shape signals (wake turns only). Both are LEANING hints —
    # they present evidence and leave the call to the model; neither ever
    # instructs a hard suppression. They outrank the VAD branches because
    # the pre-wake VAD signal can flatline (prod: 0.00 for 14 days straight)
    # and a dead-quiet reading must not mask direct transcript evidence of
    # a dialogue.
    if turn_source == "wake" and transcript:
        if has_multi_speaker_markers(transcript):
            return (
                f"[direction hint: the transcript contains multi-speaker "
                f"dialogue markers (dash-prefixed turns) — this usually "
                f"means the mic caught people (or a TV) talking to each "
                f"other, which leans toward <not_for_me/>. Still answer if "
                f"the content is unmistakably addressed to you."
                f"{ambient_note}]"
            )
        if speaker_known is not True and is_short_non_command_fragment(
            transcript
        ):
            return (
                "[direction hint: the transcript is a very short fragment "
                "from an unrecognized speaker — on a wake turn this is "
                "usually an STT fragment or overheard cross-talk, which "
                "leans toward <not_for_me/>. Still answer if it is a real "
                "request to you.]"
            )

    if pre_wake_speech_seconds is None:
        return None
    if quiet_hint is not None:
        return quiet_hint
    if pre_wake_speech_seconds > ACTIVE_THRESHOLD_S:
        return (
            f"[direction hint: continuous speech detected "
            f"({pre_wake_speech_seconds:.1f}s in the {WINDOW_SECONDS:.0f}s before wake) — "
            f"wake may have fired during a conversation between people; "
            f"emit <not_for_me/> unless the transcript is clearly addressed to you."
            f"{ambient_note}]"
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
            f"Answer only if it plausibly addresses you.{ambient_note}]"
        )
    return None

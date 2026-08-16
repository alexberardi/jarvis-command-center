"""Tests for build_direction_hint — the pre-wake VAD → prompt-hint mapping."""

from app.core.direction_hint import (
    ACTIVE_THRESHOLD_S,
    QUIET_THRESHOLD_S,
    WINDOW_SECONDS,
    build_direction_hint,
)


class TestBuildDirectionHint:
    def test_none_signal_returns_none(self):
        # No signal from node → no hint added (older clients, mobile, etc.)
        assert build_direction_hint(None) is None

    def test_quiet_room_emits_hint(self):
        hint = build_direction_hint(0.0)
        assert hint is not None
        assert "quiet" in hint.lower()
        assert "directed at you" in hint.lower()

    def test_quiet_below_threshold(self):
        # Just under QUIET_THRESHOLD_S should still emit the "quiet" hint
        hint = build_direction_hint(QUIET_THRESHOLD_S - 0.01)
        assert hint is not None
        assert "quiet" in hint.lower()

    def test_active_room_emits_not_for_me_hint(self):
        hint = build_direction_hint(ACTIVE_THRESHOLD_S + 0.5)
        assert hint is not None
        assert "<not_for_me/>" in hint
        assert "conversation" in hint.lower()

    def test_ambiguous_middle_returns_none(self):
        # The middle band is intentionally silent so we don't add prompt
        # noise for cases where the cue isn't actionable.
        mid = (QUIET_THRESHOLD_S + ACTIVE_THRESHOLD_S) / 2
        assert build_direction_hint(mid) is None

    def test_at_quiet_threshold_is_ambiguous(self):
        # Exactly at the threshold: not strictly less than QUIET → no hint
        assert build_direction_hint(QUIET_THRESHOLD_S) is None

    def test_at_active_threshold_is_ambiguous(self):
        # Exactly at the threshold: not strictly greater than ACTIVE → no hint
        assert build_direction_hint(ACTIVE_THRESHOLD_S) is None

    def test_hint_mentions_window_size(self):
        # Both hint branches should reference the measurement window so the
        # LLM can interpret the number meaningfully.
        quiet_hint = build_direction_hint(0.0)
        active_hint = build_direction_hint(ACTIVE_THRESHOLD_S + 1.0)
        assert quiet_hint is not None and f"{WINDOW_SECONDS:.0f}s" in quiet_hint
        assert active_hint is not None and f"{WINDOW_SECONDS:.0f}s" in active_hint

    def test_hint_includes_the_measured_value(self):
        # Sanity: the actual measured number is in the hint text so the
        # LLM can weigh marginal vs. severe cases. Use a value above the
        # active threshold so the hint fires.
        measured = ACTIVE_THRESHOLD_S + 0.3
        hint = build_direction_hint(measured)
        assert hint is not None
        assert f"{measured:.1f}" in hint


class TestMiddleBandLowConfidence:
    """The combined-signal branch: ambiguous VAD + marginal wake score →
    ambient-leaning hint (the kitchen-conversation signature). Middle band
    stays silent at normal confidence, for follow-up turns, and when the
    score is unknown."""

    def test_middle_band_low_confidence_wake_gets_ambient_hint(self):
        hint = build_direction_hint(3.0, wake_confidence=0.45, turn_source="wake")
        assert hint is not None
        assert "<not_for_me/>" in hint
        assert "marginal" in hint

    def test_middle_band_confident_wake_stays_silent(self):
        assert build_direction_hint(3.0, wake_confidence=0.9, turn_source="wake") is None

    def test_middle_band_unknown_confidence_stays_silent(self):
        assert build_direction_hint(3.0, turn_source="wake") is None

    def test_middle_band_follow_up_stays_silent(self):
        assert build_direction_hint(3.0, wake_confidence=0.45, turn_source="follow_up") is None

    def test_quiet_room_unaffected_by_low_confidence(self):
        hint = build_direction_hint(0.5, wake_confidence=0.45, turn_source="wake")
        assert hint is not None
        assert "directed at you" in hint

    def test_active_band_unaffected(self):
        hint = build_direction_hint(4.9, wake_confidence=0.9, turn_source="wake")
        assert hint is not None
        assert "conversation between people" in hint


class TestImperativeDeviceCommandGuard:
    """Item 2 of the dual false-wake defense: a device-shaped imperative
    ("turn on the ... lights") must never receive an ambient-leaning hint
    from acoustic-side evidence alone (prod 2026-08-15: two real lights
    commands were suppressed that way)."""

    def test_device_command_mutes_ambient_vad_hint(self):
        hint = build_direction_hint(
            ACTIVE_THRESHOLD_S + 0.5,
            turn_source="wake",
            transcript="Turn on the living room lights.",
        )
        assert hint is None

    def test_device_command_mutes_combined_signal_hint(self):
        # Middle band + marginal score would emit the kitchen-conversation
        # hint — but not for a device-shaped imperative.
        hint = build_direction_hint(
            3.0,
            wake_confidence=0.45,
            turn_source="wake",
            transcript="Turn on the playroom lights.",
        )
        assert hint is None

    def test_device_command_keeps_the_directed_quiet_hint(self):
        # Directed evidence still surfaces — only ambient-leaning is muted.
        hint = build_direction_hint(
            0.0, turn_source="wake", transcript="Turn on the living room lights."
        )
        assert hint is not None
        assert "directed at you" in hint

    def test_non_device_transcript_keeps_ambient_hint(self):
        hint = build_direction_hint(
            ACTIVE_THRESHOLD_S + 0.5,
            turn_source="wake",
            transcript="and then we went to the store",
        )
        assert hint is not None
        assert "<not_for_me/>" in hint


class TestJunkShapeHints:
    """Item 6: transcript-shape signals on wake turns. Both are ambient-
    LEANING only — they present evidence, never instruct a hard
    suppression — and the imperative guard is senior to them."""

    def test_multi_speaker_markers_emit_leaning_hint(self):
        hint = build_direction_hint(
            None, turn_source="wake", transcript="- Uh-huh. - Eat it."
        )
        assert hint is not None
        assert "leans toward <not_for_me/>" in hint
        # Leaning, not a hard gate: the model keeps an explicit answer path.
        assert "Still answer" in hint

    def test_dash_hint_fires_even_when_vad_reads_quiet(self):
        # The pre-wake VAD flatlined at 0.00 for 14 days in prod — a dead
        # "quiet" reading must not mask direct transcript evidence of a
        # dialogue.
        hint = build_direction_hint(
            0.0, turn_source="wake", transcript="- Uh-huh. - Eat it."
        )
        assert hint is not None
        assert "multi-speaker" in hint

    def test_short_fragment_unknown_speaker_emits_leaning_hint(self):
        hint = build_direction_hint(
            None, turn_source="wake", transcript="Eat it.", speaker_known=False
        )
        assert hint is not None
        assert "leans toward <not_for_me/>" in hint
        assert "Still answer" in hint

    def test_short_fragment_known_speaker_stays_silent(self):
        # Speaker-known is a directed-leaning input: a recognized household
        # voice disables the fragment signal.
        hint = build_direction_hint(
            None, turn_source="wake", transcript="Eat it.", speaker_known=True
        )
        assert hint is None

    def test_fragment_hint_only_on_wake_turns(self):
        assert (
            build_direction_hint(
                None, turn_source="follow_up", transcript="Eat it."
            )
            is None
        )

    def test_full_sentence_gets_no_junk_hint(self):
        assert (
            build_direction_hint(
                None, turn_source="wake", transcript="what's the weather today"
            )
            is None
        )

    def test_device_shaped_two_worder_gets_no_fragment_hint(self):
        # Imperative guard senior to the fragment signal.
        assert (
            build_direction_hint(
                None, turn_source="wake", transcript="stop music"
            )
            is None
        )


class TestSpeakerKnownLean:
    def test_ambient_vad_hint_carries_speaker_known_note(self):
        hint = build_direction_hint(
            ACTIVE_THRESHOLD_S + 0.5,
            turn_source="wake",
            transcript="and then we went to the store",
            speaker_known=True,
        )
        assert hint is not None
        assert "household member" in hint

    def test_ambient_vad_hint_without_speaker_has_no_note(self):
        hint = build_direction_hint(
            ACTIVE_THRESHOLD_S + 0.5,
            turn_source="wake",
            transcript="and then we went to the store",
            speaker_known=False,
        )
        assert hint is not None
        assert "household member" not in hint

    def test_combined_signal_hint_carries_speaker_known_note(self):
        hint = build_direction_hint(
            3.0,
            wake_confidence=0.45,
            turn_source="wake",
            transcript="and then we went to the store",
            speaker_known=True,
        )
        assert hint is not None
        assert "household member" in hint

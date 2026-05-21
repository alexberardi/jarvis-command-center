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

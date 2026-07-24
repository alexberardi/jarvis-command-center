"""Tests for the per-turn affect (tone) hint builder.

The two non-negotiable guardrails: SHAPE-don't-announce (every hint forbids
saying "you sound tired"), and BIAS-to-silence (only a confident low/high read
emits anything — neutral / low-confidence / malformed reads are silent, like
direction_hint's middle band).
"""
from app.core.affect_hint import MIN_CONFIDENCE, build_affect_hint


def _affect(arousal="low", confidence=0.8,
            read="subdued / low-energy — flat, even pitch"):
    return {"read": read, "arousal": arousal, "confidence": confidence}


class TestBuildAffectHint:
    def test_none_returns_none(self):
        assert build_affect_hint(None) is None

    def test_non_dict_returns_none(self):
        assert build_affect_hint("high") is None
        assert build_affect_hint(["low"]) is None

    def test_low_arousal_emits_gentle_hint(self):
        out = build_affect_hint(_affect(arousal="low"))
        assert out is not None
        assert out.startswith("[voice:") and out.endswith("]")
        assert "warmer" in out or "gentler" in out
        assert "subdued" in out  # the descriptor is carried into the hint

    def test_high_arousal_emits_energy_hint(self):
        out = build_affect_hint(_affect(
            arousal="high", read="animated / energized — animated pitch"))
        assert out is not None
        assert "energy" in out.lower() or "lively" in out
        assert "animated" in out

    def test_neutral_returns_none(self):
        # Neutral isn't actionable — silent, like direction_hint's middle band.
        assert build_affect_hint(_affect(arousal="neutral")) is None

    def test_unknown_arousal_returns_none(self):
        assert build_affect_hint(_affect(arousal="unknown")) is None

    def test_below_min_confidence_returns_none(self):
        assert build_affect_hint(_affect(confidence=MIN_CONFIDENCE - 0.01)) is None

    def test_at_min_confidence_emits(self):
        assert build_affect_hint(_affect(confidence=MIN_CONFIDENCE)) is not None

    def test_empty_or_missing_read_returns_none(self):
        assert build_affect_hint(_affect(read="")) is None
        assert build_affect_hint({"arousal": "low", "confidence": 0.9}) is None

    def test_malformed_confidence_returns_none(self):
        assert build_affect_hint({"arousal": "low", "read": "x", "confidence": "high"}) is None

    def test_missing_confidence_defaults_to_withheld(self):
        # No confidence key → 0.0 → below floor → silent.
        assert build_affect_hint({"arousal": "low", "read": "x"}) is None

    def test_shape_dont_announce_guardrail(self):
        # THE guardrail: every emitted hint must forbid announcing the mood and
        # scope itself to tone only.
        for arousal in ("low", "high"):
            out = build_affect_hint(_affect(arousal=arousal))
            assert "Do NOT mention their mood" in out
            assert "shape your tone" in out

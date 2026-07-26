"""Tests for build_turn_hint — turn provenance → per-turn not_for_me posture.

Two very different situations open the mic, with opposite failure costs:

* Fresh wake — the user said the wake word. False silence here means the
  user summoned Jarvis by name and got stonewalled (the 2026-07-25 prod
  incident: "what's my son's name?" → <not_for_me/> despite the memory
  existing and the recall tool being offered).
* Follow-up window — no wake word; the mic stayed open after TTS. False
  answers here mean Jarvis butts into the room's resumed conversation.

The hint line selects the mode per turn; NOT_FOR_ME_INSTRUCTION (in the
cached prefix) defines what each mode means.
"""

from app.core.prompt_providers.shared.core_rules import NOT_FOR_ME_INSTRUCTION
from app.core.turn_context import (
    FOLLOW_UP_STRICT_ITERATION,
    WAKE_CONFIDENT_THRESHOLD,
    build_turn_hint,
)


class TestWakeMode:
    def test_confident_wake_is_addressed(self):
        hint = build_turn_hint("wake", wake_confidence=0.95)
        assert hint is not None
        assert hint.startswith("[turn context:")
        assert hint.endswith("]")
        assert "fresh wake" in hint.lower()
        assert "0.95" in hint
        # The whole point: a confident wake is addressed to Jarvis.
        assert "addressed to you" in hint.lower()

    def test_confident_wake_routes_memory_to_recall(self):
        # Regression pin for the prod incident: on a fresh wake, a personal
        # question the model can't ground must route to recall, not silence.
        hint = build_turn_hint("wake", wake_confidence=0.9)
        assert "recall" in hint
        assert "silent" in hint.lower() or "silence" in hint.lower()

    def test_confident_wake_does_not_claim_false_wake(self):
        hint = build_turn_hint("wake", wake_confidence=0.9)
        assert "false wake" not in hint.lower()
        assert "low confidence" not in hint.lower()

    def test_low_confidence_wake_flags_possible_false_fire(self):
        hint = build_turn_hint(
            "wake", wake_confidence=WAKE_CONFIDENT_THRESHOLD - 0.2
        )
        assert hint is not None
        assert "low confidence" in hint.lower()
        assert f"{WAKE_CONFIDENT_THRESHOLD - 0.2:.2f}" in hint

    def test_low_confidence_wake_still_answers_coherent_speech(self):
        # A marginal wake with a well-formed question is still a request —
        # the transcript decides, not the score alone.
        hint = build_turn_hint("wake", wake_confidence=0.4)
        assert "question" in hint.lower() or "command" in hint.lower()

    def test_threshold_boundary_is_confident(self):
        hint = build_turn_hint("wake", wake_confidence=WAKE_CONFIDENT_THRESHOLD)
        assert "low confidence" not in hint.lower()

    def test_wake_without_score_is_still_wake(self):
        # Older node clients may send turn_source without a score.
        hint = build_turn_hint("wake")
        assert hint is not None
        assert "fresh wake" in hint.lower()
        assert "addressed to you" in hint.lower()


class TestFollowUpMode:
    def test_follow_up_reverses_the_burden(self):
        hint = build_turn_hint("follow_up", follow_up_iteration=1)
        assert hint is not None
        assert hint.startswith("[turn context:")
        assert "no wake word" in hint.lower()
        assert "<not_for_me/>" in hint

    def test_follow_up_defaults_iteration_to_one(self):
        assert build_turn_hint("follow_up") == build_turn_hint(
            "follow_up", follow_up_iteration=1
        )

    def test_early_iterations_do_not_escalate(self):
        hint = build_turn_hint(
            "follow_up", follow_up_iteration=FOLLOW_UP_STRICT_ITERATION - 1
        )
        assert "explicit" not in hint.lower()

    def test_late_iterations_require_explicit_engagement(self):
        hint = build_turn_hint(
            "follow_up", follow_up_iteration=FOLLOW_UP_STRICT_ITERATION
        )
        assert "explicit" in hint.lower()

    def test_follow_up_mentions_iteration_number(self):
        hint = build_turn_hint("follow_up", follow_up_iteration=2)
        assert "2" in hint

    def test_explicit_follow_up_beats_wake_inference(self):
        # If the node says follow_up, a stray pre_wake measurement must not
        # flip the mode back to wake.
        hint = build_turn_hint(
            "follow_up", follow_up_iteration=1, pre_wake_speech_seconds=0.0
        )
        assert "fresh wake" not in hint.lower()
        assert "no wake word" in hint.lower()


class TestInference:
    """Old node clients don't send turn_source. Only the wake path measures
    pre_wake_speech_seconds, so its presence implies a fresh wake."""

    def test_no_signal_returns_none(self):
        # Nothing to go on → no hint → behavior identical to before.
        assert build_turn_hint(None) is None

    def test_pre_wake_present_infers_wake(self):
        hint = build_turn_hint(None, pre_wake_speech_seconds=0.0)
        assert hint is not None
        assert "fresh wake" in hint.lower()

    def test_inferred_wake_makes_no_confidence_claim(self):
        hint = build_turn_hint(None, pre_wake_speech_seconds=0.0)
        assert "confidence" not in hint.lower()

    def test_unknown_source_falls_back_to_inference(self):
        # A future node sending a source we don't recognize must degrade
        # gracefully, not crash or mislabel.
        assert build_turn_hint("barge_in") is None
        hint = build_turn_hint("barge_in", pre_wake_speech_seconds=0.0)
        assert hint is not None and "fresh wake" in hint.lower()


class TestInstructionIntegration:
    """The cached-prefix instruction must explain the line this module emits."""

    def test_instruction_documents_turn_context_line(self):
        assert "[turn context:" in NOT_FOR_ME_INSTRUCTION

    def test_instruction_defines_both_modes(self):
        body = NOT_FOR_ME_INSTRUCTION.lower()
        assert "fresh wake" in body
        assert "follow-up" in body

    def test_wake_mode_kills_the_no_grounding_trigger(self):
        # The prod incident root cause: silence-trigger #3 ("no grounding")
        # fired on personal-fact questions. On a fresh wake it must not
        # apply, and the speaker's own life must always route to recall.
        body = NOT_FOR_ME_INSTRUCTION.lower()
        assert "own life" in body
        assert "recall" in body

    def test_follow_up_mode_frames_silence_as_designed(self):
        # In the follow-up window, going quiet is the intended end of the
        # exchange — the model must not read it as a failure to help.
        body = NOT_FOR_ME_INSTRUCTION.lower()
        assert "exchange" in body

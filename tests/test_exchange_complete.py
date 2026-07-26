"""Tests for the <exchange_complete/> marker — the model closing its own exchange.

The follow-up window's job is to catch continuations; when the model knows
its reply is terminal ("Timer set", "Goodnight!"), it can say so and the
node closes the mic instead of listening for 4+ seconds. False positives
are cheap by design: the user just re-wakes.
"""

from app.core.exchange_complete import (
    MARKER_LITERAL,
    apply_to_result,
    contains_marker,
    strip_marker,
)
from app.core.prompt_providers.shared.core_rules import (
    EXCHANGE_COMPLETE_INSTRUCTION,
    NOT_FOR_ME_INSTRUCTION,
)
from app.core.tts_text import clean_for_tts


class TestContainsMarker:
    def test_canonical_form(self):
        assert contains_marker("Goodnight! <exchange_complete/>")

    def test_no_self_close(self):
        assert contains_marker("Done. <exchange_complete>")

    def test_case_and_separator_variants(self):
        assert contains_marker("<Exchange_Complete/>")
        assert contains_marker("<exchange-complete/>")
        assert contains_marker("<exchange complete/>")

    def test_literal_matches_detector(self):
        assert contains_marker(MARKER_LITERAL)

    def test_prose_does_not_trigger(self):
        # The phrase in prose must not close the window.
        assert not contains_marker("The exchange is complete, want a summary?")

    def test_empty_and_none(self):
        assert not contains_marker("")
        assert not contains_marker(None)


class TestStripMarker:
    def test_strip_trailing(self):
        assert strip_marker("Timer set. <exchange_complete/>") == "Timer set."

    def test_strip_marker_only(self):
        assert strip_marker("<exchange_complete/>") == ""

    def test_untouched_without_marker(self):
        assert strip_marker("Timer set.") == "Timer set."

    def test_none_returns_empty(self):
        assert strip_marker(None) == ""


class TestApplyToResult:
    def test_marks_and_strips_final_reply(self):
        result = {"stop_reason": "complete", "assistant_message": "Goodnight! <exchange_complete/>"}
        out = apply_to_result(result)
        assert out["end_of_exchange"] is True
        assert out["assistant_message"] == "Goodnight!"

    def test_no_marker_no_flag(self):
        result = {"stop_reason": "complete", "assistant_message": "It's 72 and sunny."}
        out = apply_to_result(result)
        assert "end_of_exchange" not in out
        assert out["assistant_message"] == "It's 72 and sunny."

    def test_not_for_me_result_untouched(self):
        # A not_for_me abort is already terminal on the node side; the
        # marker path must not rewrite its (empty) message semantics.
        result = {"stop_reason": "not_for_me", "assistant_message": "<not_for_me/>"}
        out = apply_to_result(result)
        assert "end_of_exchange" not in out

    def test_handles_missing_message(self):
        result = {"stop_reason": "complete"}
        out = apply_to_result(result)
        assert "end_of_exchange" not in out


class TestTTSNeverSpeaksTheMarker:
    def test_clean_for_tts_strips_marker(self):
        # Streaming paths bypass apply_to_result — the shared TTS cleaner
        # is the backstop that keeps the token out of spoken audio.
        cleaned = clean_for_tts("Timer set. <exchange_complete/>")
        assert "exchange_complete" not in cleaned
        assert "Timer set." in cleaned


class TestInstruction:
    def test_instruction_mentions_marker(self):
        assert "<exchange_complete/>" in EXCHANGE_COMPLETE_INSTRUCTION

    def test_instruction_biases_against_overuse(self):
        # Cutting off a live exchange is the costlier error inside the
        # instruction's own scope; when unsure the model must omit.
        body = EXCHANGE_COMPLETE_INSTRUCTION.lower()
        assert "omit" in body or "when unsure" in body or "in doubt" in body

    def test_instruction_forbids_after_questions(self):
        body = EXCHANGE_COMPLETE_INSTRUCTION.lower()
        assert "question" in body

    def test_distinct_from_not_for_me(self):
        # Two different terminals: not_for_me = "this wasn't for me at
        # all"; exchange_complete = "this was for me and it's finished".
        assert "<exchange_complete/>" not in NOT_FOR_ME_INSTRUCTION

"""Tests for the not-for-me sentinel detector and prompt integration."""

from app.core.not_for_me import (
    SENTINEL_LITERAL,
    contains_sentinel,
    strip_sentinel,
)
from app.core.prompt_providers.shared.core_rules import NOT_FOR_ME_INSTRUCTION


class TestContainsSentinel:
    def test_canonical_form(self):
        assert contains_sentinel("<not_for_me/>")

    def test_no_self_close(self):
        assert contains_sentinel("<not_for_me>")

    def test_uppercase(self):
        assert contains_sentinel("<NOT_FOR_ME/>")

    def test_mixed_case(self):
        assert contains_sentinel("<Not_For_Me/>")

    def test_internal_whitespace(self):
        assert contains_sentinel("<not for me/>")

    def test_internal_dashes(self):
        assert contains_sentinel("<not-for-me/>")

    def test_padded_with_prose(self):
        assert contains_sentinel(
            "Sure, I think this wasn't for me. <not_for_me/>"
        )

    def test_inside_xml_block(self):
        assert contains_sentinel("<message><not_for_me/></message>")

    def test_with_surrounding_whitespace(self):
        assert contains_sentinel("  <not_for_me/>  ")

    def test_plain_phrase_does_not_trigger(self):
        # The phrase "not for me" in prose must not falsely trigger —
        # users might say "that's not for me" in a real reply.
        assert not contains_sentinel(
            "Sorry, that show isn't really for me."
        )

    def test_empty_returns_false(self):
        assert not contains_sentinel("")
        assert not contains_sentinel(None)

    def test_normal_response_returns_false(self):
        assert not contains_sentinel(
            "It's currently seventy-two degrees and sunny."
        )


class TestStripSentinel:
    def test_strip_canonical(self):
        assert strip_sentinel("<not_for_me/>") == ""

    def test_strip_with_prose(self):
        assert strip_sentinel("hello <not_for_me/> world") == "hello  world".strip()

    def test_strip_does_not_touch_unrelated_text(self):
        assert strip_sentinel("hello world") == "hello world"

    def test_strip_empty(self):
        assert strip_sentinel("") == ""
        assert strip_sentinel(None) == ""


class TestSentinelLiteral:
    def test_literal_matches_pattern(self):
        # If we ever document the sentinel format somewhere, this protects
        # against a typo in the literal drifting from what the regex
        # accepts.
        assert contains_sentinel(SENTINEL_LITERAL)


class TestNotForMeInstruction:
    """The prompt instruction must mention the sentinel + the safety bar."""

    def test_mentions_sentinel(self):
        assert "<not_for_me/>" in NOT_FOR_ME_INSTRUCTION

    def test_includes_safety_bar(self):
        # Conservative bar: false-suppressing a real command must be
        # explicitly framed as worse than answering one stray utterance.
        assert "doubt" in NOT_FOR_ME_INSTRUCTION.lower()
        assert "respond normally" in NOT_FOR_ME_INSTRUCTION.lower()

    def test_mentions_at_least_one_pattern(self):
        # The instruction should give the LLM concrete patterns to look for.
        body = NOT_FOR_ME_INSTRUCTION.lower()
        assert any(
            cue in body
            for cue in ("third-person", "narrative", "mid-sentence", "overheard")
        )

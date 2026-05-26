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
        body = NOT_FOR_ME_INSTRUCTION.lower()
        # The bar can be phrased as "default to answering", "silencing a
        # real request is worse", "when in doubt", etc. — any of these is
        # an explicit framing that pushes the model toward responding.
        assert any(
            phrase in body
            for phrase in (
                "doubt",
                "silencing a real",
                "default to answering",
                "default to responding",
                "worse bug",
            )
        )

    def test_mentions_overheard_cue(self):
        # The instruction must name overheard speech as the silence trigger
        # — that's the entire reason this guard exists.
        assert "overheard" in NOT_FOR_ME_INSTRUCTION.lower()

    def test_instruction_is_loose_no_pattern_examples(self):
        # Regression: prior versions listed silence-case examples like
        # "yeah but then she said" / "uh huh". The 14B was pattern-matching
        # memory questions onto those examples and silencing. The loose
        # version drops the example list entirely so the model has no
        # silence patterns to anchor on — must reason from first principles.
        body = NOT_FOR_ME_INSTRUCTION.lower()
        # The memory-relevance directive moved out of this instruction and
        # into the User Profile header, so this body should NOT carry the
        # old example list any more.
        assert "yeah but then she said" not in body
        assert "uh huh" not in body
        assert "hey sarah" not in body
        # And the must-respond example list also lives in the User Profile
        # header now, not here.
        assert "who's my favorite" not in body


class TestProvidersDoNotInjectInstruction:
    """The instruction lives in conversation_handler._get_system_prompt now;
    individual providers must not include it. If a provider re-introduces
    the import it'll double-inject — these guards catch that at test time.
    """

    def test_no_qwen3_provider_imports_instruction(self):
        # Re-importing into a provider is the entry point for double-injection.
        # Read the source files directly — importing modules and inspecting
        # symbols would miss conditional imports.
        from pathlib import Path

        provider_root = (
            Path(__file__).resolve().parent.parent
            / "app" / "core" / "prompt_providers"
        )
        offenders: list[str] = []
        for path in provider_root.rglob("*.py"):
            text = path.read_text()
            if "NOT_FOR_ME_INSTRUCTION" in text and path.name != "core_rules.py":
                offenders.append(str(path.relative_to(provider_root)))
        assert not offenders, (
            f"NOT_FOR_ME_INSTRUCTION must only live in core_rules.py + "
            f"conversation_handler. Found in: {offenders}"
        )

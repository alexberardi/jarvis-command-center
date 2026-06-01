"""Tests for clean_for_tts — pre-TTS text sanitation.

Covers emoji stripping (legacy behavior) plus markdown stripping (added
because TTS engines were literally pronouncing asterisks/backticks aloud
when the LLM produced formatted prose).
"""

from app.core.tts_text import clean_for_tts


class TestPassthrough:
    """Plain prose must round-trip unchanged."""

    def test_returns_empty_string_unchanged(self):
        assert clean_for_tts("") == ""

    def test_returns_none_safely(self):
        assert clean_for_tts(None) is None  # type: ignore[arg-type]

    def test_plain_sentence_unchanged(self):
        text = "Hello world, how are you today?"
        assert clean_for_tts(text) == text

    def test_punctuation_preserved(self):
        text = "Sure! Here's the answer: it's 42 (probably)."
        assert clean_for_tts(text) == text

    def test_math_glyphs_preserved(self):
        # 5*3 has word chars on both sides of *; we leave it alone.
        assert clean_for_tts("5*3 equals 15") == "5*3 equals 15"

    def test_identifier_underscores_preserved(self):
        assert clean_for_tts("Set user_id to 42") == "Set user_id to 42"


class TestEmojiStripping:
    """Existing emoji behavior must not regress."""

    def test_strips_smiley(self):
        assert clean_for_tts("you 😄 there") == "you there"

    def test_strips_compound_emoji(self):
        assert clean_for_tts("ready 👨‍👩‍👧 set") == "ready set"

    def test_currency_glyphs_preserved(self):
        # $, €, £ are not in the targeted emoji ranges.
        assert clean_for_tts("That'll be $5.") == "That'll be $5."


class TestMarkdownEmphasis:
    """Asterisks and underscores wrapping text get unwrapped."""

    def test_strips_bold(self):
        assert clean_for_tts("**hello** world") == "hello world"

    def test_strips_italic(self):
        assert clean_for_tts("*hello* world") == "hello world"

    def test_strips_bold_italic(self):
        assert clean_for_tts("***hello*** world") == "hello world"

    def test_strips_underscore_bold(self):
        assert clean_for_tts("__hello__ world") == "hello world"

    def test_strips_underscore_italic_at_word_boundary(self):
        assert clean_for_tts("say _hello_ now") == "say hello now"

    def test_strips_strikethrough(self):
        assert clean_for_tts("~~nope~~ yes") == "nope yes"

    def test_multiple_emphases_in_one_sentence(self):
        assert (
            clean_for_tts("It is **really** *very* important.")
            == "It is really very important."
        )


class TestMarkdownCode:
    """Inline and fenced code markers get unwrapped."""

    def test_strips_inline_code(self):
        assert clean_for_tts("run `npm test` first") == "run npm test first"

    def test_strips_fenced_code_block(self):
        text = "Try this:\n```python\nprint('hi')\n```\nDone."
        # Fenced code keeps the inner content; we just want the markers gone.
        result = clean_for_tts(text)
        assert "```" not in result
        assert "print('hi')" in result
        assert "Done." in result


class TestMarkdownStructure:
    """Headings, lists, blockquotes, links, rules."""

    def test_strips_headings(self):
        assert clean_for_tts("# Title\nbody") == "Title\nbody"

    def test_strips_subheading(self):
        assert clean_for_tts("### Sub\nbody") == "Sub\nbody"

    def test_strips_bullet_dash(self):
        assert clean_for_tts("- one\n- two") == "one\ntwo"

    def test_strips_bullet_asterisk(self):
        assert clean_for_tts("* one\n* two") == "one\ntwo"

    def test_strips_numbered_list(self):
        assert clean_for_tts("1. first\n2. second") == "first\nsecond"

    def test_strips_blockquote(self):
        assert clean_for_tts("> quoted line") == "quoted line"

    def test_strips_horizontal_rule(self):
        assert "---" not in clean_for_tts("above\n---\nbelow")

    def test_unwraps_link_keeps_text(self):
        assert (
            clean_for_tts("Visit [our site](https://example.com) today")
            == "Visit our site today"
        )

    def test_drops_image_entirely(self):
        # Alt text is rarely worth saying aloud.
        result = clean_for_tts("Look ![cat picture](https://x/c.png) here")
        assert "cat picture" not in result
        assert "Look" in result and "here" in result


class TestOrphans:
    """Unbalanced or stray formatting glyphs the LLM emits."""

    def test_strips_leading_orphan_asterisks(self):
        assert clean_for_tts("**unclosed bold here") == "unclosed bold here"

    def test_strips_trailing_orphan_asterisks(self):
        assert clean_for_tts("trailing asterisks**") == "trailing asterisks"

    def test_strips_mid_sentence_orphan(self):
        assert clean_for_tts("He said * to me") == "He said to me"


class TestRealWorldLLMOutput:
    """End-to-end-ish samples of the kind the LLM actually produces."""

    def test_typical_response(self):
        text = (
            "Here's what I found:\n"
            "- **Temperature**: 72°F\n"
            "- *Humidity*: 45%\n"
            "Need anything else?"
        )
        result = clean_for_tts(text)
        assert "**" not in result
        assert "*" not in result
        assert "Temperature" in result
        assert "Humidity" in result
        assert "Need anything else?" in result

    def test_code_explanation(self):
        text = "Call `setUser()` with the **id** parameter."
        assert clean_for_tts(text) == "Call setUser() with the id parameter."

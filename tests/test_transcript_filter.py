"""Tests for the pre-LLM STT-noise filter.

These tests lock in the 2026-06-02 prod regression: ``*sniff*`` reached
the LLM, which hallucinated ``"I smell something burning."`` ``is_stt_noise``
is the server-side gate that prevents bracketed action notation, empty
strings, and pure punctuation from ever reaching inference.
"""

from app.core.transcript_filter import is_stt_noise


class TestIsSttNoiseEmptyOrWhitespace:
    def test_none_is_noise(self):
        assert is_stt_noise(None)

    def test_empty_string_is_noise(self):
        assert is_stt_noise("")

    def test_whitespace_only_is_noise(self):
        assert is_stt_noise("   ")
        assert is_stt_noise("\n\t  ")


class TestIsSttNoiseBracketedAnnotation:
    """Whisper-style annotations the LLM should never see."""

    def test_asterisk_sniff(self):
        # The exact 2026-06-02 prod regression input.
        assert is_stt_noise("*sniff*")

    def test_asterisk_sad_noises_multiword(self):
        # Internal whitespace inside the asterisks.
        assert is_stt_noise("*sad noises*")

    def test_asterisk_with_trailing_period(self):
        # Whisper sometimes appends a period after the closing asterisk.
        assert is_stt_noise("*sniff*.")

    def test_square_blank_audio(self):
        assert is_stt_noise("[BLANK_AUDIO]")

    def test_square_music(self):
        assert is_stt_noise("[music]")

    def test_paren_silence(self):
        assert is_stt_noise("(silence)")

    def test_paren_wind_blowing(self):
        assert is_stt_noise("(wind blowing)")

    def test_angle_inaudible(self):
        # Less common form Whisper-derived tools sometimes emit.
        assert is_stt_noise("<inaudible>")

    def test_consecutive_bracketed_tokens(self):
        # Two annotations back-to-back are also noise.
        assert is_stt_noise("*sniff* *cough*")
        assert is_stt_noise("[laughter] (sigh)")


class TestIsSttNoisePurePunctuation:
    def test_ellipsis(self):
        assert is_stt_noise("...")
        assert is_stt_noise("…")

    def test_dashes(self):
        assert is_stt_noise("--")
        assert is_stt_noise("---")

    def test_repeated_question_marks(self):
        assert is_stt_noise("???")


class TestIsSttNoiseLegitimateUtterances:
    """Real commands must NEVER be flagged as noise."""

    def test_short_imperative(self):
        assert not is_stt_noise("Stop.")
        assert not is_stt_noise("Pause")

    def test_question(self):
        assert not is_stt_noise("What's the weather?")

    def test_filler_word(self):
        # Critically: single-word fillers ("okay", "yeah") are NOT noise —
        # they're valid follow-ups. Only bracketed/empty inputs should be
        # flagged.
        assert not is_stt_noise("okay")
        assert not is_stt_noise("yeah")
        assert not is_stt_noise("yes")
        assert not is_stt_noise("bye")

    def test_brackets_mid_sentence_not_noise(self):
        # A real utterance that happens to contain brackets mid-string is
        # not bracketed-only and must not be filtered.
        assert not is_stt_noise("open the *kitchen* light")
        assert not is_stt_noise("turn on [the] lamp")

    def test_command_starting_with_punctuation(self):
        # An accidental leading punctuation char shouldn't sink the rest.
        assert not is_stt_noise("?what time is it")

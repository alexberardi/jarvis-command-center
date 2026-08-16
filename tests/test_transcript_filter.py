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


class TestIsDeviceCommandShaped:
    """The imperative device-command guard (2026-08-15): a transcript shaped
    like "turn on the ... lights" must be recognizable so acoustic-side
    evidence alone can never produce a suppression-leaning hint for it."""

    def test_the_two_real_suppressed_commands(self):
        # The literal 2026-08-15 losses — suppressed by wake verification
        # reading garbage clips. These MUST match.
        from app.core.transcript_filter import is_device_command_shaped

        assert is_device_command_shaped("Turn on the living room lights.")
        assert is_device_command_shaped("Turn on the playroom lights.")

    def test_common_device_imperatives_match(self):
        from app.core.transcript_filter import is_device_command_shaped

        assert is_device_command_shaped("turn off the kitchen lights")
        assert is_device_command_shaped("switch off the fan")
        assert is_device_command_shaped("dim the bedroom lights")
        assert is_device_command_shaped("lock the front door")
        assert is_device_command_shaped("unlock the back door")
        assert is_device_command_shaped("set the thermostat to 70")
        assert is_device_command_shaped("start the coffee maker")
        assert is_device_command_shaped("stop the music")

    def test_politeness_prefix_matches(self):
        from app.core.transcript_filter import is_device_command_shaped

        assert is_device_command_shaped("please turn on the lights")
        assert is_device_command_shaped("Jarvis, turn on the lights")

    def test_junk_counter_examples_do_not_match(self):
        # The real prod junk that SHOULD keep its ambient-leaning hints.
        from app.core.transcript_filter import is_device_command_shaped

        assert not is_device_command_shaped("- Uh-huh. - Eat it.")
        assert not is_device_command_shaped(
            "Where's your nose? Oh, you want to do this?"
        )

    def test_questions_do_not_match(self):
        from app.core.transcript_filter import is_device_command_shaped

        assert not is_device_command_shaped("Can you turn on the lights?")
        assert not is_device_command_shaped("Did you turn off the oven?")

    def test_generic_speech_does_not_match(self):
        from app.core.transcript_filter import is_device_command_shaped

        assert not is_device_command_shaped("Leo took his medicine.")
        assert not is_device_command_shaped("Eat it.")
        assert not is_device_command_shaped("We got paid last year in August")
        assert not is_device_command_shaped("")
        assert not is_device_command_shaped(None)

    def test_bare_verb_without_object_does_not_match(self):
        # "Stop." alone is a follow-up/fragment, not a device command shape.
        from app.core.transcript_filter import is_device_command_shaped

        assert not is_device_command_shaped("Stop.")

    def test_multi_speaker_device_command_does_not_match(self):
        # Dash-marked dialogue containing an imperative is still dialogue.
        from app.core.transcript_filter import is_device_command_shaped

        assert not is_device_command_shaped("- Turn on the lights. - No, don't.")


class TestHasMultiSpeakerMarkers:
    def test_dialogue_matches(self):
        from app.core.transcript_filter import has_multi_speaker_markers

        assert has_multi_speaker_markers("- Uh-huh. - Eat it.")
        assert has_multi_speaker_markers("- Yeah. - Eat it.")

    def test_single_leading_dash_is_not_dialogue(self):
        # Whisper sometimes prefixes a lone utterance with one dash.
        from app.core.transcript_filter import has_multi_speaker_markers

        assert not has_multi_speaker_markers("- Turn on the lights")

    def test_hyphenated_words_do_not_match(self):
        from app.core.transcript_filter import has_multi_speaker_markers

        assert not has_multi_speaker_markers("Uh-huh, that's a mother-in-law")

    def test_plain_speech_does_not_match(self):
        from app.core.transcript_filter import has_multi_speaker_markers

        assert not has_multi_speaker_markers("turn on the lights")
        assert not has_multi_speaker_markers("")
        assert not has_multi_speaker_markers(None)


class TestIsShortNonCommandFragment:
    def test_two_word_fragments_match(self):
        from app.core.transcript_filter import is_short_non_command_fragment

        assert is_short_non_command_fragment("Eat it.")
        assert is_short_non_command_fragment("Oh.")

    def test_device_shaped_two_worders_are_excluded(self):
        # The imperative guard is senior to the junk-shape signals.
        from app.core.transcript_filter import is_short_non_command_fragment

        assert not is_short_non_command_fragment("stop music")

    def test_three_plus_words_are_not_fragments(self):
        from app.core.transcript_filter import is_short_non_command_fragment

        assert not is_short_non_command_fragment("what time is it")

    def test_empty_is_not_a_fragment(self):
        # Empty input is the STT-noise filter's job, not the fragment hint's.
        from app.core.transcript_filter import is_short_non_command_fragment

        assert not is_short_non_command_fragment("")
        assert not is_short_non_command_fragment(None)


class TestIsMusicControlShaped:
    """Music-control shape detector for SELF-PLAYBACK turns. Unlike the
    device-command guard, bare verbs count ("pause", "skip") — the playing
    music supplies the object. Only ever consulted when the node reported
    self-playback, and only ever PREVENTS a suppression-leaning hint."""

    def test_bare_verbs_match(self):
        from app.core.transcript_filter import is_music_control_shaped

        for cmd in ("Pause.", "Stop", "Skip.", "Next", "Resume.", "Mute"):
            assert is_music_control_shaped(cmd), cmd

    def test_verb_object_forms_match(self):
        from app.core.transcript_filter import is_music_control_shaped

        for cmd in (
            "Stop the music.",
            "Skip this song.",
            "Next track.",
            "Play some jazz.",
            "Play the next one.",
            "Turn it down.",
            "Turn down the volume.",
            "Turn off the music.",
            "Volume up.",
            "Louder.",
            "Quieter.",
        ):
            assert is_music_control_shaped(cmd), cmd

    def test_politeness_and_wake_prefixes_match(self):
        from app.core.transcript_filter import is_music_control_shaped

        assert is_music_control_shaped("Jarvis, skip this song.")
        assert is_music_control_shaped("Please pause.")
        assert is_music_control_shaped("Hey Jarvis, turn it down.")

    def test_questions_do_not_match(self):
        from app.core.transcript_filter import is_music_control_shaped

        assert not is_music_control_shaped("Can you skip this song?")
        assert not is_music_control_shaped("Skip this one?")

    def test_multi_speaker_dialogue_does_not_match(self):
        from app.core.transcript_filter import is_music_control_shaped

        assert not is_music_control_shaped("- Pause. - Eat it.")

    def test_generic_speech_does_not_match(self):
        from app.core.transcript_filter import is_music_control_shaped

        for text in (
            "I love this song.",
            "We stopped by the store earlier.",
            "He was going to play outside.",
            "Eat it.",
            "Oh.",
        ):
            assert not is_music_control_shaped(text), text

    def test_empty_is_not_music_control(self):
        from app.core.transcript_filter import is_music_control_shaped

        assert not is_music_control_shaped("")
        assert not is_music_control_shaped(None)

    def test_mid_word_verb_prefix_does_not_match(self):
        # "stopped"/"playing" must not read as "stop"/"play" (word boundary).
        from app.core.transcript_filter import is_music_control_shaped

        assert not is_music_control_shaped("Stopping by later.")
        assert not is_music_control_shaped("Playtime is over soon.")

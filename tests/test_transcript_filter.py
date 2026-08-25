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


class TestAddressedHouseholdMember:
    """Named-person addressing shapes (2026-08-17 incident: a follow-up
    captured "already done. Wow. Miles, come here." — a parent calling
    their child by name). Form-only detection, feeding a lean hint; the
    device/music guards stay senior at every call site."""

    MEMBERS = ["Miles", "Jess", "Alex"]

    def test_incident_transcript_matches(self):
        from app.core.transcript_filter import addressed_household_member

        # Vocative in the THIRD sentence — matching must be per-sentence.
        assert addressed_household_member(
            "already done. Wow. Miles, come here.", self.MEMBERS
        ) == "Miles"

    def test_leading_name_with_comma_matches(self):
        from app.core.transcript_filter import addressed_household_member

        assert addressed_household_member(
            "Jess, can you grab that", self.MEMBERS
        ) == "Jess"

    def test_leading_name_into_imperative_matches(self):
        # STT frequently drops the vocative comma.
        from app.core.transcript_filter import addressed_household_member

        assert addressed_household_member(
            "Miles come here", self.MEMBERS
        ) == "Miles"

    def test_trailing_vocative_matches(self):
        from app.core.transcript_filter import addressed_household_member

        assert addressed_household_member(
            "come here, Miles", self.MEMBERS
        ) == "Miles"
        assert addressed_household_member(
            "come here Miles", self.MEMBERS
        ) == "Miles"

    def test_case_insensitive(self):
        from app.core.transcript_filter import addressed_household_member

        assert addressed_household_member(
            "miles, come here", self.MEMBERS
        ) == "Miles"

    def test_speech_about_a_member_does_not_match(self):
        from app.core.transcript_filter import is_addressed_to_other_person

        for text in (
            "Miles said he wants pizza",
            "I told Miles to clean his room already",
            "Miles is a little busy with his toys",
            "we should take Miles to the park tomorrow",
        ):
            assert not is_addressed_to_other_person(text, self.MEMBERS), text

    def test_word_boundary_no_substring_match(self):
        from app.core.transcript_filter import is_addressed_to_other_person

        # "Milestone, come quick" must not read as addressing Miles.
        assert not is_addressed_to_other_person(
            "Milestone, come quick", self.MEMBERS
        )

    def test_jarvis_and_wake_phrase_excluded(self):
        from app.core.transcript_filter import is_addressed_to_other_person

        assert not is_addressed_to_other_person(
            "Jarvis, what time is it", ["Jarvis", "Miles"]
        )
        assert not is_addressed_to_other_person(
            "hey, come look at this", ["Hey", "Miles"]
        )

    def test_no_members_is_a_no_op(self):
        from app.core.transcript_filter import (
            addressed_household_member,
            is_addressed_to_other_person,
        )

        text = "Miles, come here"
        assert addressed_household_member(text, None) is None
        assert addressed_household_member(text, []) is None
        assert not is_addressed_to_other_person(text, None)

    def test_empty_transcript_is_a_no_op(self):
        from app.core.transcript_filter import addressed_household_member

        assert addressed_household_member(None, self.MEMBERS) is None
        assert addressed_household_member("", self.MEMBERS) is None

    def test_multi_word_display_name_uses_first_token(self):
        from app.core.transcript_filter import addressed_household_member

        assert addressed_household_member(
            "Jess, come here", ["Jess Berardi"]
        ) == "Jess"

    def test_non_string_member_entries_are_ignored(self):
        from app.core.transcript_filter import addressed_household_member

        assert addressed_household_member(
            "Miles, come here", [None, 42, "Miles"]  # type: ignore[list-item]
        ) == "Miles"


class TestIsQuestionShaped:
    """Question shapes are exempt from every tool-forcing guard (2026-08-17
    doctrine: the model owns how to answer questions)."""

    def test_incident_utterance_is_question(self):
        from app.core.transcript_filter import is_question_shaped
        assert is_question_shaped("What should I do with Miles today?")

    def test_leading_question_word_without_question_mark(self):
        from app.core.transcript_filter import is_question_shaped
        assert is_question_shaped("what does my schedule look like today")
        assert is_question_shaped("should I bring an umbrella")
        assert is_question_shaped("can you turn on the lights")
        assert is_question_shaped("is there anything on my calendar")
        assert is_question_shaped("do we have milk")

    def test_trailing_question_mark_alone_qualifies(self):
        from app.core.transcript_filter import is_question_shaped
        assert is_question_shaped("turn off the lights?")

    def test_wake_prefix_is_skipped(self):
        from app.core.transcript_filter import is_question_shaped
        assert is_question_shaped("Jarvis, what is on my calendar")
        assert is_question_shaped("hey jarvis, when is my meeting")

    def test_imperatives_and_reports_are_not_questions(self):
        from app.core.transcript_filter import is_question_shaped
        assert not is_question_shaped("Turn off the living room lights")
        assert not is_question_shaped("Leo took his medicine")
        assert not is_question_shaped("Set a timer for ten minutes")

    def test_empty_and_none(self):
        from app.core.transcript_filter import is_question_shaped
        assert not is_question_shaped(None)
        assert not is_question_shaped("")
        assert not is_question_shaped("   ")


class TestIsActionCommandShaped:
    """Generalized imperative shape — the only utterance family (with
    reports) allowed to arm the force-tool-calls guard."""

    def test_device_imperatives_still_match(self):
        from app.core.transcript_filter import is_action_command_shaped
        assert is_action_command_shaped("Turn off the living room lights")
        assert is_action_command_shaped("set the thermostat to 68")
        assert is_action_command_shaped("lock the front door")

    def test_extended_action_verbs_match(self):
        from app.core.transcript_filter import is_action_command_shaped
        assert is_action_command_shaped("Play some jazz")
        assert is_action_command_shaped("Remind me to call mom")
        assert is_action_command_shaped("Add milk to the shopping list")
        assert is_action_command_shaped("Cancel my three o'clock")
        assert is_action_command_shaped("Log that Leo took his medicine")

    def test_questions_never_match(self):
        from app.core.transcript_filter import is_action_command_shaped
        assert not is_action_command_shaped("What should I do with Miles today?")
        assert not is_action_command_shaped("can you turn on the lights")
        assert not is_action_command_shaped("turn off the lights?")

    def test_conversational_prose_does_not_match(self):
        from app.core.transcript_filter import is_action_command_shaped
        assert not is_action_command_shaped("Thank you.")
        assert not is_action_command_shaped("Yum.")
        assert not is_action_command_shaped("the lights in here are so warm")

    def test_multi_speaker_transcripts_do_not_match(self):
        from app.core.transcript_filter import is_action_command_shaped
        assert not is_action_command_shaped("- Uh-huh. - Turn it off.")

    def test_empty_and_none(self):
        from app.core.transcript_filter import is_action_command_shaped
        assert not is_action_command_shaped(None)
        assert not is_action_command_shaped("")


class TestIsReportShaped:
    """Reports feeding logging tools ("Leo took his medicine") — the second
    utterance family where a forced tool call is legitimate."""

    def test_third_person_medication_report(self):
        from app.core.transcript_filter import is_report_shaped
        assert is_report_shaped("Leo took his medicine")
        assert is_report_shaped("Leo took his medicine.")

    def test_first_person_medication_report(self):
        from app.core.transcript_filter import is_report_shaped
        assert is_report_shaped("I took my pills")
        assert is_report_shaped("I just took my morning meds")

    def test_gave_and_administered_forms(self):
        from app.core.transcript_filter import is_report_shaped
        assert is_report_shaped("She gave the dog its meds")
        assert is_report_shaped("Sam administered the insulin")

    def test_questions_never_match(self):
        from app.core.transcript_filter import is_report_shaped
        assert not is_report_shaped("did Leo take his medicine?")
        assert not is_report_shaped("did I take my pills")

    def test_conversational_prose_does_not_match(self):
        from app.core.transcript_filter import is_report_shaped
        assert not is_report_shaped("Thank you.")
        assert not is_report_shaped("that was so good")

    def test_multi_speaker_transcripts_do_not_match(self):
        from app.core.transcript_filter import is_report_shaped
        assert not is_report_shaped("- Leo took his medicine. - Good boy.")

    def test_empty_and_none(self):
        from app.core.transcript_filter import is_report_shaped
        assert not is_report_shaped(None)
        assert not is_report_shaped("")


class TestResponseClaimsAction:
    """``response_claims_action`` classifies the MODEL's own reply, not the
    user's transcript.

    2026-08-25 prod incident (kitchen, 7:03 AM): STT truncated "Leo took his
    medicine" down to "his medicine." — the verb was lost with the head of the
    utterance. The force-tool-calls guard classifies by UTTERANCE shape, so a
    bare noun phrase read as "neither action- nor report-shaped" and the guard
    stood down. The model answered "I'll check on Leo's meds for you." with
    ``tool_calls: []`` and nothing ran; the dose went unlogged. The same model
    failure on the typed path ("Leo took his medicine" — report-shaped) WAS
    caught and retried into ``medication(action=mark)``.

    The model's own words are the signal the truncated transcript lost: a reply
    that PROMISES or CLAIMS an action while calling no tool is a correctness
    failure regardless of how the transcript reads.
    """

    def test_promise_to_act_is_a_claim(self):
        from app.core.transcript_filter import response_claims_action
        # The exact prod line that went unlogged.
        assert response_claims_action("I'll check on Leo's meds for you.")
        assert response_claims_action("I will turn off the lights.")
        assert response_claims_action("Sure, I'll add that to your list.")
        assert response_claims_action("Let me look that up for you.")
        assert response_claims_action("I'm going to set a reminder.")

    def test_claim_of_completed_action(self):
        from app.core.transcript_filter import response_claims_action
        # The typed-path line the guard already catches — same shape family.
        assert response_claims_action("Got it, Leo's medicine is marked as taken.")
        assert response_claims_action("I've marked it as taken.")
        assert response_claims_action("Done — I added that to your list.")
        assert response_claims_action("I just logged that for you.")
        assert response_claims_action("That's been scheduled.")

    def test_in_progress_claim(self):
        from app.core.transcript_filter import response_claims_action
        assert response_claims_action("I'm marking that now.")
        assert response_claims_action("Setting a reminder for 8 AM.")

    def test_plain_answers_are_not_claims(self):
        from app.core.transcript_filter import response_claims_action
        # 2026-08-17 calendar incident: the model ANSWERED an open question.
        # Widening the guard must not drag these back into a forced retry.
        assert not response_claims_action(
            "You've got soccer at four and dinner with the Harrisons at six."
        )
        assert not response_claims_action("It's seventy-two degrees and sunny.")
        assert not response_claims_action("Hey there. What's on your mind?")
        assert not response_claims_action("Sounds like someone's been busy.")

    def test_non_action_first_person_is_not_a_claim(self):
        from app.core.transcript_filter import response_claims_action
        # "Let me know" / "I'll keep" are conversational, not tool work — the
        # verb list is the wall.
        assert not response_claims_action("Let me know if you need anything else.")
        assert not response_claims_action("I'll keep that in mind.")
        assert not response_claims_action("I'm not sure what you mean.")
        assert not response_claims_action("I don't know that one.")
        assert not response_claims_action("I'm a bit confused — what did you say?")

    def test_empty_and_none(self):
        from app.core.transcript_filter import response_claims_action
        assert not response_claims_action(None)
        assert not response_claims_action("")
        assert not response_claims_action("   ")


class TestReportShapeDitransitive:
    """`<subject> gave <indirect object> <possessive> <thing>` is report shape.

    2026-08-25 evening, prod: "I gave Leo his medicine." — clean transcript, no
    truncation — logged "Force-tool-calls guard skipped — utterance is neither
    action- nor report-shaped" and the dose went unlogged, twice on voice and
    twice more in the app.

    _REPORT_SHAPE_RE only accepted `<subject> gave <possessive>`, so naming the
    recipient broke it. Every other phrasing of the same sentence passed. The
    most natural way to say it was the one way that failed.
    """

    def test_named_recipient_is_report_shaped(self):
        from app.core.transcript_filter import is_report_shaped
        # The exact prod utterance.
        assert is_report_shaped("I gave Leo his medicine.")
        assert is_report_shaped("I gave Leo his meds")
        assert is_report_shaped("Kait gave Groot his pill")
        assert is_report_shaped("I gave the dog his medicine")
        assert is_report_shaped("I gave the kids their vitamins")

    def test_previously_working_shapes_still_work(self):
        from app.core.transcript_filter import is_report_shaped
        # Regression net: these all passed before the ditransitive change and
        # must keep passing (the optional indirect object has to backtrack out).
        assert is_report_shaped("Leo took his medicine")
        assert is_report_shaped("I took my pills")
        assert is_report_shaped("I gave him his medicine")
        assert is_report_shaped("she gave the dog its meds")
        assert is_report_shaped("Leo got his pill")
        assert is_report_shaped("Leo already took his medicine")

    def test_questions_and_multi_speaker_still_excluded(self):
        from app.core.transcript_filter import is_report_shaped
        # The senior guards are unchanged: a question is never a report, and
        # dash-marked cross-talk is never a single-speaker report.
        assert not is_report_shaped("Did I give Leo his medicine?")
        assert not is_report_shaped("Should I give Leo his medicine")
        assert not is_report_shaped("- I gave Leo his medicine. - Did you?")

    def test_non_reports_still_rejected(self):
        from app.core.transcript_filter import is_report_shaped
        assert not is_report_shaped("what's the weather")
        assert not is_report_shaped("turn off the lights")
        assert not is_report_shaped("his medicine.")

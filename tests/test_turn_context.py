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

    def test_confident_wake_action_report_still_calls_tool(self):
        # Load-bearing carve-out: the "answer directly from profile" guidance
        # must NOT suppress the tool on an action report whose subject is a
        # profile fact ("I gave Leo his medicine" — profile has the Keppra
        # routine). Verbatim-prompt A/B vs the prod 9B: 31% -> 98% tool calls.
        hint = build_turn_hint("wake", wake_confidence=0.9)
        assert "REPORTS or REQUESTS an action" in hint
        assert "call the tool that does it" in hint

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

    def test_follow_up_offers_the_tool_affordance(self):
        # Regression pin for the follow-up tool-call drop: a continuation that
        # needs a tool must be allowed to RUN it, not just answered as prose.
        # The wake hint always said "run the tool it calls for"; the follow-up
        # hint had dropped it, steering the model to prose/silence.
        hint = build_turn_hint("follow_up", follow_up_iteration=1)
        assert "run the tool" in hint.lower()


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


class TestShouldDoubleCheckSentinel:
    """Gate for the engine's sentinel double-check. Turns whose acoustics
    clearly say "directed at Jarvis" qualify (a quiet-room wake or a
    high-confidence OWW fire), as do typed chat and EARLY follow-up iterations
    (a real continuation may need a tool). Ambient/middle-band turns and LATE
    follow-up iterations don't — genuine overheard-speech and end-of-window
    silences stay single-pass."""

    def test_quiet_room_wake_qualifies(self):
        from app.core.turn_context import should_double_check_sentinel

        # The 2026-07-27 "who is Leo?" failure: pre_wake=0.00, snap sentinel.
        assert should_double_check_sentinel(None, None, None, 0.0) is True

    def test_confident_explicit_wake_qualifies(self):
        from app.core.turn_context import should_double_check_sentinel

        assert should_double_check_sentinel("wake", 0.95, None, None) is True

    def test_middle_band_does_not_qualify(self):
        from app.core.turn_context import should_double_check_sentinel

        # 2.24s pre-wake speech (the correctly-rejected "We got paid last
        # year in August") — ambiguous acoustics, no second pass.
        assert should_double_check_sentinel(None, None, None, 2.24) is False

    def test_low_confidence_wake_does_not_qualify(self):
        from app.core.turn_context import should_double_check_sentinel

        assert should_double_check_sentinel("wake", 0.4, None, None) is False

    def test_early_follow_up_qualifies_for_rescue(self):
        from app.core.turn_context import should_double_check_sentinel

        # Regression pin for the follow-up tool-call drop: a tool-needing
        # continuation ("...actually set a timer for 5 minutes") must not be
        # snap-silenced — early follow-up iterations buy the /think re-check.
        assert should_double_check_sentinel("follow_up", None, 1, 0.0) is True
        assert (
            should_double_check_sentinel(
                "follow_up", None, FOLLOW_UP_STRICT_ITERATION - 1, None
            )
            is True
        )

    def test_follow_up_default_iteration_qualifies(self):
        from app.core.turn_context import should_double_check_sentinel

        # No iteration supplied → treated as iteration 1 → qualifies.
        assert should_double_check_sentinel("follow_up", None, None, None) is True

    def test_late_follow_up_stays_single_pass(self):
        from app.core.turn_context import should_double_check_sentinel

        # By the strict iteration the room has likely moved on — silence is
        # the window's designed ending, so don't argue the model out of it.
        assert (
            should_double_check_sentinel(
                "follow_up", None, FOLLOW_UP_STRICT_ITERATION, None
            )
            is False
        )

    def test_no_signal_does_not_qualify(self):
        from app.core.turn_context import should_double_check_sentinel

        assert should_double_check_sentinel(None, None, None, None) is False


class TestChatSource:
    """Typed app messages: no microphone in the loop, so "overheard" is
    impossible by construction. 2026-07-27: "Who is Leo" typed in the
    mobile app was snap-sentineled — mobile chat sent no provenance at
    all, so no hint fired and no double-check gate opened."""

    def test_chat_hint_says_typed_and_kills_not_for_me(self):
        hint = build_turn_hint("chat")
        assert hint is not None
        assert hint.startswith("[turn context:")
        assert "typed" in hint.lower()
        assert "<not_for_me/>" in hint

    def test_chat_always_qualifies_for_double_check(self):
        from app.core.turn_context import should_double_check_sentinel

        assert should_double_check_sentinel("chat") is True

    def test_chat_makes_no_acoustic_claims(self):
        hint = build_turn_hint("chat")
        assert "wake" not in hint.lower()
        assert "confidence" not in hint.lower()


class TestWakeVerifiedSoftBias:
    """2026-08-15 regression pins: verified=False demoted from a hard
    suppression instruction to a mild ambient lean. Two REAL lights commands
    were suppressed because wake verification read garbage clips — one bad
    clip must never single-handedly silence a turn."""

    def test_unverified_hint_is_a_mild_lean_not_an_instruction(self):
        hint = build_turn_hint("wake", wake_confidence=0.95, wake_verified=False)
        assert hint is not None
        assert "one signal" in hint
        # The old hard posture must be gone.
        assert "almost certainly" not in hint
        assert "Do NOT act" not in hint

    def test_unverified_hint_keeps_the_answer_path_open(self):
        hint = build_turn_hint("wake", wake_confidence=0.95, wake_verified=False)
        assert "should still be answered" in hint

    def test_unverified_hint_does_not_require_addressing_by_name(self):
        # The old text demanded the transcript "explicitly addresses you by
        # name" — nobody says "Jarvis" mid-command after the wake word.
        hint = build_turn_hint("wake", wake_confidence=0.95, wake_verified=False)
        assert "by name" not in hint

    def test_device_command_overrides_unverified_lean(self):
        # Imperative guard senior to the verdict: the literal 2026-08-15
        # losses keep the normal directed posture.
        for cmd in (
            "Turn on the living room lights.",
            "Turn on the playroom lights.",
        ):
            hint = build_turn_hint(
                "wake",
                wake_confidence=0.97,
                wake_verified=False,
                transcript=cmd,
            )
            assert hint is not None
            assert "addressed to you" in hint.lower()
            assert "misfire" not in hint.lower()

    def test_junk_transcript_keeps_unverified_lean(self):
        hint = build_turn_hint(
            "wake",
            wake_confidence=0.97,
            wake_verified=False,
            transcript="- Uh-huh. - Eat it.",
        )
        assert hint is not None
        assert "one signal" in hint

    def test_verified_true_gets_normal_posture(self):
        hint = build_turn_hint("wake", wake_confidence=0.95, wake_verified=True)
        assert "addressed to you" in hint.lower()


class TestUnverifiedWakeKeepsSentinelRescue:
    """The /think sentinel double-check must stay available on
    unverified-clip turns — the verify clip itself can be garbage
    (2026-08-15), so the rescue is the last line against a wrong silence."""

    def test_signature_no_longer_takes_wake_verified(self):
        import inspect

        from app.core.turn_context import should_double_check_sentinel

        params = inspect.signature(should_double_check_sentinel).parameters
        assert "wake_verified" not in params

    def test_confident_wake_qualifies_regardless_of_clip_verdict(self):
        # Before: wake_verified=False short-circuited the rescue off. Now
        # the verdict is not consulted at all — a confident wake keeps it.
        from app.core.turn_context import should_double_check_sentinel

        assert should_double_check_sentinel("wake", 0.97, None, None) is True


class TestSharedConfidenceConstant:
    def test_threshold_is_the_direction_hint_constant(self):
        # Item 7 hygiene: the duplicated 0.75s are unified — one source.
        from app.core.direction_hint import BORDERLINE_CONFIDENCE

        assert WAKE_CONFIDENT_THRESHOLD is BORDERLINE_CONFIDENCE or (
            WAKE_CONFIDENT_THRESHOLD == BORDERLINE_CONFIDENCE
        )


class TestSelfPlaybackMediaContext:
    """Media-aware wake hints: when the node reported SELF-PLAYBACK (its own
    speaker was playing music at wake), the wake hint carries a mild
    two-sided media-context line — music is a known false-wake source, but
    users also talk to Jarvis over music, most often to control it. Soft
    bias only; never a suppression instruction."""

    def test_wake_hint_carries_media_context_line(self):
        hint = build_turn_hint(
            "wake",
            wake_confidence=0.95,
            self_playback=True,
            self_playback_kind="music",
        )
        assert hint is not None
        assert hint.startswith("[turn context:")
        assert hint.endswith("]")
        assert "own speaker" in hint
        assert "music" in hint.lower()

    def test_media_line_is_two_sided_not_a_suppression(self):
        hint = build_turn_hint(
            "wake",
            wake_confidence=0.95,
            self_playback=True,
            self_playback_kind="music",
        )
        # Both sides present: false-wake awareness AND talk-over-music.
        assert "false-wake" in hint or "false wake" in hint.lower()
        assert "talk to you over" in hint or "over their music" in hint
        # And it must never instruct silence outright.
        assert "do not act" not in hint.lower()

    def test_media_line_calls_music_control_directed(self):
        hint = build_turn_hint(
            "wake",
            wake_confidence=0.95,
            self_playback=True,
            self_playback_kind="music",
        )
        assert "music-control" in hint.lower()
        assert "directed at you" in hint.lower()

    def test_no_media_line_without_self_playback(self):
        for kwargs in (
            {},
            {"self_playback": None, "self_playback_kind": None},
            {"self_playback": False, "self_playback_kind": None},
        ):
            hint = build_turn_hint("wake", wake_confidence=0.95, **kwargs)
            assert "own speaker" not in hint

    def test_old_node_missing_fields_identical_output(self):
        assert build_turn_hint("wake", wake_confidence=0.95) == build_turn_hint(
            "wake",
            wake_confidence=0.95,
            self_playback=None,
            self_playback_kind=None,
        )

    def test_future_non_music_kind_gets_no_media_line(self):
        hint = build_turn_hint(
            "wake",
            wake_confidence=0.95,
            self_playback=True,
            self_playback_kind="podcast",
        )
        assert "own speaker" not in hint

    def test_missing_kind_still_counts_as_music(self):
        hint = build_turn_hint(
            "wake", wake_confidence=0.95, self_playback=True
        )
        assert "own speaker" in hint

    def test_low_confidence_variant_also_carries_media_line(self):
        hint = build_turn_hint(
            "wake",
            wake_confidence=WAKE_CONFIDENT_THRESHOLD - 0.2,
            self_playback=True,
            self_playback_kind="music",
        )
        assert "low confidence" in hint.lower()
        assert "own speaker" in hint

    def test_unverified_lean_also_carries_media_line(self):
        hint = build_turn_hint(
            "wake",
            wake_confidence=0.95,
            wake_verified=False,
            transcript="what's the weather tomorrow",
            self_playback=True,
            self_playback_kind="music",
        )
        assert "one signal" in hint
        assert "own speaker" in hint

    def test_music_control_overrides_unverified_lean_during_media(self):
        # Residual music bleed degrades the verify clip, so unverified is
        # the EXPECTED verdict on a real mid-music wake — and a bare
        # music-control command is the most likely real utterance. It gets
        # the directed posture, like the device-command guard.
        for cmd in ("Pause.", "Skip this song.", "Turn it down."):
            hint = build_turn_hint(
                "wake",
                wake_confidence=0.97,
                wake_verified=False,
                transcript=cmd,
                self_playback=True,
                self_playback_kind="music",
            )
            assert hint is not None, cmd
            assert "addressed to you" in hint.lower(), cmd
            assert "misfire" not in hint.lower(), cmd

    def test_music_control_without_media_keeps_unverified_lean(self):
        # Off-music, a bare "Pause." with an unverified clip keeps the
        # existing soft misfire lean — the music-control guard only exists
        # during self-playback.
        hint = build_turn_hint(
            "wake",
            wake_confidence=0.97,
            wake_verified=False,
            transcript="Pause.",
        )
        assert "one signal" in hint

    def test_follow_up_hint_unaffected_by_self_playback(self):
        assert build_turn_hint(
            "follow_up",
            follow_up_iteration=1,
            self_playback=True,
            self_playback_kind="music",
        ) == build_turn_hint("follow_up", follow_up_iteration=1)

    def test_chat_hint_unaffected_by_self_playback(self):
        assert build_turn_hint(
            "chat", self_playback=True, self_playback_kind="music"
        ) == build_turn_hint("chat")

    def test_inferred_wake_carries_media_line(self):
        # Old-signature path (no turn_source, pre-wake present) still routes
        # media context when the new fields ride along.
        hint = build_turn_hint(
            None,
            pre_wake_speech_seconds=0.0,
            self_playback=True,
            self_playback_kind="music",
        )
        assert hint is not None
        assert "own speaker" in hint


class TestFollowUpNamedPersonAddressing:
    """2026-08-17 prod incident: a follow-up captured "already done. Wow.
    Miles, come here." — a parent calling their child, a KNOWN household
    member. Telemetry: wake_verdict=none (no speaker clip → verification
    never ran), so the unverified-keyed doubt machinery never engaged and
    the normal follow-up posture answered into the family's conversation.

    A transcript that directly addresses a household member by name is
    near-certainly not for Jarvis — SHAPE evidence, not acoustic, so the
    lean applies on ANY verdict. It is a hint, never a hard cut; the
    imperative device/music guard stays senior; and with no member names
    (old cache, resolve failure) or no match the hint is byte-identical.
    """

    INCIDENT = "already done. Wow. Miles, come here."
    MEMBERS = ["Miles", "Alex"]

    def test_incident_transcript_gets_addressing_lean(self):
        hint = build_turn_hint(
            "follow_up",
            follow_up_iteration=1,
            transcript=self.INCIDENT,
            member_names=self.MEMBERS,
        )
        assert hint is not None
        assert "directly addresses Miles" in hint
        assert "another member of this household" in hint
        assert "<not_for_me/>" in hint

    def test_addressing_lean_is_not_a_hard_cut(self):
        # The hint must keep the answer path open for "Miles, ask Jarvis
        # what time it is" style utterances.
        hint = build_turn_hint(
            "follow_up",
            follow_up_iteration=1,
            transcript=self.INCIDENT,
            member_names=self.MEMBERS,
        )
        assert "unless it ALSO asks you something directly" in hint

    def test_applies_on_any_verdict_not_just_doubted(self):
        # The incident's exact telemetry: no conversation_wake_verdict at
        # all — the lean must not be keyed on "unverified".
        for verdict in (None, "verified", "clip_unreliable", "unverified"):
            hint = build_turn_hint(
                "follow_up",
                follow_up_iteration=1,
                transcript=self.INCIDENT,
                member_names=self.MEMBERS,
                conversation_wake_verdict=verdict,
            )
            assert "directly addresses Miles" in hint, f"verdict={verdict}"

    def test_no_member_names_is_byte_identical(self):
        base = build_turn_hint(
            "follow_up", follow_up_iteration=1, transcript=self.INCIDENT
        )
        for names in (None, []):
            assert build_turn_hint(
                "follow_up",
                follow_up_iteration=1,
                transcript=self.INCIDENT,
                member_names=names,
            ) == base

    def test_no_name_match_is_byte_identical(self):
        transcript = "what about tomorrow then"
        base = build_turn_hint(
            "follow_up", follow_up_iteration=1, transcript=transcript
        )
        assert build_turn_hint(
            "follow_up",
            follow_up_iteration=1,
            transcript=transcript,
            member_names=self.MEMBERS,
        ) == base

    def test_jarvis_name_is_excluded(self):
        # A household member literally named "Jarvis" (or the wake phrase
        # itself) must never earn the not-for-you lean — "Jarvis, what
        # time is it" is the MOST directed shape there is.
        hint = build_turn_hint(
            "follow_up",
            follow_up_iteration=1,
            transcript="Jarvis, what about tomorrow",
            member_names=["Jarvis", "Miles"],
        )
        assert "directly addresses" not in hint

    def test_device_command_guard_stays_senior(self):
        # "Miles, turn off the lights" — imperative device shape wins;
        # the model must not be talked out of a real command.
        hint = build_turn_hint(
            "follow_up",
            follow_up_iteration=1,
            transcript="Miles, turn off the lights",
            member_names=self.MEMBERS,
        )
        assert "directly addresses" not in hint

    def test_music_control_guard_stays_senior_during_playback(self):
        hint = build_turn_hint(
            "follow_up",
            follow_up_iteration=1,
            transcript="Miles, pause",
            member_names=self.MEMBERS,
            self_playback=True,
            self_playback_kind="music",
        )
        assert "directly addresses" not in hint

    def test_doubted_posture_also_carries_the_lean(self):
        hint = build_turn_hint(
            "follow_up",
            follow_up_iteration=2,
            transcript=self.INCIDENT,
            member_names=self.MEMBERS,
            conversation_wake_verdict="unverified",
        )
        assert "suspected detector misfire" in hint
        assert "directly addresses Miles" in hint

    def test_wake_turns_are_unaffected(self):
        # Shape lean is follow-up-only: a fresh wake saying a member's
        # name ("Jarvis, call Miles down for dinner" mis-split) keeps the
        # directed posture.
        base = build_turn_hint("wake", wake_confidence=0.9, transcript=self.INCIDENT)
        assert build_turn_hint(
            "wake",
            wake_confidence=0.9,
            transcript=self.INCIDENT,
            member_names=self.MEMBERS,
        ) == base

    def test_hint_stays_bracketed(self):
        hint = build_turn_hint(
            "follow_up",
            follow_up_iteration=1,
            transcript=self.INCIDENT,
            member_names=self.MEMBERS,
        )
        assert hint.startswith("[turn context:")
        assert hint.endswith("]")

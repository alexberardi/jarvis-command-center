"""
Tests for shared core rules module.

Verifies:
- All rule constants contain expected key phrases
- build_identity_header produces correct identity + context header
- build_rules_block produces correct multi-line rules with customization
- build_fallback_line produces correct fallback for both styles
"""

from app.core.prompt_providers.shared.core_rules import (
    ANTI_HALLUCINATION_MANDATE,
    FALLBACK_BRIEF_REPLY,
    FALLBACK_BRIEF_REPLY_HERMES,
    NOT_FOR_ME_INSTRUCTION,
    RULE_BEST_MATCH_INTENT,
    RULE_DATE_PARAMS,
    RULE_EXTRACT_PARAMS,
    RULE_ONE_AT_A_TIME,
    RULE_POPULATE_REQUIRED,
    RULE_USE_ACTUAL_PARAM_NAMES,
    build_fallback_line,
    build_identity_header,
    build_personality_block,
    build_personality_reminder,
    build_rules_block,
    build_speaker_block,
)


class TestRuleConstants:
    """Test that rule constants contain expected key phrases."""

    def test_rule_one_at_a_time(self):
        assert "ONE" in RULE_ONE_AT_A_TIME
        assert "{terminology}" in RULE_ONE_AT_A_TIME

    def test_rule_use_actual_param_names(self):
        assert "actual parameter names" in RULE_USE_ACTUAL_PARAM_NAMES
        assert "{terminology}" in RULE_USE_ACTUAL_PARAM_NAMES

    def test_rule_best_match_intent(self):
        assert "actual topic" in RULE_BEST_MATCH_INTENT
        assert "keyword overlap" in RULE_BEST_MATCH_INTENT
        assert "get_command_utterance_examples" in RULE_BEST_MATCH_INTENT

    def test_rule_extract_params(self):
        assert "Extract parameters" in RULE_EXTRACT_PARAMS
        assert "clarification" in RULE_EXTRACT_PARAMS

    def test_rule_extract_params_covers_optional_params(self):
        # Decisiveness guardrail: the model must apply an OPTIONAL param's default
        # instead of asking (e.g. "what time is it" → get_current_time with no
        # location, not "what location?"). Locks the behavior-corpus fix.
        lowered = RULE_EXTRACT_PARAMS.lower()
        assert "optional" in lowered
        assert "default" in lowered
        # Clarification is restricted to REQUIRED params only.
        assert "required" in lowered
        # Bias toward acting rather than asking.
        assert "acting" in lowered or "act" in lowered

    def test_rule_date_params(self):
        assert "resolved_datetimes" in RULE_DATE_PARAMS
        assert "MUST use ONLY" in RULE_DATE_PARAMS
        assert "NEVER output ISO dates" in RULE_DATE_PARAMS
        assert "today" in RULE_DATE_PARAMS
        assert "tomorrow" in RULE_DATE_PARAMS
        assert "yesterday" in RULE_DATE_PARAMS
        assert "next_week" in RULE_DATE_PARAMS

    def test_rule_populate_required(self):
        assert "ALL required" in RULE_POPULATE_REQUIRED
        assert "{terminology}" in RULE_POPULATE_REQUIRED
        assert "most natural default" in RULE_POPULATE_REQUIRED

    def test_anti_hallucination_mandate(self):
        assert "MUST call a function" in ANTI_HALLUCINATION_MANDATE
        assert "NEVER fabricate" in ANTI_HALLUCINATION_MANDATE
        # The mandate now carves out personal memory explicitly so the
        # model doesn't refuse to answer User Profile questions when no
        # tool matches the domain (the bug that caused "who's my favorite
        # baseball team?" to get silenced).
        assert "User Profile" in ANTI_HALLUCINATION_MANDATE

    def test_fallback_brief_reply(self):
        assert "brief spoken reply" in FALLBACK_BRIEF_REPLY
        assert "NO matching" in FALLBACK_BRIEF_REPLY

    def test_fallback_brief_reply_hermes(self):
        assert "brief spoken reply" in FALLBACK_BRIEF_REPLY_HERMES
        assert "no tool needed" in FALLBACK_BRIEF_REPLY_HERMES


class TestNotForMeInstruction:
    """Lock in the concrete cues the LLM uses for the not_for_me decision.

    The instruction is the *single* place where Jarvis is taught WHEN to
    emit ``<not_for_me/>``. Each cue below is load-bearing — silently
    dropping one in a future edit would walk us back toward the v1
    "default to answer, beyond reasonable doubt" stance that was missing
    the false-wakes we deployed against in beta.
    """

    def test_mentions_sentinel_literal(self):
        # The instruction must name the literal token the parser detects.
        assert "<not_for_me/>" in NOT_FOR_ME_INSTRUCTION

    def test_specifies_no_prose_no_tool(self):
        # Without this the LLM tends to apologize + emit the sentinel
        # alongside prose, which the parser still catches but defeats
        # the "silent abort" intent on the streaming path.
        assert "no prose" in NOT_FOR_ME_INSTRUCTION
        assert "no tool" in NOT_FOR_ME_INSTRUCTION

    def test_keeps_default_to_answer_guardrail(self):
        # The hard floor — without this the model gets too eager and
        # silences real but oddly-phrased requests.
        assert "Silencing a real request is worse" in NOT_FOR_ME_INSTRUCTION

    def test_addressee_cue_present(self):
        # Cue 1: someone else by name. The most distinctive ambient
        # signature when a household has multiple people in the room.
        for name in ("Mom", "Sarah", "Dad"):
            assert name in NOT_FOR_ME_INSTRUCTION, name

    def test_mid_narrative_cue_present(self):
        # Cue 2: fragment of an ongoing exchange between others.
        assert "Mid-narrative fragment" in NOT_FOR_ME_INSTRUCTION
        assert "and then I told him" in NOT_FOR_ME_INSTRUCTION

    def test_ungrounded_reference_cue_present(self):
        # Cue 3: references nothing in the conversation, profile, or
        # tool output. This is the "current conversation" cue.
        assert "not in this conversation" in NOT_FOR_ME_INSTRUCTION
        assert "not in the user's profile" in NOT_FOR_ME_INSTRUCTION
        assert "not in any tool result" in NOT_FOR_ME_INSTRUCTION

    def test_conjunction_opener_cue_present(self):
        # Cue 4: starts mid-sentence with a conjunction — wake fired
        # while two people were already talking.
        assert "Opens with a conjunction" in NOT_FOR_ME_INSTRUCTION

    def test_direction_hint_cue_present(self):
        # Cue 5: trust the acoustic hint the node ships when present.
        # If we ever rename ``[direction hint: ...]`` the LLM stops
        # noticing it — fail loudly here so we catch the drift.
        assert "[direction hint:" in NOT_FOR_ME_INSTRUCTION
        assert "trust it" in NOT_FOR_ME_INSTRUCTION

    def test_borderline_examples_kept_addressed(self):
        # Examples that LOOK borderline but should still get answered —
        # without these the model errs aggressive on common short utterances.
        assert "thanks" in NOT_FOR_ME_INSTRUCTION
        assert "never mind" in NOT_FOR_ME_INSTRUCTION

    def test_sentinel_is_framed_as_first_class_action(self):
        # The prompt must name the sentinel as a terminal action that
        # outranks other rules. Without this framing the model sees the
        # sentinel as a meta-comment that competes with the "must call a
        # tool" guards, which is what let the 2026-06-02 regression slip
        # through.
        body = NOT_FOR_ME_INSTRUCTION.lower()
        assert "first-class" in body or "first class" in body
        assert "terminal action" in body
        assert "outranks" in body

    def test_requires_positive_addressing_signal(self):
        # The new rule: don't answer just because nothing fired the
        # negative cues — require a positive signal that this utterance
        # is for you.
        body = NOT_FOR_ME_INSTRUCTION.lower()
        assert "positive" in body
        # All three positive-signal categories must be named.
        assert "names you explicitly" in body
        assert "clear imperative" in body
        # The wording "question only you can answer" is load-bearing — it
        # tells the model the question must require Jarvis specifically.
        assert "only you can answer" in body

    def test_stt_artifact_cue_present(self):
        # The 2026-06-02 prod hallucination ("*sniff*" → "I smell
        # something burning") needs the LLM to recognize Whisper's
        # bracketed action notation as non-speech, not as content.
        body = NOT_FOR_ME_INSTRUCTION
        assert "STT artifact" in body
        assert "*sniff*" in body
        assert "*laughter*" in body or "[laughter]" in body or "<inaudible>" in body

    def test_emotional_fragment_cue_present(self):
        # Prevents "I'm sorry" / "oh no" / a sob from triggering a
        # therapist-style response from Jarvis.
        body = NOT_FOR_ME_INSTRUCTION
        assert "emotional reaction" in body or "emotional fragment" in body
        # The exact "I'm sorry" example is the smoking-gun input from the
        # 2026-06-02 incident — lock the wording.
        assert "I'm sorry" in body

    def test_quiet_hint_branch_holds_the_answer_bias(self):
        # The "Silencing a real request is worse" guardrail must live
        # under the quiet-hint branch, NOT as the unconditional default
        # — under an ambient/no-hint signal we require positive evidence
        # instead.
        body = NOT_FOR_ME_INSTRUCTION
        # The phrase still appears (existing guardrail test) AND must be
        # qualified with "quiet-hint" framing nearby. Soft check: look
        # for the qualifier within ~200 chars of the phrase.
        idx = body.find("Silencing a real request is worse")
        assert idx != -1
        window = body[max(0, idx - 200):idx + 200].lower()
        assert "quiet" in window, (
            "The 'silencing a real request is worse' bias must be scoped "
            "to the quiet-hint branch; otherwise it overrides the new "
            "positive-evidence rule on ambient turns."
        )


class TestBuildIdentityHeader:
    """build_identity_header is now SPEAKER-AGNOSTIC: identity + room/style
    only, no speaker name and no memories (those moved to build_speaker_block
    so the cached prefix is identical across household members)."""

    def test_agnostic_output(self):
        result = build_identity_header("kitchen", "brief")
        assert result == (
            "You are Jarvis, a function calling voice assistant.\n"
            "Context: room=kitchen, style=brief"
        )

    def test_no_speaker_or_memories_for_any_input(self):
        result = build_identity_header("living_room", "verbose")
        assert "room=living_room" in result
        assert "style=verbose" in result
        assert "You are speaking with" not in result
        assert "User Profile" not in result

    def test_starts_with_jarvis_two_lines(self):
        result = build_identity_header("r", "v")
        assert result.startswith("You are Jarvis")
        assert len(result.split("\n")) == 2

    def test_identical_across_speakers(self):
        # The whole point: the cached prefix must be byte-identical regardless
        # of who is speaking (speaker is no longer an input).
        assert build_identity_header("kitchen", "brief") == build_identity_header("kitchen", "brief")


class TestBuildSpeakerBlock:
    """build_speaker_block carries the per-turn speaker name + memories that
    used to live in the identity header — now injected as a trailing system
    message after the cached prefix."""

    def test_name_and_memories(self):
        result = build_speaker_block("alex", "- Likes coffee black")
        assert "You are speaking with alex." in result
        assert "User Profile" in result
        # Load-bearing anti-recall directive: the model must answer profile
        # facts DIRECTLY, never fire a redundant `recall` round-trip for a fact
        # already listed here (the ~2.2s-per-call latency fix on the 9B).
        assert "answer DIRECTLY from this list" in result
        assert "do NOT call recall" in result
        # Action carve-out: a REPORT/REQUEST to do something must still call the
        # tool even when its subject is a listed profile fact (medication bug).
        assert "MUST call the tool" in result
        assert "- Likes coffee black" in result

    def test_default_name_no_memories_is_empty(self):
        assert build_speaker_block("default", "") == ""

    def test_name_only(self):
        result = build_speaker_block("alex", "")
        assert result == "You are speaking with alex."
        assert "User Profile" not in result

    def test_memories_only_no_name(self):
        # default sentinel suppresses the speaking-with line, memories remain
        result = build_speaker_block("default", "- Vegetarian")
        assert "You are speaking with" not in result
        assert "User Profile" in result
        assert "- Vegetarian" in result

    def test_empty_inputs(self):
        assert build_speaker_block("", "") == ""


class TestBuildPersonalityBlock:
    """build_personality_block renders the household voice into a fenced
    <personality> block. VOICE-only; must never be empty-fenced or leak into
    tool-calling — the frame line + tag are the wall."""

    def test_empty_returns_empty(self):
        assert build_personality_block("") == ""
        assert build_personality_block("   ") == ""
        assert build_personality_block(None) == ""

    def test_wraps_in_personality_tags(self):
        out = build_personality_block("You're warm and folksy.")
        assert out.startswith("<personality>")
        assert out.endswith("</personality>")
        assert "You're warm and folksy." in out

    def test_includes_voice_only_frame(self):
        from app.services.persona_presets import PERSONA_FRAME
        out = build_personality_block("Be terse.")
        assert PERSONA_FRAME in out
        assert "never your tools or safety" in out


class TestBuildPersonalityReminder:
    """build_personality_reminder restates the voice at the END of the prompt for
    recency — the fix for a small model losing the top-of-prompt persona."""

    def test_empty_returns_empty(self):
        assert build_personality_reminder("") == ""
        assert build_personality_reminder(None) == ""

    def test_restates_and_is_scoped_to_voice(self):
        out = build_personality_reminder("Be warm.")
        assert "Be warm." in out
        assert "every spoken reply" in out
        assert "function" in out  # walls off tool-calling
        assert "Don't announce" in out or "don't announce" in out


class TestBuildRulesBlock:
    """Test build_rules_block output."""

    def test_default_produces_all_rules(self):
        result = build_rules_block()
        assert result.startswith("Rules:\n")
        assert "- Always populate ALL required function parameters" in result
        assert "- Call ONE function at a time" in result
        assert "- Use the actual parameter names from the function schema above." in result
        assert "actual topic" in result
        assert "NEVER output ISO dates" in result
        assert "- Extract parameters from the user's words" in result

    def test_terminology_tool(self):
        result = build_rules_block(terminology="tool")
        assert "Call ONE tool at a time" in result
        assert "actual topic" in result
        assert "Always populate ALL required tool parameters" in result
        # "function" should NOT appear in the rules when terminology="tool"
        # (except in the date/extract rules which don't use {terminology})
        assert "Call ONE function" not in result

    def test_param_names_rule_none_omits_it(self):
        result = build_rules_block(param_names_rule=None)
        assert "actual parameter names" not in result
        # Other rules should still be present
        assert "Call ONE function" in result
        assert "NEVER output ISO dates" in result

    def test_custom_param_names_rule(self):
        custom = 'Use the actual parameter names from the function schema above \u2014 NOT "param".'
        result = build_rules_block(param_names_rule=custom)
        assert custom in result

    def test_extra_rules_appended(self):
        result = build_rules_block(extra_rules=["Always say please.", "Be polite."])
        assert "- Always say please." in result
        assert "- Be polite." in result
        # Extra rules come after the standard rules
        lines = result.split("\n")
        standard_count = 8  # "Rules:" + 7 standard rules
        assert len(lines) == standard_count + 2

    def test_extra_rules_none(self):
        result_none = build_rules_block(extra_rules=None)
        result_default = build_rules_block()
        assert result_none == result_default

    def test_line_count_default(self):
        """Default: Rules: header + 7 rules = 8 lines."""
        result = build_rules_block()
        lines = result.split("\n")
        assert len(lines) == 8

    def test_line_count_no_param_names(self):
        """With param_names_rule=None: Rules: header + 6 rules = 7 lines."""
        result = build_rules_block(param_names_rule=None)
        lines = result.split("\n")
        assert len(lines) == 7

    def test_terminology_substitution_in_extra_rules(self):
        result = build_rules_block(
            terminology="tool",
            extra_rules=["Always use the {terminology} correctly."],
        )
        assert "Always use the tool correctly." in result


class TestBuildFallbackLine:
    """Test build_fallback_line output."""

    def test_default_standard_style(self):
        result = build_fallback_line()
        assert "Only respond with a brief spoken reply" in result
        assert "NO matching tool" in result

    def test_hermes_style(self):
        result = build_fallback_line(hermes_style=True)
        assert "For final answers with no tool needed" in result
        assert "brief spoken reply" in result

    def test_default_not_hermes_style(self):
        result = build_fallback_line()
        assert "no tool needed" not in result

    def test_hermes_not_standard_style(self):
        result = build_fallback_line(hermes_style=True)
        assert "NO matching tool" not in result


class TestProviderOutputConsistency:
    """Verify that the shared rules produce output matching each provider's needs."""

    def test_qwen25_rules_match(self):
        """Qwen25 uses all defaults."""
        result = build_rules_block()
        expected_lines = [
            "Rules:",
            "- Always populate ALL required function parameters. If a required parameter is implied but not stated, use the most natural default (e.g., date parameters default to 'today' when no date is mentioned).",
            "- Call ONE function at a time to fulfill requests.",
            "- Use the actual parameter names from the function schema above.",
            "- Pick the function whose described purpose matches the user's actual topic — match on meaning, not keyword overlap. Use get_command_utterance_examples if unsure.",
            '- For date parameters like resolved_datetimes, you MUST use ONLY these natural date keys as string values: "today", "tomorrow", "day_after_tomorrow", "yesterday", "this_weekend", "this_year", "next_week". NEVER output ISO dates (e.g. 2025-01-01T00:00:00Z) or timestamps — the downstream system resolves these keys to actual dates.',
            "- Extract parameters from the user's words. Only request clarification if a REQUIRED parameter is truly missing or ambiguous — never ask about an OPTIONAL parameter; apply the tool's documented default instead (e.g. omit location for local time). Prefer acting on sensible defaults over asking.",
            '- User input comes from speech-to-text and may contain transcription errors. Interpret homophones and near-misses charitably (e.g., "watts" → "what\'s", "won" → "one", "whether" → "weather"). Proper nouns like team names, cities, and people are often misspelled — infer the intended name (e.g., "Ankeys" → "Yankees", "Albukirky" → "Albuquerque").',
        ]
        assert result == "\n".join(expected_lines)

    def test_llama31_rules_match(self):
        """Llama31 uses custom param_names_rule with NOT 'param'."""
        result = build_rules_block(
            param_names_rule='Use the actual parameter names from the function schema above \u2014 NOT "param".',
        )
        assert '— NOT "param".' in result
        assert "Call ONE function" in result

    def test_mistral_rules_match(self):
        """Mistral uses custom param_names_rule referencing 'definitions'."""
        result = build_rules_block(
            param_names_rule="Use the actual parameter names from the function definitions above.",
        )
        assert "function definitions above" in result

    def test_hermes_rules_match(self):
        """Hermes uses terminology='tool' and no param_names_rule."""
        result = build_rules_block(param_names_rule=None, terminology="tool")
        expected_lines = [
            "Rules:",
            "- Always populate ALL required tool parameters. If a required parameter is implied but not stated, use the most natural default (e.g., date parameters default to 'today' when no date is mentioned).",
            "- Call ONE tool at a time to fulfill requests.",
            "- Pick the tool whose described purpose matches the user's actual topic — match on meaning, not keyword overlap. Use get_command_utterance_examples if unsure.",
            '- For date parameters like resolved_datetimes, you MUST use ONLY these natural date keys as string values: "today", "tomorrow", "day_after_tomorrow", "yesterday", "this_weekend", "this_year", "next_week". NEVER output ISO dates (e.g. 2025-01-01T00:00:00Z) or timestamps — the downstream system resolves these keys to actual dates.',
            "- Extract parameters from the user's words. Only request clarification if a REQUIRED parameter is truly missing or ambiguous — never ask about an OPTIONAL parameter; apply the tool's documented default instead (e.g. omit location for local time). Prefer acting on sensible defaults over asking.",
            '- User input comes from speech-to-text and may contain transcription errors. Interpret homophones and near-misses charitably (e.g., "watts" → "what\'s", "won" → "one", "whether" → "weather"). Proper nouns like team names, cities, and people are often misspelled — infer the intended name (e.g., "Ankeys" → "Yankees", "Albukirky" → "Albuquerque").',
        ]
        assert result == "\n".join(expected_lines)

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
    RULE_BEST_MATCH_INTENT,
    RULE_DATE_PARAMS,
    RULE_EXTRACT_PARAMS,
    RULE_ONE_AT_A_TIME,
    RULE_POPULATE_REQUIRED,
    RULE_USE_ACTUAL_PARAM_NAMES,
    build_fallback_line,
    build_identity_header,
    build_rules_block,
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


class TestBuildIdentityHeader:
    """Test build_identity_header output."""

    def test_default_values(self):
        result = build_identity_header("kitchen", "alex", "brief")
        assert result == (
            "You are Jarvis, a function calling voice assistant.\n"
            "Context: room=kitchen, style=brief\n"
            "You are speaking with alex."
        )

    def test_unknown_defaults(self):
        # user="default" sentinel suppresses the "speaking with" line
        result = build_identity_header("unknown", "default", "brief")
        assert "room=unknown" in result
        assert "style=brief" in result
        assert "You are speaking with" not in result

    def test_custom_values(self):
        result = build_identity_header("living_room", "bob", "verbose")
        assert "room=living_room" in result
        assert "style=verbose" in result
        assert "You are speaking with bob." in result

    def test_starts_with_jarvis(self):
        result = build_identity_header("r", "u", "v")
        assert result.startswith("You are Jarvis")

    def test_three_lines_with_user(self):
        # Identity + Context + "speaking with" line = 3 lines when user is set
        result = build_identity_header("r", "u", "v")
        lines = result.split("\n")
        assert len(lines) == 3

    def test_two_lines_without_user(self):
        # user="default" omits the speaking-with line → 2 lines
        result = build_identity_header("r", "default", "v")
        lines = result.split("\n")
        assert len(lines) == 2


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
            "- Extract parameters from the user's words; only request clarification if required params are truly missing/ambiguous.",
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
            "- Extract parameters from the user's words; only request clarification if required params are truly missing/ambiguous.",
            '- User input comes from speech-to-text and may contain transcription errors. Interpret homophones and near-misses charitably (e.g., "watts" → "what\'s", "won" → "one", "whether" → "weather"). Proper nouns like team names, cities, and people are often misspelled — infer the intended name (e.g., "Ankeys" → "Yankees", "Albukirky" → "Albuquerque").',
        ]
        assert result == "\n".join(expected_lines)

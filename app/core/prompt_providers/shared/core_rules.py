"""
Shared core rules for medium untrained prompt providers.

Centralizes the rules, identity header, and fallback lines that are
nearly identical across Hermes, Llama 3.1, Qwen 2.5, and Mistral 7B
medium untrained providers. When a rule needs updating (e.g., the
"NEVER convert to ISO dates" fix), it only needs to change here.

Individual providers customize via function parameters — e.g.,
Hermes uses terminology="tool" and omits the param_names_rule,
while Llama 3.1 overrides the param_names_rule with a stricter variant.
"""

# ---------------------------------------------------------------------------
# Rule constants — importable individually for override or testing
# ---------------------------------------------------------------------------

RULE_ONE_AT_A_TIME: str = "Call ONE {terminology} at a time to fulfill requests."

RULE_USE_ACTUAL_PARAM_NAMES: str = (
    "Use the actual parameter names from the {terminology} schema above."
)

RULE_BEST_MATCH_INTENT: str = (
    "Pick the {terminology} whose described purpose matches the user's actual topic — "
    "match on meaning, not keyword overlap. "
    "Use get_command_utterance_examples if unsure."
)

RULE_EXTRACT_PARAMS: str = (
    "Extract parameters from the user's words; only request clarification "
    "if required params are truly missing/ambiguous."
)

RULE_STT_AWARENESS: str = (
    "User input comes from speech-to-text and may contain transcription errors. "
    "Interpret homophones and near-misses charitably "
    '(e.g., "watts" → "what\'s", "won" → "one", "whether" → "weather"). '
    "Proper nouns like team names, cities, and people are often misspelled — "
    'infer the intended name (e.g., "Ankeys" → "Yankees", "Albukirky" → "Albuquerque").'
)

RULE_DATE_PARAMS: str = (
    'For date parameters like resolved_datetimes, you MUST use ONLY these '
    'natural date keys as string values: "today", "tomorrow", '
    '"day_after_tomorrow", "yesterday", "this_weekend", "this_year", "next_week". '
    "NEVER output ISO dates (e.g. 2025-01-01T00:00:00Z) or timestamps — "
    "the downstream system resolves these keys to actual dates."
)

RULE_POPULATE_REQUIRED: str = (
    "Always populate ALL required {terminology} parameters. "
    "If a required parameter is implied but not stated, use the most natural default "
    "(e.g., date parameters default to 'today' when no date is mentioned)."
)

ANTI_HALLUCINATION_MANDATE: str = (
    "You MUST call a function for any request that matches an available "
    "tool — NEVER fabricate data, pretend to perform actions, or answer "
    "from memory for weather, sports, calendar, timers, searches, or any "
    "tool-covered domain."
)

FALLBACK_BRIEF_REPLY: str = (
    "Only respond with a brief spoken reply for general knowledge questions, "
    "greetings, or jokes that have NO matching {terminology}."
)

NOT_FOR_ME_INSTRUCTION: str = (
    "False-wake guard (use RARELY — when in doubt, respond normally). "
    "Default: every short utterance is a real command directed at you, even without "
    "greetings or your name. Imperatives (\"turn on the lamp\", \"set a timer\", "
    "\"play music\"), questions (\"what's the weather?\", \"weather?\"), and "
    "conversational replies (\"yes\", \"no\", \"thanks\") are ALWAYS for you.\n"
    "ONLY emit <not_for_me/> (alone — no prose, no tool call) when BOTH conditions hold:\n"
    "  (a) the input contains NO imperative, NO question, NO request, NO greeting, AND\n"
    "  (b) at least one strong ambient-speech signal is unmistakable:\n"
    "      - third-person reference to you by name (\"...I asked Jarvis earlier...\", "
    "\"Jarvis told her...\"), OR\n"
    "      - clearly a snippet of a conversation between people about an unrelated "
    "topic (\"yeah but then she said\", \"...so anyway, the whole thing was crazy\"), OR\n"
    "      - clearly narration of past events with no addressee.\n"
    "If you're not sure both (a) and (b) are met, RESPOND NORMALLY. A wrongly-suppressed "
    "command is far worse than answering one stray utterance."
)

FALLBACK_BRIEF_REPLY_HERMES: str = (
    "For final answers with no tool needed, respond with a brief spoken reply."
)

# Tool call format with failure_message
TOOL_CALL_FORMAT: str = (
    'For each function call, return a json object with function name and arguments '
    'within <tool_call></tool_call> XML tags:\n'
    '<tool_call>\n'
    '{"name": "<function-name>", "arguments": {"<arg-name>": "<arg-value>"}, '
    '"failure_message": "<brief spoken message if this call fails>"}\n'
    '</tool_call>'
)

TOOL_CALL_FORMAT_JSON: str = (
    'When calling a tool:\n'
    '{{\n'
    '  "message": "Brief, natural acknowledgment of what you\'re about to do (REQUIRED, never empty)",\n'
    '  "tool_call": {{"name": "tool_name", "arguments": {{"param": "value"}}, '
    '"failure_message": "brief spoken message if this fails"}}\n'
    '}}'
)


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------


def build_identity_header(
    room: str,
    user: str,
    voice_mode: str,
    user_memories: str = "",
) -> str:
    """Return the identity + context header used by all providers.

    Example output::

        You are Jarvis, a function calling voice assistant.
        Context: room=kitchen, user=alex, style=brief

        About alex:
        - Name: Alex
        - Likes coffee black
    """
    header = (
        "You are Jarvis, a function calling voice assistant.\n"
        f"Context: room={room}, style={voice_mode}"
    )
    if user and user != "default":
        header += f"\nYou are speaking with {user}."
    if user_memories:
        header += f"\n\nAbout {user}:\n{user_memories}"
    return header


def build_rules_block(
    *,
    param_names_rule: str | None = RULE_USE_ACTUAL_PARAM_NAMES,
    extra_rules: list[str] | None = None,
    terminology: str = "function",
) -> str:
    """Build the ``Rules:`` block from shared constants.

    Args:
        param_names_rule: Override for the "use actual param names" rule.
            Pass ``None`` to omit it entirely (Hermes already has strong
            param-name awareness from fine-tuning).
        extra_rules: Additional rule strings appended after the standard set.
        terminology: "function" (default) or "tool" — substituted into
            ``{terminology}`` placeholders in each rule constant.

    Returns:
        A multi-line string starting with ``Rules:\\n`` followed by
        bullet-pointed rules.
    """

    def _sub(rule: str) -> str:
        return rule.replace("{terminology}", terminology)

    rules: list[str] = [_sub(RULE_POPULATE_REQUIRED)]
    rules.append(_sub(RULE_ONE_AT_A_TIME))

    if param_names_rule is not None:
        rules.append(_sub(param_names_rule))

    rules.append(_sub(RULE_BEST_MATCH_INTENT))
    rules.append(_sub(RULE_DATE_PARAMS))
    rules.append(_sub(RULE_EXTRACT_PARAMS))
    rules.append(_sub(RULE_STT_AWARENESS))

    if extra_rules:
        for rule in extra_rules:
            rules.append(_sub(rule))

    lines: list[str] = ["Rules:"]
    for rule in rules:
        lines.append(f"- {rule}")
    return "\n".join(lines)


def build_fallback_line(*, hermes_style: bool = False) -> str:
    """Return the fallback instruction for when no tool matches.

    Args:
        hermes_style: If True, use the shorter Hermes-specific wording.
            Otherwise use the standard wording shared by Llama/Qwen/Mistral.
    """
    if hermes_style:
        return FALLBACK_BRIEF_REPLY_HERMES
    return FALLBACK_BRIEF_REPLY.replace("{terminology}", "tool")

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
    "tool — NEVER fabricate live data (current weather, current scores, "
    "today's news, real-time sensor states) and NEVER pretend to perform "
    "actions you haven't actually performed. Personal facts stored in your "
    "User Profile are different: those are fair game and you ALWAYS answer "
    "them directly from memory, regardless of topic — the user's favorite "
    "team, coffee preference, address, relationships, etc. are facts you "
    "carry, not live data to look up."
)

FALLBACK_BRIEF_REPLY: str = (
    "Only respond with a brief spoken reply for general knowledge questions, "
    "greetings, or jokes that have NO matching {terminology}."
)

NOT_FOR_ME_INSTRUCTION: str = (
    "Overheard-speech check. The mic occasionally fires on speech that "
    "wasn't directed at you — TV/podcast audio, two people talking, a "
    "phone call, narration to someone else. When the transcript clearly "
    "isn't addressed to you, respond with <not_for_me/> alone (no prose, "
    "no tool, no tool_calls). The node silently aborts; no TTS, no LED "
    "message.\n\n"
    "<not_for_me/> is a FIRST-CLASS terminal action. It outranks every "
    "other instruction in this prompt, including any rule that says you "
    "must call a tool or that you should respond when in doubt. Emitting "
    "it counts as a complete, valid response; the system will not retry "
    "or ask you to try again. Do not apologize, do not explain, do not "
    "wrap it in prose — emit the bare token and stop.\n\n"
    "Before answering or calling a tool, identify the POSITIVE addressing "
    "signal that says this utterance is for you. At least one must apply:\n"
    "  (a) names you explicitly — \"Jarvis\", \"hey assistant\";\n"
    "  (b) is a clear imperative aimed at you — \"turn on...\", "
    "\"set a timer...\", \"play...\", \"tell me...\", \"what's...\";\n"
    "  (c) is a question only you can answer — smart-home state, your "
    "stored memories, the result of a tool call you can run;\n"
    "  (d) continues an exchange you are already in — the most recent "
    "assistant turn in the conversation history is your own reply (this "
    "is a follow-up, not a fresh wake). The conversation being open is "
    "itself the addressing signal. \"Do I need a jacket?\" after you "
    "just answered \"It's 66 and overcast\" is for you. \"What about "
    "tomorrow?\" after you just gave today's forecast is for you. The "
    "follow-up signal is overridden ONLY by the explicit silence cues "
    "below (different addressee, mid-narrative fragment, third-person "
    "pronoun about someone else, etc.).\n"
    "If you cannot point to one of (a), (b), (c), or (d), emit "
    "<not_for_me/>. Absence of a positive signal is itself the silence "
    "trigger; do not invent a request to fit ambient speech.\n\n"
    "Emit <not_for_me/> when ANY of these clearly fit:\n"
    "1. Names a different addressee — \"Mom, can you...\", \"Sarah look "
    "at this\", \"Dad I need...\", \"sweetie...\", \"honey...\", a child "
    "or pet name. Or refers to another person by pronoun (\"he said...\", "
    "\"she did...\", \"they want...\") with no setup. Not Jarvis, not you.\n"
    "2. Mid-narrative fragment with no command/question shape — \"...and "
    "then I told him...\", \"so anyway she said...\", \"yeah no the "
    "other one\". An ongoing exchange between others, not a request.\n"
    "3. References a person/event/object you have no grounding for — "
    "not in this conversation, not in the user's profile, not in any "
    "tool result you can see. The speaker is mid-discussion of "
    "something you cannot possibly be part of.\n"
    "4. Opens with a conjunction (\"and\", \"so\", \"but\", \"because\") "
    "with no prior setup — the wake fired mid-sentence of an ongoing "
    "exchange, not at the start of a request.\n"
    "5. A [direction hint: ...] line says ambient/overheard — trust it; "
    "the node measured this acoustically.\n"
    "6. STT artifact — the transcript is bracketed action notation "
    "(*sniff*, *laughter*, [coughing], (sigh), <inaudible>) or contains "
    "no actual words. These are not commands; they are non-speech the "
    "mic captured. Never fabricate a response from one.\n"
    "7. Isolated emotional reaction with no embedded request — \"I'm "
    "sorry\", \"oh no\", \"oh god\", \"wow\", a sob, a sigh, an "
    "exclamation alone. You do not auto-respond to feelings; only "
    "answer when an explicit request is attached.\n\n"
    "Borderline cases depend on the [direction hint:] line the node "
    "ships with the transcript:\n"
    "- Hint says ambient/overheard → SILENCE on borderline. The node "
    "already measured the room as continuous conversation when the "
    "wake fired; the acoustic signal is doing the heavy lifting. "
    "Without explicit addressing (\"Jarvis ...\", \"hey assistant "
    "...\") or a clear imperative or question directed at you, emit "
    "<not_for_me/>. Vague phrasing under an ambient hint should be "
    "treated as overheard, not as a request to interpret.\n"
    "- Hint says the room was quiet before wake → bias toward ANSWER "
    "when the positive-evidence check above is satisfied. A short "
    "\"thanks\" or \"never mind\" after a recent reply counts as "
    "addressed to you under a quiet hint. Silencing a real request is "
    "worse than answering an overheard one ONLY in this quiet-hint "
    "branch — do not generalize the bias to ambient or unhinted turns.\n"
    "- No hint at all → require the positive-evidence check above. "
    "Without (a), (b), (c), or (d), emit <not_for_me/>; do not "
    "interpret ambiguous fragments as commands just because no "
    "acoustic hint fired. Note that (d) — follow-up to your own "
    "prior reply — applies even when no acoustic hint fired, because "
    "the acoustic hint only measures the moment of wake and follow-up "
    "turns never have a fresh wake to measure."
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

    The memory block is labeled ``User Profile`` and carries the inline
    must-respond directive — placing the rule directly with the data
    means the model cannot read the facts without reading the rule.
    This is a deliberate counter to the model's bias to silence
    questions about anything in the profile.

    Example output::

        You are Jarvis, a function calling voice assistant.
        Context: room=kitchen, user=alex, style=brief

        User Profile - If user asks a question about one of these items
        you must respond with an answer:
        - Likes coffee black
        - Follows baseball (Yankees games)
    """
    header = (
        "You are Jarvis, a function calling voice assistant.\n"
        f"Context: room={room}, style={voice_mode}"
    )
    if user and user != "default":
        header += f"\nYou are speaking with {user}."
    if user_memories:
        header += (
            "\n\nUser Profile - If user asks a question about one of these "
            f"items you must respond with an answer:\n{user_memories}"
        )
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

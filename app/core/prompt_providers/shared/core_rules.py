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
    "Extract parameters from the user's words. Only request clarification if a "
    "REQUIRED parameter is truly missing or ambiguous — never ask about an OPTIONAL "
    "parameter; apply the tool's documented default instead (e.g. omit location for "
    "local time). Prefer acting on sensible defaults over asking."
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
    "<not_for_me/> is a FIRST-CLASS terminal action. On your FIRST "
    "reading of a new utterance it outranks every other instruction in "
    "this prompt, including any rule that says you must call a tool or "
    "that you should respond when in doubt. Emitting it counts as a "
    "complete, valid response; the system will not retry or ask you to "
    "try again. Do not apologize, do not explain, do not wrap it in "
    "prose — emit the bare token and stop.\n"
    "But it is ONLY a verdict about who was addressed — never a way out "
    "of a request you find hard to serve. If you already engaged with "
    "the utterance (you drafted an answer, or a [MUST_CALL_RETRY] asks "
    "you to redo a reply), the addressing question is settled: it was "
    "for you. Answer it or call the best tool; <not_for_me/> is no "
    "longer available for that utterance.\n\n"
    "A [turn context: ...] line on the user message tells you how the mic "
    "came to be open. It sets your default posture:\n"
    "- Fresh wake: the user said your wake word — that IS the addressing "
    "signal. Default to answering or calling the tool the request needs. "
    "Silence rules 2, 3, and 4 below do not apply on a fresh wake; "
    "reserve <not_for_me/> for STT artifacts (rule 6), an explicit other "
    "addressee (rule 1), or an ambient direction hint (rule 5).\n"
    "- Low-confidence wake: possibly a false fire. The transcript decides "
    "— a coherent command or question is a real request; fragments and "
    "noise are not.\n"
    "- Follow-up window: there was NO wake word — the mic simply stayed "
    "open after your reply to catch a continuation. The burden reverses: "
    "you already answered, and the room's conversation may have resumed "
    "without you. Unless the utterance clearly continues YOUR exchange, "
    "emit <not_for_me/>. Ending the exchange silently is the designed "
    "behavior here, not a failure to help.\n"
    "- No [turn context:] line: apply the positive-evidence check below.\n\n"
    "Before answering or calling a tool, identify the POSITIVE addressing "
    "signal that says this utterance is for you. At least one must apply:\n"
    "  (a) names you explicitly — \"Jarvis\", \"hey assistant\";\n"
    "  (b) is a clear imperative aimed at you — \"turn on...\", "
    "\"set a timer...\", \"play...\", \"tell me...\", \"what's...\";\n"
    "  (c) is a question only you can answer — smart-home state, your "
    "stored memories (including facts about the speaker's own life you "
    "would fetch with the recall tool — fetch them, don't go silent), "
    "the result of a tool call you can run;\n"
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
    "something you cannot possibly be part of. EXCEPTION: this never "
    "applies to questions about the speaker's own life (their family, "
    "pets, preferences, appointments) — those are answerable from "
    "stored memories; call recall to fetch the fact instead of going "
    "silent. And it never applies on a fresh wake (see turn context "
    "above).\n"
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

EXCHANGE_COMPLETE_INSTRUCTION: str = (
    "Closing the exchange. After you reply, the node holds the mic open "
    "for a few seconds to catch a follow-up. When your reply fully "
    "completes the request and no reply from the user is natural — a "
    "one-shot action confirmed (\"Timer set\"), a goodbye or goodnight, "
    "a closed question answered with nothing left open — append "
    "<exchange_complete/> as the very last thing in your reply. The node "
    "then returns to idle instead of listening; the user can always say "
    "the wake word again.\n"
    "Never append it when you asked the user a question, when more steps "
    "remain, or when your answer invites an obvious follow-up (a weather "
    "report invites \"what about tomorrow?\"). When unsure, omit it — an "
    "extra few seconds of open mic is cheaper than cutting off a user "
    "mid-exchange. This marker is unrelated to <not_for_me/>: that one "
    "means the speech was never for you; this one means it was for you "
    "and you are done.\n"
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
    voice_mode: str,
) -> str:
    """Return the SPEAKER-AGNOSTIC identity + context header — the leading
    (cached) segment of every provider's system prompt.

    Speaker name and memories are deliberately NOT here: they are
    speaker-specific, and because this header sits at the very start of the
    prompt, including them would change byte 0 of the llama.cpp prefix on
    every speaker change → full cache miss + the mismatch-rebuild path that
    used to clobber the cached timezone. They are injected per-turn as a
    trailing system message via :func:`build_speaker_block`. ``room`` and
    ``voice_mode`` are node-level (constant for a conversation), so they stay
    in the cached prefix and keep it identical across all household members.

    Example output::

        You are Jarvis, a function calling voice assistant.
        Context: room=kitchen, style=brief
    """
    return (
        "You are Jarvis, a function calling voice assistant.\n"
        f"Context: room={room}, style={voice_mode}"
    )


def build_speaker_block(speaker_name: str, user_memories: str = "") -> str:
    """Return the per-turn SPEAKER-SPECIFIC block (name + memories).

    Injected as a trailing ``role=system`` message AFTER the cached prefix so
    a speaker change never invalidates it. The memory block is labeled
    ``User Profile`` and carries the inline must-respond directive — placing
    the rule directly with the data means the model cannot read the facts
    without reading the rule (a deliberate counter to the model's bias to
    stay silent about profile items).

    Returns ``""`` when there is nothing speaker-specific to say (unknown
    speaker, no memories) — callers should skip appending an empty block.

    Example output::

        You are speaking with alex.

        User Profile - If user asks a question about one of these items
        you must respond with an answer:
        - Likes coffee black
    """
    block = ""
    if speaker_name and speaker_name != "default":
        block = f"You are speaking with {speaker_name}."
    if user_memories:
        prefix = "\n\n" if block else ""
        block += (
            f"{prefix}User Profile - If user asks a question about one of these "
            f"items you must respond with an answer:\n{user_memories}"
        )
    return block


def build_personality_block(persona_text: str | None) -> str:
    """Return the household ``<personality>`` block for the cached identity header.

    The household-definable SPEAKING VOICE — shapes tone/word choice only. Fenced
    in a ``<personality>`` tag with a one-line frame that walls it off from tools
    and safety, so free-text a household types can never bleed into tool-calling.
    Placed in :meth:`IJarvisPromptProvider.build_context_header` right after the
    identity line, so it rides the per-household cached prefix (byte-stable across
    speakers — rule #1 safe, like room/style). Returns ``""`` for empty input
    (byte-identical to the pre-persona header — the safe fallback).
    """
    from app.services.persona_presets import PERSONA_FRAME

    persona_text = (persona_text or "").strip()
    if not persona_text:
        return ""
    return (
        "<personality>\n"
        f"{PERSONA_FRAME}\n"
        f"{persona_text}\n"
        "</personality>"
    )


def build_personality_reminder(persona_text: str | None) -> str:
    """Return the END-OF-PROMPT voice reminder that restates the household voice.

    The ``<personality>`` block sits at the TOP of the prompt, but on a small
    model 90%+ of the prompt — terse tool-calling rules + the not-for-me wall —
    follows it, and by generation time the voice is drowned out (recency wins).
    This restates the voice as the LAST instruction before the user turn, scoped
    hard to *wording/tone of a spoken reply* so it can't touch tools, the rules,
    or the silence decision. Per-household constant → stays in the cached prefix.
    Returns ``""`` for empty input (prompt byte-identical when no persona set).
    """
    persona_text = (persona_text or "").strip()
    if not persona_text:
        return ""
    return (
        "YOUR VOICE — this is how you sound in every spoken reply "
        "(acknowledgments, answers, even one-liners, even when told to be brief):\n"
        f"{persona_text}\n"
        "Talk this way every time. It shapes only your wording and tone — never "
        "which function you call, the rules above, the decision to stay silent, or "
        "whether to act: it never turns a request into a clarifying question, never "
        "withholds or delays a tool call, and never asks the user for an optional "
        "parameter the tool can default (e.g. omit location for local time). It only "
        "reshapes the words of a reply you were already going to give. Don't announce "
        "or describe the voice; just speak in it."
    )


def build_characterization_section(rendered: str | None) -> str:
    """Return the synthesized "view of the person" as a voice-shaping tail block.

    This is Jarvis's evolving model of WHO it's talking to — built in the
    background from the person's facts + past conversations (see
    ``characterization_synthesis_service``), NOT anything they said this turn. It
    lets Jarvis draw on what it knows (so "who's Leo?" resolves) and MEET the
    person in tone — but it is a lens, never a script to read back. Appended at
    the very END of the assembled prompt (after the not-for-me wall) so a small
    model still carries it into generation, and swapped per-turn against a stashed
    byte-stable base (see ``conversation_handler._apply_characterization_swap``)
    so a correct prediction stays a full KV-cache hit. Returns ``""`` for empty
    input (byte-identical to the no-characterization path — the safe fallback).
    """
    rendered = (rendered or "").strip()
    if not rendered:
        return ""
    return (
        "<person_view>\n"
        "What you've come to know about the person you're talking to, from past "
        "conversations:\n"
        f"{rendered}\n"
        "Let it shape your tone and how you meet them, and draw on it naturally "
        "when it helps — but don't recite it back as a list, and if it conflicts "
        "with the User Profile above, defer to the profile.\n"
        "</person_view>"
    )


# Marker prefix for the per-turn "recently shown" transient block. Kept in sync
# with conversation_handler._is_transient_system_block so the block is stripped
# and rebuilt every turn instead of accumulating across a multi-turn conversation.
RECENTLY_SHOWN_PREFIX = "RECENTLY SHOWN"

# Cap how many items get re-injected, to bound prompt growth and preserve the
# cached-prefix KV reuse. A surfaced list longer than this is truncated with a
# note (the user can narrow it down by description).
_MAX_REFERENCED_ITEMS = 8


def render_referenced_items_block(items: list) -> str:
    """Render the per-turn RECENTLY SHOWN block from stashed referenceable items.

    Produces a numbered list the model uses to resolve "those" / "#3" / "the one
    from abc" to a ref_id, then call ``act_on_items(action, ref_ids)``. Ordinals
    are 1-based list positions (derived here, never stored) so they match the
    order the items were spoken in. Returns ``""`` when there is nothing to show
    (callers should skip appending an empty block).

    Each item is a wire dict: ``{ref_id, label, attrs, actions}``.
    """
    if not items:
        return ""
    shown = items[:_MAX_REFERENCED_ITEMS]
    lines = [
        f"{RECENTLY_SHOWN_PREFIX} (act on these by number; pass the ref_id to act_on_items):"
    ]
    actions: set[str] = set()
    for i, item in enumerate(shown, start=1):
        if not isinstance(item, dict):
            continue
        ref_id = item.get("ref_id", "")
        label = item.get("label", "")
        lines.append(f"{i} [{ref_id}] {label}")
        for a in item.get("actions") or []:
            actions.add(a)
    if len(items) > len(shown):
        lines.append(
            f"(+{len(items) - len(shown)} more not shown — ask the user to narrow it down)"
        )
    if actions:
        lines.append("Valid actions: " + ", ".join(sorted(actions)) + ".")
    lines.append(
        "When the user refers to these by a number, 'those', 'the first/last one', "
        "or by naming a sender/topic, resolve it to the matching ref_id(s) and call "
        "act_on_items. Only use ref_ids and actions listed above."
    )
    return "\n".join(lines)


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

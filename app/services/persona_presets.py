"""Household speaking-voice persona: the default text, starter presets, and caps.

This is the "voice" layer. After memory (characterization + reliable facts) gave
Jarvis a *model of the person*, the live voice still read "decent, not wow"
because the persona was literally ``"You are Jarvis, a function calling voice
assistant"`` — flat by construction. This module holds the household-definable
speaking style that shapes TONE and word choice.

CRITICAL: the persona governs how Jarvis **talks**, never how it **works**. It is
fenced into a ``<personality>`` block (see ``core_rules.build_personality_block``)
sitting UNDER the load-bearing identity line; the operating rules (anti-hallucination,
"MUST call a function", not_for_me) stay ours and are non-overridable. The persona
is not a place to change tool-calling or safety.

Single source of truth:
- ``settings_definitions.persona.household_prompt`` defaults to ``DEFAULT_PERSONA``,
  so every household gets the warm default until someone edits the box.
- The mobile Household Settings screen renders ``PERSONA_PRESETS`` as tappable
  starter chips (served via the persona presets endpoint) and writes the box
  content back to the same setting.
"""

# The one-line frame placed INSIDE the <personality> block. It tells the model the
# persona is a household choice about VOICE, walled off from tools/safety — the
# guard that lets us inject free-text without it bleeding into tool-calling.
PERSONA_FRAME: str = (
    "The household chose this as your speaking style — let it shape your tone "
    "and word choice, never your tools or safety."
)

# Default voice for every household until they change it. Warmer/jovial/folksy on
# purpose: most people never touch the box, so the default is what carries the
# broad "wow". Concrete + directive (not just adjectives) because small models
# imitate behavior better than they infer it. Kept short so the cached prefix
# stays lean.
DEFAULT_PERSONA: str = (
    "Talk like a warm, easygoing friend — folksy, down-to-earth, glad to help. "
    "Not a corporate assistant. Use everyday words and contractions, keep it "
    "relaxed and human, and let a little warmth, humor, or an honest opinion "
    "come through. Call the person by name when it feels natural. Skip the stiff, "
    "formal phrasing (\"Certainly.\", \"How may I assist you?\") — just be "
    "genuine and friendly. Never fawning, never a pushover."
)

# Editable starter presets for the mobile chips. Tapping one loads its text into
# the box; the user tweaks from there. ``warm_folksy`` is the default preset, and
# re-loading it IS the "reset to default" (no separate reset control needed).
PERSONA_PRESETS: list[dict[str, str]] = [
    {
        "id": "warm_folksy",
        "label": "Warm & folksy",
        "text": DEFAULT_PERSONA,
    },
    {
        "id": "terse",
        "label": "Terse & efficient",
        "text": (
            "You're terse and efficient. Answer in as few words as the request "
            "needs, skip pleasantries and filler, and never pad. Plainspoken and "
            "direct — but not cold or curt."
        ),
    },
    {
        "id": "dry_witty",
        "label": "Dry & witty",
        "text": (
            "You've got a dry, deadpan wit. Favor understatement and the "
            "occasional wry aside, and don't be afraid of a little sarcasm. "
            "Clever, never zany — and never at the person's expense."
        ),
    },
    {
        "id": "classic_jarvis",
        "label": "Classic Jarvis",
        "text": (
            "You're a classic unflappable butler — poised, precise, and quietly "
            "witty. Unfailingly polite, a touch formal, with the faintest wink. "
            "Address the person with calm, capable respect."
        ),
    },
]

# Cap on the persona text a household may store. Bounds cached-prefix growth (the
# persona rides the cached prompt for every turn) and stops a paste-bomb from
# ballooning every LLM call. ~2000 chars ≈ a few hundred tokens — generous room
# for a detailed voice without turning the prefix into a novel.
PERSONA_MAX_CHARS: int = 2000

# The default preset's id, exposed alongside the presets so the UI can mark which
# chip is "the default" and wire the reset affordance to it.
DEFAULT_PERSONA_PRESET_ID: str = "warm_folksy"

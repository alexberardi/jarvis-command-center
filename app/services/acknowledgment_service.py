"""
Fast acknowledgments for user requests.

Returns a brief, conversational acknowledgment instantly using keyword
matching + randomized templates. Does NOT call the LLM — the acknowledgment
must be zero-latency to avoid competing with the main pipeline for
llama.cpp's single inference slot or evicting its prefix cache.
"""

import random
import re

# Keyword → pool of contextual acknowledgments
_KEYWORD_POOLS: dict[str, list[str]] = {
    "weather|forecast|temperature|rain|umbrella": [
        "Checking the forecast.",
        "Let me check the weather.",
        "One moment, pulling up the forecast.",
    ],
    "timer|alarm|remind": [
        "Setting that up.",
        "On it.",
        "Got it.",
    ],
    "light|lamp|switch|plug|turn on|turn off": [
        "On it.",
        "Sure thing.",
        "Right away.",
    ],
    "play|pause|stop|music|song|volume|speaker": [
        "Sure.",
        "On it.",
        "You got it.",
    ],
    "search|look up|find|who|what|when|where|why|how": [
        "Let me look into that.",
        "Good question, give me a moment.",
        "Looking into it.",
        "Let me find out.",
    ],
    "recipe|cook|ingredient|meal": [
        "Let me pull that up.",
        "Checking on that.",
        "One moment.",
    ],
    "news|headline|happening": [
        "Let me check.",
        "Pulling up the latest.",
        "One moment.",
    ],
    "score|game|sport": [
        "Let me check the scores.",
        "Checking on that.",
        "One moment.",
    ],
}

# Fallback pool for unmatched queries
_GENERIC_POOL: list[str] = [
    "Let me look into that.",
    "Working on it.",
    "One moment.",
    "Give me a second.",
    "On it.",
    "Let me check.",
]

# Compiled patterns for fast matching
_COMPILED_POOLS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(pattern, re.IGNORECASE), pool)
    for pattern, pool in _KEYWORD_POOLS.items()
]


def generate_acknowledgment(user_text: str) -> str:
    """Generate a brief, contextual acknowledgment for a user request.

    Uses keyword matching against the user's text to pick a relevant
    acknowledgment from curated pools. Falls back to a generic pool
    for unmatched queries. Instant — no LLM call.

    Args:
        user_text: The user's original message/command.

    Returns:
        Acknowledgment text.
    """
    for pattern, pool in _COMPILED_POOLS:
        if pattern.search(user_text):
            return random.choice(pool)
    return random.choice(_GENERIC_POOL)

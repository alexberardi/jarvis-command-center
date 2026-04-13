"""
Fast LLM-generated acknowledgments for user requests.

Generates a brief, conversational acknowledgment (~1s on a warm model)
that can be streamed/spoken immediately while the main pipeline processes.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.llm_proxy_client import LLMProxyClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are Jarvis, a voice assistant. Generate ONLY a brief, natural "
    "acknowledgment (max 10 words) for the user's request. Be conversational "
    "and varied. Do not answer the question — just acknowledge it.\n"
    "Examples: 'Let me look into that.', 'On it.', "
    "'Checking the forecast now.', 'Good question, give me a moment.'"
)


async def generate_acknowledgment(
    llm_client: "LLMProxyClient",
    user_text: str,
) -> str:
    """Generate a brief, natural acknowledgment for a user request.

    Uses a lightweight LLM call with max_tokens=30 and no tools — typically
    returns in ~1s on a warm model.

    Args:
        llm_client: The LLM proxy client to use for generation.
        user_text: The user's original message/command.

    Returns:
        Acknowledgment text, or empty string on failure.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    try:
        result = await llm_client.chat_completion(
            messages=messages,
            max_tokens=30,
            temperature=0.7,
        )
        content: str = result["choices"][0]["message"]["content"]
        return content.strip().strip('"')
    except Exception as e:
        logger.debug("Acknowledgment generation failed (non-fatal): %s", e)
        return ""

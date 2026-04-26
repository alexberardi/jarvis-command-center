"""Wake-response text generation.

Generates the short greeting the node plays right after wake-word detection
("Glad to help!", "At your service", etc.). Previously lived in jarvis-tts
— moved here in the v0.1.10 refactor so TTS and llm-proxy can stay dumb
(text → audio / chat → chat) while the provider-aware sanitation happens
at the only place that already knows which model is serving.

Flow:
  node → POST /api/v0/wake-response
       → CC builds messages, appends provider.user_message_suffix
         (/no_think on Qwen3) to the user turn
       → CC calls llm-proxy via the existing LLM client
       → CC runs provider.sanitize_text on the result
         (strips <think>...</think> on Qwen3; identity elsewhere)
       → CC returns {"text": "..."} to the node
       → node calls jarvis-tts /speak/stream with the clean text
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.llm_proxy_client import LLMProxyClient
from app.deps import get_model_service, verify_api_key

router = APIRouter(prefix="/wake-response", tags=["wake-response"])
logger = logging.getLogger("uvicorn")


_WAKE_RESPONSE_FALLBACK = "Yes?"

_WAKE_SYSTEM_PROMPT = (
    "You are Jarvis, a voice-assistant butler. The user has just called your "
    "name. Respond with ONE short, charming, varied greeting — STRICTLY 1 to "
    "3 words ONLY (HARD CAP at 3 words), gender neutral. "
    "CRITICAL: speed matters more than wit — every extra word delays the "
    "user. You are generating many of these throughout the day; pick "
    "something DIFFERENT every time, but never go over 3 words. "
    "Reply with JUST the greeting, no labels, no quotes, no XML, no "
    "trailing explanation.\n\n"
    "Style examples (riff, don't copy verbatim):\n"
    "- Yes?\n"
    "- Mhm?\n"
    "- Listening.\n"
    "- Go ahead.\n"
    "- What's up?\n"
    "- Hit me.\n"
    "- Shoot.\n"
    "- Whatcha got?\n"
    "- Right here.\n"
    "- You rang?\n"
    "- Speak.\n"
    "- Tell me.\n"
    "- Sure thing?\n"
    "- I'm here.\n"
    "- Sup?\n"
    "- Ready.\n"
)


class WakeResponseReply(BaseModel):
    text: str


@router.post("", response_model=WakeResponseReply)
async def generate_wake_response(
    _node=Depends(verify_api_key),  # Node auth — same style as other node endpoints
    model_service=Depends(get_model_service),
) -> WakeResponseReply:
    """Return a short wake-greeting string for the node to TTS.

    Always returns 200 with a usable text (falling back to "Yes?" on any
    failure) so the node's wake path never hangs on an upstream glitch.
    """
    prompt_provider = getattr(model_service, "prompt_provider", None)
    user_suffix: str = prompt_provider.user_message_suffix if prompt_provider else ""
    user_message: str = "Hello Jarvis"
    if user_suffix:
        user_message = f"{user_message}\n{user_suffix}"

    try:
        client = LLMProxyClient()
        result = await client.chat_completion(
            messages=[
                {"role": "system", "content": _WAKE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=1.1,  # High entropy — butler greetings should feel fresh, not template-matched
            max_tokens=12,  # 1-3 word constraint; cuts cached WAV ~5x vs the original 12-word prompt
            include_date_context=False,
        )
    except Exception as exc:
        logger.warning(
            "Wake-response LLM call failed: type=%s repr=%r",
            type(exc).__name__, exc, exc_info=True,
        )
        return WakeResponseReply(text=_WAKE_RESPONSE_FALLBACK)

    # Extract content from the standard OpenAI-ish response shape.
    content: str = ""
    choices = result.get("choices") if isinstance(result, dict) else None
    if choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            content = str(msg.get("content") or "")

    if not content:
        logger.warning("Wake-response LLM returned no content — using fallback")
        return WakeResponseReply(text=_WAKE_RESPONSE_FALLBACK)

    # Provider-aware sanitation. Strips Qwen3 <think> blocks + any other
    # model-specific artifacts the provider knows about. No-op for
    # providers that don't override sanitize_text.
    if prompt_provider:
        content = prompt_provider.sanitize_text(content)

    # Universal TTS-safe scrub — emojis specifically, since the wake
    # greeting is going straight to the speech engine.
    from app.core.tts_text import clean_for_tts
    content = clean_for_tts(content)

    text = content.strip()
    if not text:
        return WakeResponseReply(text=_WAKE_RESPONSE_FALLBACK)

    return WakeResponseReply(text=text)

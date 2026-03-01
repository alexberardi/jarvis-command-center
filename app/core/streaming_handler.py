"""Streaming voice response utilities.

Provides sentence splitting and text-to-audio streaming for the unified
voice command endpoint. The LLM processing is handled upstream by
process_voice_command_with_tools(); this module's only job is converting
a completed text response into streamed PCM audio via TTS.
"""

import logging
import re
from typing import AsyncIterator

from app.core.clients.tts_client import TTSClient

logger = logging.getLogger("uvicorn")


def extract_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Returns:
        List of sentence strings (may be a single element if no split points).
    """
    # Split on sentence-ending punctuation followed by whitespace
    pattern = r"(?<=[.!?])\s+"
    parts = re.split(pattern, text)
    return [s.strip() for s in parts if s.strip()]


async def stream_text_as_audio(
    text: str,
    tts_client: TTSClient,
) -> AsyncIterator[bytes]:
    """Split text into sentences, stream each through TTS as PCM audio.

    Args:
        text: The full text to synthesize.
        tts_client: An initialized TTSClient instance.

    Yields:
        Raw PCM audio bytes (format determined by TTS service).
    """
    sentences = extract_sentences(text)
    if not sentences:
        return

    for sentence in sentences:
        try:
            audio_iter, _audio_meta = await tts_client.speak_stream(sentence)
            async for chunk in audio_iter:
                yield chunk
        except Exception as e:
            logger.error("TTS streaming error for sentence '%s': %s", sentence[:50], e)
            # Skip this sentence and continue with the next

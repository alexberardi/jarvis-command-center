"""Streaming voice response utilities.

Provides text chunking and text-to-audio streaming for the unified
voice command endpoint. The LLM processing is handled upstream by
process_voice_command_with_tools(); this module's only job is converting
a completed text response into streamed PCM audio via TTS.

For short responses (≤ 2 sentences), each sentence is streamed individually
for lowest latency. For longer responses (briefings, stories), sentences
are grouped into paragraph-sized chunks and the next chunk is prefetched
while the current one plays, eliminating inter-chunk gaps.
"""

import asyncio
import logging
import re
from typing import AsyncIterator

from app.core.clients.tts_client import TTSClient

logger = logging.getLogger("uvicorn")

# Sentences per chunk for long-form content.
# Balances latency (smaller = faster first audio) against gap risk
# (larger = fewer HTTP round-trips). 3-4 sentences ≈ 5-10 seconds of audio,
# which gives plenty of time to prefetch the next chunk.
_SENTENCES_PER_CHUNK = 4

# Timeout for long-form TTS requests (paragraph chunks are larger than
# single sentences, so give Piper more time).
_LONG_FORM_TIMEOUT = 60.0


def extract_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Returns:
        List of sentence strings (may be a single element if no split points).
    """
    # Split on sentence-ending punctuation followed by whitespace
    pattern = r"(?<=[.!?])\s+"
    parts = re.split(pattern, text)
    return [s.strip() for s in parts if s.strip()]


def _group_into_chunks(sentences: list[str], chunk_size: int) -> list[str]:
    """Group sentences into paragraph-sized chunks.

    Args:
        sentences: List of individual sentences.
        chunk_size: Number of sentences per chunk.

    Returns:
        List of chunk strings (each is multiple sentences joined by space).
    """
    chunks: list[str] = []
    for i in range(0, len(sentences), chunk_size):
        group = sentences[i : i + chunk_size]
        chunks.append(" ".join(group))
    return chunks


async def _collect_chunk_audio(
    chunk_text: str,
    tts_client: TTSClient,
    timeout: float,
) -> bytes:
    """Synthesize a chunk and collect all PCM bytes into a buffer.

    This runs as a prefetch task — the result is buffered so it can be
    yielded without any inter-chunk delay.
    """
    audio_iter, _meta = await tts_client.speak_stream(chunk_text, timeout=timeout)
    parts: list[bytes] = []
    async for piece in audio_iter:
        parts.append(piece)
    return b"".join(parts)


async def stream_text_as_audio(
    text: str,
    tts_client: TTSClient,
) -> AsyncIterator[bytes]:
    """Convert text to streamed PCM audio via TTS.

    Strategy:
    - Short text (≤ 2 sentences): stream each sentence directly for lowest
      first-byte latency (same as before).
    - Long text (> 2 sentences): group into paragraph chunks and prefetch
      the next chunk while the current one plays, eliminating audible gaps.

    Args:
        text: The full text to synthesize.
        tts_client: An initialized TTSClient instance.

    Yields:
        Raw PCM audio bytes (format determined by TTS service).
    """
    sentences = extract_sentences(text)
    if not sentences:
        return

    # Short responses: stream sentence-by-sentence (low latency, minimal gap)
    if len(sentences) <= 2:
        for sentence in sentences:
            try:
                audio_iter, _audio_meta = await tts_client.speak_stream(sentence)
                async for chunk in audio_iter:
                    yield chunk
            except Exception as e:
                logger.error("TTS streaming error for sentence '%s': %s", sentence[:50], e)
        return

    # Long responses: paragraph chunking with prefetch
    chunks = _group_into_chunks(sentences, _SENTENCES_PER_CHUNK)
    logger.info(
        "Long-form TTS: %d sentences → %d chunks (%d sentences/chunk)",
        len(sentences),
        len(chunks),
        _SENTENCES_PER_CHUNK,
    )

    # Stream the first chunk directly (no prefetch delay — fastest first byte)
    prefetch_task: asyncio.Task[bytes] | None = None

    try:
        for i, chunk_text in enumerate(chunks):
            is_first = i == 0
            is_last = i == len(chunks) - 1

            if is_first:
                # Stream first chunk directly for lowest latency
                try:
                    audio_iter, _meta = await tts_client.speak_stream(
                        chunk_text, timeout=_LONG_FORM_TIMEOUT,
                    )

                    # While streaming chunk 0, start prefetching chunk 1
                    if not is_last:
                        prefetch_task = asyncio.create_task(
                            _collect_chunk_audio(
                                chunks[i + 1], tts_client, _LONG_FORM_TIMEOUT,
                            )
                        )

                    async for audio_bytes in audio_iter:
                        yield audio_bytes
                except Exception as e:
                    logger.error("TTS error on first chunk: %s", e)
                    return
            else:
                # Subsequent chunks: use prefetched buffer (no gap)
                if prefetch_task is not None:
                    try:
                        buffered_audio = await prefetch_task
                        prefetch_task = None

                        # Start prefetching the NEXT chunk while yielding this one
                        if not is_last:
                            prefetch_task = asyncio.create_task(
                                _collect_chunk_audio(
                                    chunks[i + 1], tts_client, _LONG_FORM_TIMEOUT,
                                )
                            )

                        # Yield buffered audio in playback-sized pieces
                        offset = 0
                        while offset < len(buffered_audio):
                            yield buffered_audio[offset : offset + 4096]
                            offset += 4096
                    except Exception as e:
                        logger.error(
                            "TTS prefetch error for chunk %d: %s", i, e,
                        )
                else:
                    # Fallback: no prefetch available, stream directly
                    try:
                        audio_iter, _meta = await tts_client.speak_stream(
                            chunk_text, timeout=_LONG_FORM_TIMEOUT,
                        )
                        if not is_last:
                            prefetch_task = asyncio.create_task(
                                _collect_chunk_audio(
                                    chunks[i + 1], tts_client, _LONG_FORM_TIMEOUT,
                                )
                            )
                        async for audio_bytes in audio_iter:
                            yield audio_bytes
                    except Exception as e:
                        logger.error("TTS streaming error for chunk %d: %s", i, e)
    finally:
        # Clean up any in-flight prefetch
        if prefetch_task is not None and not prefetch_task.done():
            prefetch_task.cancel()
            try:
                await prefetch_task
            except (asyncio.CancelledError, Exception):
                pass

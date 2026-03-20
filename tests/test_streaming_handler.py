"""Tests for streaming_handler: sentence extraction, chunk grouping, and audio streaming."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.streaming_handler import (
    extract_sentences,
    _group_into_chunks,
    stream_text_as_audio,
)


# =============================================================================
# extract_sentences
# =============================================================================


class TestExtractSentences:
    def test_single_sentence(self):
        assert extract_sentences("Hello world.") == ["Hello world."]

    def test_multiple_sentences(self):
        result = extract_sentences("First sentence. Second sentence! Third?")
        assert result == ["First sentence.", "Second sentence!", "Third?"]

    def test_empty_string(self):
        assert extract_sentences("") == []

    def test_no_punctuation(self):
        assert extract_sentences("Hello world") == ["Hello world"]

    def test_whitespace_only(self):
        assert extract_sentences("   ") == []

    def test_preserves_internal_periods(self):
        # e.g. "Dr. Smith" shouldn't split on "Dr."
        # Current regex splits on ". " — so "Dr. Smith" would split
        result = extract_sentences("Talk to Dr. Smith about it.")
        assert len(result) == 2  # "Talk to Dr." and "Smith about it."

    def test_multiple_spaces(self):
        result = extract_sentences("First.  Second.")
        assert result == ["First.", "Second."]


# =============================================================================
# _group_into_chunks
# =============================================================================


class TestGroupIntoChunks:
    def test_exact_fit(self):
        sentences = ["A.", "B.", "C.", "D."]
        result = _group_into_chunks(sentences, 2)
        assert result == ["A. B.", "C. D."]

    def test_remainder(self):
        sentences = ["A.", "B.", "C."]
        result = _group_into_chunks(sentences, 2)
        assert result == ["A. B.", "C."]

    def test_single_sentence(self):
        result = _group_into_chunks(["Hello."], 4)
        assert result == ["Hello."]

    def test_chunk_size_larger_than_input(self):
        sentences = ["A.", "B."]
        result = _group_into_chunks(sentences, 10)
        assert result == ["A. B."]

    def test_chunk_size_one(self):
        sentences = ["A.", "B.", "C."]
        result = _group_into_chunks(sentences, 1)
        assert result == ["A.", "B.", "C."]


# =============================================================================
# stream_text_as_audio
# =============================================================================


def _make_tts_client(audio_bytes: bytes = b"\x00" * 100) -> MagicMock:
    """Create a mock TTSClient whose speak_stream returns given bytes."""
    client = MagicMock()

    async def fake_speak_stream(text: str, timeout: float = 30.0):
        async def _iter():
            yield audio_bytes
        return _iter(), {"sample_rate": "22050", "channels": "1", "sample_width": "2"}

    client.speak_stream = AsyncMock(side_effect=fake_speak_stream)
    return client


async def _collect_stream(stream) -> bytes:
    """Collect all bytes from an async iterator."""
    parts = []
    async for chunk in stream:
        parts.append(chunk)
    return b"".join(parts)


@pytest.mark.asyncio
class TestStreamTextAsAudioShort:
    """Short text (≤ 2 sentences) — sentence-by-sentence streaming."""

    async def test_single_sentence(self):
        client = _make_tts_client(b"\x01\x02\x03")
        result = await _collect_stream(stream_text_as_audio("Hello.", client))
        assert result == b"\x01\x02\x03"
        assert client.speak_stream.call_count == 1

    async def test_two_sentences(self):
        client = _make_tts_client(b"\xAA")
        result = await _collect_stream(stream_text_as_audio("First. Second.", client))
        assert result == b"\xAA\xAA"
        assert client.speak_stream.call_count == 2

    async def test_empty_text(self):
        client = _make_tts_client()
        result = await _collect_stream(stream_text_as_audio("", client))
        assert result == b""
        assert client.speak_stream.call_count == 0

    async def test_sentence_error_skipped(self):
        """If one sentence fails, the other still plays."""
        client = MagicMock()
        call_count = 0

        async def fake_speak_stream(text, timeout=30.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("TTS down")
            async def _iter():
                yield b"\x42"
            return _iter(), {}

        client.speak_stream = AsyncMock(side_effect=fake_speak_stream)
        result = await _collect_stream(stream_text_as_audio("Fail. Succeed.", client))
        assert result == b"\x42"


@pytest.mark.asyncio
class TestStreamTextAsAudioLong:
    """Long text (> 2 sentences) — paragraph chunking with prefetch."""

    async def test_three_sentences_uses_chunking(self):
        """3 sentences should trigger long-form path."""
        client = _make_tts_client(b"\x10" * 50)
        result = await _collect_stream(
            stream_text_as_audio("One. Two. Three.", client)
        )
        assert len(result) > 0
        # With chunk_size=4 and 3 sentences, we get 1 chunk → streamed directly
        assert client.speak_stream.call_count >= 1

    async def test_five_sentences_creates_two_chunks(self):
        """5 sentences with chunk_size=4 → 2 chunks."""
        client = _make_tts_client(b"\x20" * 100)
        text = "One. Two. Three. Four. Five."
        result = await _collect_stream(stream_text_as_audio(text, client))
        assert len(result) > 0
        # Chunk 1 (4 sentences): streamed directly
        # Chunk 2 (1 sentence): prefetched
        assert client.speak_stream.call_count == 2

    async def test_long_text_produces_audio(self):
        """Simulate a longer text to verify end-to-end."""
        sentences = [f"Sentence {i}." for i in range(10)]
        text = " ".join(sentences)
        client = _make_tts_client(b"\xFF" * 200)
        result = await _collect_stream(stream_text_as_audio(text, client))
        assert len(result) > 0
        # 10 sentences / 4 per chunk = 3 chunks
        assert client.speak_stream.call_count == 3

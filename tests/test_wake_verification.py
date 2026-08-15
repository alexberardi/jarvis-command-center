"""Wake-clip verification — slicing, fuzzy matching, verdict resolution.

The feature exists because openWakeWord misfires at high confidence look
acoustically identical to deliberate wakes (prod 2026-08-15: two 0.95-score
quiet-room misfires; one marked a medication as taken off overheard family
talk). The clip that fired the wake is the ground truth, and it was already
being shipped to CC as the leading seconds of speaker_audio.
"""

import asyncio
import io
import wave

import pytest

from app.core import wake_verification as wv
from app.core.conversation_cache import conversation_cache


def _make_wav(seconds: float, rate: int = 16000) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return out.getvalue()


class TestSliceLeadingWav:
    def test_slices_to_requested_length(self):
        sliced = wv.slice_leading_wav(_make_wav(5.0), seconds=2.2)
        with wave.open(io.BytesIO(sliced), "rb") as w:
            assert w.getnframes() == pytest.approx(int(16000 * 2.2))
            assert w.getframerate() == 16000

    def test_short_clip_returned_whole(self):
        sliced = wv.slice_leading_wav(_make_wav(1.0), seconds=2.2)
        with wave.open(io.BytesIO(sliced), "rb") as w:
            assert w.getnframes() == 16000

    def test_garbage_bytes_fail_open_as_none(self):
        assert wv.slice_leading_wav(b"not a wav at all") is None


class TestWakePhrasePresent:
    def test_exact_match(self):
        assert wv.wake_phrase_present("hey jarvis what time is it") is True

    def test_whisper_manglings_match(self):
        # Whisper routinely mangles proper nouns on 2s clips.
        assert wv.wake_phrase_present("hey travis") is True
        assert wv.wake_phrase_present("hey jervis!") is True
        assert wv.wake_phrase_present("Jarvis.") is True

    def test_ambient_speech_does_not_match(self):
        # The literal prod misfires.
        assert wv.wake_phrase_present("Leo took his medicine.") is False
        assert wv.wake_phrase_present("Oh.") is False
        assert wv.wake_phrase_present("") is False
        assert wv.wake_phrase_present(None) is False

    def test_unrelated_similar_length_words_do_not_match(self):
        assert wv.wake_phrase_present("the harvest is ready") is False

    def test_blank_phrase_verifies_everything(self):
        # No phrase configured → nothing to verify against → fail open.
        assert wv.wake_phrase_present("anything", phrase="") is True


class TestResolveWakeVerification:
    def setup_method(self):
        conversation_cache.set(
            "conv-wv", [{"role": "system", "content": "x"}], [], "UTC", [], {}
        )

    def teardown_method(self):
        conversation_cache.remove("conv-wv")

    def _resolve(self, turn_context, mode="bias"):
        async def _run():
            return await wv.resolve_wake_verification("conv-wv", turn_context)
        import unittest.mock as m
        with m.patch.object(wv, "_get_mode", return_value=mode):
            return asyncio.run(_run())

    def test_non_wake_turn_returns_none(self):
        assert self._resolve({"source": "follow_up"}) is None
        assert self._resolve({"source": "chat"}) is None
        assert self._resolve(None) is None

    def test_mode_off_returns_none(self):
        conversation_cache.set_wake_verification(
            "conv-wv", {"verified": False, "transcript": "x"}
        )
        assert self._resolve({"source": "wake"}, mode="off") is None

    def test_resolved_verdict_carries_mode(self):
        conversation_cache.set_wake_verification(
            "conv-wv", {"verified": False, "transcript": "leo took his medicine"}
        )
        verdict = self._resolve({"source": "wake"}, mode="enforce")
        assert verdict["verified"] is False
        assert verdict["mode"] == "enforce"

    def test_pending_verdict_times_out_open(self, monkeypatch):
        monkeypatch.setattr(wv, "VERDICT_WAIT_SECONDS", 0.2)
        assert self._resolve({"source": "wake"}) is None

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


class TestGetModeFailSafe:
    """_get_mode fails SAFE to off on BOTH failure classes: unreachable
    settings service and unrecognized stored values. A typo'd mode must
    disable the feature, never silently activate bias."""

    def _mode_with_setting(self, raw):
        import unittest.mock as m

        svc = m.MagicMock()
        svc.get.return_value = raw
        with m.patch(
            "app.services.settings_service.get_settings_service", return_value=svc
        ):
            return wv._get_mode("hh-1", "node-1")

    def test_recognized_values_pass_through(self):
        assert self._mode_with_setting("bias") == "bias"
        assert self._mode_with_setting("enforce") == "enforce"
        assert self._mode_with_setting("off") == "off"
        assert self._mode_with_setting("  Bias ") == "bias"

    def test_unrecognized_value_is_off(self):
        # Was "bias" before 2026-08-15 — a typo silently enabled the feature.
        assert self._mode_with_setting("banana") == "off"
        assert self._mode_with_setting("biased") == "off"

    def test_missing_row_is_off(self):
        assert self._mode_with_setting(None) == "off"
        assert self._mode_with_setting("") == "off"

    def test_settings_service_error_is_off(self):
        import unittest.mock as m

        with m.patch(
            "app.services.settings_service.get_settings_service",
            side_effect=RuntimeError("db down"),
        ):
            assert wv._get_mode() == "off"

    def test_setting_definition_default_is_off(self):
        from app.services.settings_definitions import SETTINGS_DEFINITIONS

        definition = next(
            d for d in SETTINGS_DEFINITIONS
            if d.key == "voice.wake_verification_mode"
        )
        assert definition.default == "off"


class TestWakePhraseSimilarity:
    def test_exact_and_substring_are_full_similarity(self):
        assert wv.wake_phrase_similarity("hey jarvis") == 1.0
        assert wv.wake_phrase_similarity("jarvis, lights") == 1.0

    def test_whisper_manglings_score_high(self):
        assert wv.wake_phrase_similarity("hey travis") >= wv.CLIP_SIMILARITY_FLOOR
        assert wv.wake_phrase_similarity("jervis") >= wv.CLIP_SIMILARITY_FLOOR

    def test_partially_wake_shaped_scores_above_floor(self):
        # "harvest" shares 'arv...s' with 'jarvis' — a trace, not zero.
        assert wv.wake_phrase_similarity("the harvest is ready") >= wv.CLIP_SIMILARITY_FLOOR

    def test_garbage_clips_score_below_floor(self):
        # The capture-bug signature: nothing remotely wake-shaped.
        assert wv.wake_phrase_similarity("Turn on the living room lights.") < wv.CLIP_SIMILARITY_FLOOR
        assert wv.wake_phrase_similarity("Eat it.") < wv.CLIP_SIMILARITY_FLOOR
        assert wv.wake_phrase_similarity("") == 0.0
        assert wv.wake_phrase_similarity(None) == 0.0

    def test_blank_phrase_is_full_similarity(self):
        assert wv.wake_phrase_similarity("anything", phrase="") == 1.0


class TestClipPlausibilityGate:
    """A 0.9+ OWW fire whose verify clip has ZERO fuzzy trace of the wake
    phrase is the capture-bug signature, not the misfire signature (prod
    2026-08-15: two real lights commands suppressed by garbage clips).
    Verdict becomes clip_unreliable and downstream treats it as no-verdict
    (fail open)."""

    def setup_method(self):
        conversation_cache.set(
            "conv-cpg", [{"role": "system", "content": "x"}], [], "UTC", [], {}
        )

    def teardown_method(self):
        conversation_cache.remove("conv-cpg")

    def _resolve(self, turn_context, mode="bias"):
        async def _run():
            return await wv.resolve_wake_verification("conv-cpg", turn_context)
        import unittest.mock as m
        with m.patch.object(wv, "_get_mode", return_value=mode):
            return asyncio.run(_run())

    def _store(self, transcript, similarity=None, **extra):
        verdict = {
            "verified": False,
            "verdict": "unverified",
            "transcript": transcript,
            "phrase": "jarvis",
            "node_id": "node-1",
            **extra,
        }
        if similarity is not None:
            verdict["similarity"] = similarity
        conversation_cache.set_wake_verification("conv-cpg", verdict)

    def test_high_confidence_no_playback_stays_unverified(self):
        """RETIRED 2026-08-26 — a confident fire no longer rescues a clip.

        The rule read a 0.9+ fire with a no-trace clip as a CAPTURE bug:
        "the detector heard something extremely wake-like, so a clip with
        zero trace of it points at the clip pipeline". Node telemetry
        disproves that. Across 128 consecutive fires the clip came from
        the consumed-chunks deque 128/128 times — it IS the audio that was
        scored, by construction — written a median of 1 ms after the fire.
        There is no capture bug to fail open for.

        The 2026-08-15 incident it was built for predates v0.2.4, which
        replaced the evictable ring-buffer snapshot that produced those
        garbage clips.

        Meanwhile the rule fired on exactly the false-wake signature: prod
        misfires score 0.94-0.99, so every one of them was rescued. On
        2026-08-26 "Okay." and "-" both reached the model this way and were
        answered aloud into an empty kitchen.
        """
        self._store("Turn on the living room lights.", similarity=0.2)
        resolved = self._resolve({"source": "wake", "wake_confidence": 0.97})
        assert resolved is not None
        assert resolved["verified"] is False

    def test_no_playback_verdict_stays_unverified_in_cache(self):
        self._store("Turn on the living room lights.", similarity=0.2)
        self._resolve({"source": "wake", "wake_confidence": 0.97})
        cached = conversation_cache.get_wake_verification("conv-cpg")
        assert cached["verdict"] == "unverified"

    def test_low_confidence_zero_similarity_stays_unverified(self):
        # A marginal fire with a garbage clip really is a misfire signature.
        self._store("Eat it.", similarity=0.1)
        resolved = self._resolve({"source": "wake", "wake_confidence": 0.55})
        assert resolved is not None
        assert resolved["verified"] is False

    def test_high_confidence_partial_similarity_stays_unverified(self):
        # A trace of the phrase means the clip pipeline worked — the fuzzy
        # match just failed. That is NOT the capture-bug signature.
        self._store("the harvest is ready", similarity=0.62)
        resolved = self._resolve({"source": "wake", "wake_confidence": 0.97})
        assert resolved is not None
        assert resolved["verified"] is False

    def test_missing_wake_confidence_stays_unverified(self):
        self._store("Eat it.", similarity=0.0)
        resolved = self._resolve({"source": "wake"})
        assert resolved is not None
        assert resolved["verified"] is False

    def test_similarity_recomputed_when_absent(self):
        # Verdicts stored before the similarity field existed are still
        # scored. The gate that used to consume this is retired, so assert
        # the recompute directly rather than through it.
        sim = wv._verdict_similarity(
            {"transcript": "Turn on the living room lights.", "phrase": "jarvis"}
        )
        assert sim < wv.CLIP_SIMILARITY_FLOOR
        assert wv._verdict_similarity(
            {"transcript": "hey jarvis", "phrase": "jarvis"}
        ) >= wv.CLIP_SIMILARITY_FLOOR

    def test_verified_clip_unaffected(self):
        conversation_cache.set_wake_verification(
            "conv-cpg",
            {"verified": True, "verdict": "verified", "transcript": "hey jarvis",
             "phrase": "jarvis", "similarity": 1.0},
        )
        resolved = self._resolve({"source": "wake", "wake_confidence": 0.97})
        assert resolved is not None
        assert resolved["verified"] is True


class TestClipUnreliableFullyRetired:
    """The self-playback fail-open is gone too (2026-08-27).

    #124 kept it, reasoning that music bleed degrades the verify transcript
    so a no-trace clip during playback is a degraded-clip signature rather
    than a misfire. Prod falsified that within a day. The clips recorded
    during self-playback were not degraded at all:

        "- That's it."
        "No, it's younger."
        "power just to grow and see it around."
        "The law presumes an offense to be innocent."

    Clean, fluent transcriptions of television dialogue that simply do not
    contain the wake word. A degraded clip garbles; these do not. And the
    self-playback bar was 0.5, so essentially every fire during playback
    failed open, straight back into the full directed posture.
    """

    def setup_method(self):
        conversation_cache.set(
            "conv-cpg", [{"role": "system", "content": "x"}], [], "UTC", [], {}
        )

    def teardown_method(self):
        conversation_cache.remove("conv-cpg")

    def _resolve(self, turn_context, mode="bias"):
        async def _run():
            return await wv.resolve_wake_verification("conv-cpg", turn_context)
        import unittest.mock as m
        with m.patch.object(wv, "_get_mode", return_value=mode):
            return asyncio.run(_run())

    def _store(self, transcript, similarity=None, **extra):
        verdict = {
            "verified": False, "verdict": "unverified", "transcript": transcript,
            "phrase": "jarvis", "node_id": "node-1", **extra,
        }
        if similarity is not None:
            verdict["similarity"] = similarity
        conversation_cache.set_wake_verification("conv-cpg", verdict)

    def test_self_playback_no_longer_fails_open(self):
        self._store("No, it's younger.", similarity=0.40)
        resolved = self._resolve({
            "source": "wake", "wake_confidence": 0.97,
            "self_playback": True, "self_playback_kind": "music",
        })
        assert resolved is not None
        assert resolved["verified"] is False

    def test_verdict_stays_unverified_in_cache_during_playback(self):
        self._store("- That's it.", similarity=0.33)
        self._resolve({
            "source": "wake", "wake_confidence": 1.0,
            "self_playback": True, "self_playback_kind": "music",
        })
        cached = conversation_cache.get_wake_verification("conv-cpg")
        assert cached["verdict"] == "unverified"

    def test_verified_clip_still_unaffected(self):
        conversation_cache.set_wake_verification(
            "conv-cpg",
            {"verified": True, "verdict": "verified", "transcript": "hey jarvis",
             "phrase": "jarvis", "similarity": 1.0},
        )
        resolved = self._resolve({
            "source": "wake", "wake_confidence": 0.97, "self_playback": True,
        })
        assert resolved is not None
        assert resolved["verified"] is True

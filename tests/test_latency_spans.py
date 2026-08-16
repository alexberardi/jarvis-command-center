"""Latency instrumentation coverage: STT, TTS, continue-stream, and the
streaming trace-closure fix.

Prod latency analysis (2026-08-15) found three blind spots in the request
traces:

1. STT and TTS had ZERO spans — the whisper transcribe proxy had no
   RequestTiming at all, and the TTS client never recorded first-chunk /
   total-synth timings.
2. ``/voice/command/continue/stream`` (56% of stream traces run this second
   sequential LLM leg) had no RequestTiming either.
3. The streaming voice paths called ``latency_logger.end_request()`` BEFORE
   returning the StreamingResponse, so total_duration_ms excluded all
   time-to-first-byte and streaming time — and any span recorded during
   streaming landed on a dead trace.

These tests pin the new spans and the closure fix. Instrumentation only —
none of them assert behavior changes beyond when the trace is closed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.utils.latency_logger import RequestTiming, latency_logger


class _CapturingLatencyLogger:
    """Stand-in for the module-global latency_logger that keeps real
    RequestTiming objects (so ``measure`` / ``record_span`` work) but records
    start/end calls for assertions instead of persisting anything."""

    def __init__(self):
        self.timings: dict[str, RequestTiming] = {}
        self.started: list[tuple[str, str]] = []
        self.ended: list[str] = []

    def start_request(self, request_id: str, request_type: str) -> RequestTiming:
        timing = RequestTiming(request_id=request_id, request_type=request_type)
        self.timings[request_id] = timing
        self.started.append((request_id, request_type))
        return timing

    def get_request(self, request_id: str):
        return self.timings.get(request_id)

    def end_request(self, request_id: str) -> None:
        self.ended.append(request_id)
        self.timings.pop(request_id, None)


def _span_labels(timing: RequestTiming) -> list[str]:
    return [e.label for e in timing.entries]


def _get_entry(timing: RequestTiming, label: str):
    matches = [e for e in timing.entries if e.label == label]
    assert matches, f"no span labeled {label!r} in {_span_labels(timing)}"
    return matches[0]


class TestRecordSpan:
    """RequestTiming.record_span — explicit-timestamp spans for async
    iterators where a ``with measure(...)`` block can't wrap the work."""

    def test_records_entry_with_duration_and_service(self):
        timing = RequestTiming(request_id="r1", request_type="test")
        start = timing.start_time + 0.010  # 10ms in
        end = timing.start_time + 0.035    # 25ms long

        timing.record_span(
            "tts_first_chunk", start, end,
            service="tts", metadata={"text_chars": 12},
        )

        entry = _get_entry(timing, "tts_first_chunk")
        assert entry.service == "tts"
        assert entry.status == "ok"
        assert entry.metadata == {"text_chars": 12}
        assert entry.duration_ms == pytest.approx(25, abs=1)
        assert entry.start_time == pytest.approx(10, abs=1)

    def test_appears_in_to_spans_export(self):
        timing = RequestTiming(request_id="r2", request_type="test")
        timing.record_span(
            "stt_transcribe",
            timing.start_time,
            timing.start_time + 0.005,
            service="whisper",
        )

        spans = timing.to_spans()
        assert [s["name"] for s in spans] == ["stt_transcribe"]
        assert spans[0]["service"] == "whisper"
        assert spans[0]["duration_ms"] > 0


class TestSttTranscribeTrace:
    """The whisper transcribe proxy creates its own STT trace with an
    ``stt_transcribe`` span (service=whisper) and audio-size metadata."""

    def _post_transcribe(self, client, node, fake_logger, form=None):
        from app.main import app
        from app.deps import verify_api_key

        mock_context = MagicMock()
        mock_context.node = node
        mock_context.household_id = "house-1"
        mock_context.household_member_ids = [1, 2]
        app.dependency_overrides[verify_api_key] = lambda: mock_context

        with patch("app.api.media.WhisperClient") as mock_whisper_class:
            mock_client = MagicMock()
            mock_client.transcribe = AsyncMock(
                return_value={"text": "hello world"}
            )
            mock_whisper_class.return_value = mock_client

            with patch("app.api.media.latency_logger", fake_logger):
                try:
                    return client.post(
                        "/api/v0/media/whisper/transcribe",
                        files={"file": ("recording.wav", b"audio data")},
                        data=form or {},
                        headers={"X-API-Key": node.api_key},
                    )
                finally:
                    app.dependency_overrides.pop(verify_api_key, None)

    def test_creates_stt_trace_with_span_and_metadata(
        self, client_with_test_db, test_node_with_household
    ):
        node, _household_id = test_node_with_household
        fake = _CapturingLatencyLogger()

        response = self._post_transcribe(client_with_test_db, node, fake)

        assert response.status_code == 200
        assert len(fake.started) == 1
        trace_key, request_type = fake.started[0]
        assert request_type == "stt"
        # Ended in the route's finally — under the same (unique) key.
        assert fake.ended == [trace_key]

    def test_span_details_and_conversation_join(
        self, client_with_test_db, test_node_with_household
    ):
        node, _household_id = test_node_with_household
        fake = _CapturingLatencyLogger()
        captured: dict = {}

        original_end = fake.end_request

        def _capture_and_end(request_id: str) -> None:
            captured["timing"] = fake.timings.get(request_id)
            original_end(request_id)

        fake.end_request = _capture_and_end

        response = self._post_transcribe(
            client_with_test_db, node, fake, form={"conversation_id": "conv-42"}
        )

        assert response.status_code == 200
        timing = captured["timing"]
        assert timing is not None

        # Trace metadata: node source, node id, and — critically — the
        # persisted row joins to the voice turn via conversation_id even
        # though the in-flight dict key is unique (no warmup clobbering).
        assert timing.source == "node"
        assert timing.node_id == node.node_id
        assert timing.household_id == "house-1"
        assert timing.request_id == "conv-42"
        assert timing.user_command == "hello world"

        entry = _get_entry(timing, "stt_transcribe")
        assert entry.service == "whisper"
        assert entry.metadata["audio_bytes"] == len(b"audio data")
        assert entry.metadata["speaker_audio_bytes"] == 0

    def test_trace_key_is_not_conversation_id(
        self, client_with_test_db, test_node_with_household
    ):
        """The in-flight trace key must NOT be the conversation_id — a
        concurrently open warmup trace under the same conversation_id would
        be clobbered by start_request otherwise."""
        node, _household_id = test_node_with_household
        fake = _CapturingLatencyLogger()

        self._post_transcribe(
            client_with_test_db, node, fake, form={"conversation_id": "conv-42"}
        )

        trace_key, _ = fake.started[0]
        assert trace_key != "conv-42"
        assert trace_key.startswith("stt-")


class TestTtsStreamSpans:
    """TTSClient.speak_stream records tts_first_chunk / tts_stream_total on
    the conversation's open trace (service=tts)."""

    class _FakeStreamResp:
        headers = {
            "X-Audio-Sample-Rate": "22050",
            "X-Audio-Channels": "1",
            "X-Audio-Sample-Width": "2",
        }

        def raise_for_status(self):
            pass

        async def aiter_bytes(self, chunk_size=4096):
            yield b"\x00" * 10
            yield b"\x01" * 6

        async def aclose(self):
            pass

    class _FakeAsyncClient:
        def __init__(self, timeout=None):
            pass

        def build_request(self, *args, **kwargs):
            return MagicMock()

        async def send(self, request, stream=False):
            return TestTtsStreamSpans._FakeStreamResp()

        async def aclose(self):
            pass

    def _make_client(self, conversation_id):
        from app.core.clients.tts_client import TTSClient

        settings = MagicMock()
        settings.get.return_value = "http://tts.test"
        with patch(
            "app.core.clients.tts_client.get_settings_service",
            return_value=settings,
        ):
            client = TTSClient(
                household_id="house-1",
                node_id="node-1",
                conversation_id=conversation_id,
            )
        return client

    @pytest.mark.asyncio
    async def test_records_first_chunk_and_total_spans(self):
        timing = latency_logger.start_request("conv-tts", "voice_command_stream")
        try:
            client = self._make_client("conv-tts")
            with patch(
                "app.core.clients.tts_client.httpx.AsyncClient",
                self._FakeAsyncClient,
            ), patch.object(type(client), "_build_headers", return_value={}):
                audio_iter, meta = await client.speak_stream("Hello there.")
                chunks = [c async for c in audio_iter]

            assert b"".join(chunks) == b"\x00" * 10 + b"\x01" * 6
            assert meta["sample_rate"] == "22050"

            first = _get_entry(timing, "tts_first_chunk")
            assert first.service == "tts"
            assert first.metadata == {"text_chars": len("Hello there.")}

            total = _get_entry(timing, "tts_stream_total")
            assert total.service == "tts"
            assert total.metadata["audio_bytes"] == 16
            assert total.end_time >= first.end_time
        finally:
            latency_logger.end_request("conv-tts")

    @pytest.mark.asyncio
    async def test_no_spans_without_open_trace(self):
        """No conversation trace open (or no conversation_id) → zero span
        writes, identical audio behavior."""
        client = self._make_client("conv-without-trace")
        with patch(
            "app.core.clients.tts_client.httpx.AsyncClient",
            self._FakeAsyncClient,
        ), patch.object(type(client), "_build_headers", return_value={}):
            audio_iter, _meta = await client.speak_stream("Hi.")
            chunks = [c async for c in audio_iter]

        assert b"".join(chunks) == b"\x00" * 10 + b"\x01" * 6
        assert latency_logger.get_request("conv-without-trace") is None


class TestEndTraceAfterStream:
    """The streaming trace-closure fix: the trace ends when the audio stream
    finishes (or the client disconnects), never before."""

    @pytest.mark.asyncio
    async def test_trace_stays_open_until_stream_consumed(self):
        from app.main import _end_trace_after_stream

        timing = latency_logger.start_request("conv-close", "voice_command_stream")

        async def _audio():
            yield b"aa"
            yield b"bb"

        wrapped = _end_trace_after_stream(_audio(), "conv-close")

        first = await wrapped.__anext__()
        assert first == b"aa"
        # Mid-stream: the trace must still be open.
        assert latency_logger.get_request("conv-close") is timing

        rest = [chunk async for chunk in wrapped]
        assert rest == [b"bb"]

        # Fully consumed: trace is closed now.
        assert latency_logger.get_request("conv-close") is None

        # Spans: first-byte checkpoint + streaming-window span survive on the
        # timing object (recorded before end_request popped it).
        labels = _span_labels(timing)
        assert "first_audio_byte" in labels
        assert "audio_stream" in labels
        stream_span = _get_entry(timing, "audio_stream")
        assert stream_span.service == "cc"

    @pytest.mark.asyncio
    async def test_trace_closed_on_client_disconnect(self):
        """GeneratorExit (Starlette closes the generator when the node drops
        the connection) still runs the finally and closes the trace."""
        from app.main import _end_trace_after_stream

        latency_logger.start_request("conv-drop", "voice_command_stream")

        async def _audio():
            yield b"aa"
            yield b"bb"
            yield b"cc"

        wrapped = _end_trace_after_stream(_audio(), "conv-drop")
        await wrapped.__anext__()
        await wrapped.aclose()  # simulated disconnect

        assert latency_logger.get_request("conv-drop") is None

    @pytest.mark.asyncio
    async def test_no_trace_is_a_noop(self):
        """Wrapping a stream whose trace was never started must not raise."""
        from app.main import _end_trace_after_stream

        async def _audio():
            yield b"aa"

        chunks = [c async for c in _end_trace_after_stream(_audio(), "conv-none")]
        assert chunks == [b"aa"]


class TestContinueStreamTrace:
    """/voice/command/continue/stream now opens a voice_command_continue
    trace, spans the dispatch, and closes on stream completion (or before
    the 202 fallback)."""

    def _call_route(self, client, fake_logger, streaming_audio):
        from app.main import app
        from app.deps import verify_api_key, get_model_service

        mock_context = MagicMock()
        mock_context.node.node_id = "node-1"
        mock_context.household_id = "house-1"
        app.dependency_overrides[verify_api_key] = lambda: mock_context

        mock_service = MagicMock()
        mock_service.try_stream_continue_with_tool_results = AsyncMock(
            return_value=streaming_audio
        )
        app.dependency_overrides[get_model_service] = lambda: mock_service

        mock_tts = MagicMock()
        mock_tts.get_audio_format = AsyncMock(return_value={
            "sample_rate": "22050", "channels": "1", "sample_width": "2",
        })

        try:
            with patch("app.core.clients.tts_client.TTSClient", return_value=mock_tts), \
                 patch("app.main.latency_logger", fake_logger):
                return client.post(
                    "/api/v0/voice/command/continue/stream",
                    json={
                        "conversation_id": "conv-cont",
                        "tool_results": [
                            {"tool_call_id": "c1", "output": '{"message": "done"}'},
                        ],
                    },
                    headers={"X-API-Key": "node-1:key"},
                )
        finally:
            app.dependency_overrides.pop(verify_api_key, None)
            app.dependency_overrides.pop(get_model_service, None)

    def test_fallback_202_ends_trace_immediately(self, client_with_test_db):
        fake = _CapturingLatencyLogger()

        response = self._call_route(client_with_test_db, fake, streaming_audio=None)

        assert response.status_code == 202
        assert fake.started == [("conv-cont", "voice_command_continue")]
        assert fake.ended == ["conv-cont"]

    def test_streaming_200_ends_trace_after_stream(self, client_with_test_db):
        fake = _CapturingLatencyLogger()
        captured: dict = {}

        original_end = fake.end_request

        def _capture_and_end(request_id: str) -> None:
            captured["timing"] = fake.timings.get(request_id)
            original_end(request_id)

        fake.end_request = _capture_and_end

        async def _audio():
            yield b"pcm-1"
            yield b"pcm-2"

        response = self._call_route(client_with_test_db, fake, streaming_audio=_audio())

        assert response.status_code == 200
        assert response.content == b"pcm-1pcm-2"
        assert fake.ended == ["conv-cont"]

        timing = captured["timing"]
        assert timing.source == "node"
        assert timing.node_id == "node-1"
        assert timing.household_id == "house-1"

        labels = _span_labels(timing)
        assert "inbox_actions_push" in labels
        assert "continue_stream_dispatch" in labels
        # Closure-fix spans: recorded while the response body streamed —
        # impossible before the fix (the trace was already dead).
        assert "first_audio_byte" in labels
        assert "audio_stream" in labels


class TestFinalResponseGenerationSpan:
    """The post-tool-loop formatting LLM call gets a span — previously an
    unspanned gap of up to ~1.8s after tool_execution_loop."""

    def _handler(self):
        from app.core.conversation_handler import ConversationHandler

        handler = ConversationHandler(model=MagicMock(), llm_client=AsyncMock())
        handler.prompt_provider = None
        return handler

    @pytest.mark.asyncio
    async def test_process_voice_command_spans_server_tool_formatting(self):
        """stop_reason=server_tool_complete → _format_tool_result_text_mode
        runs inside a final_response_generation span (service=llm_proxy)."""
        handler = self._handler()
        timing = latency_logger.start_request("conv-frg", "voice_command_stream")

        try:
            with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
                mock_cache.get_messages.return_value = [
                    {"role": "system", "content": "sys"},
                ]
                mock_cache.get_tools.return_value = []
                mock_cache.get_available_commands.return_value = []
                mock_cache.get_node_context.return_value = {}
                mock_cache.get_timezone.return_value = None

                with patch("app.core.conversation_handler.ToolExecutionEngine") as MockEngine:
                    mock_engine = MagicMock()
                    mock_engine.execute = AsyncMock(return_value={
                        "stop_reason": "server_tool_complete",
                        "server_tool_results": [{
                            "tool_call_id": "c1",
                            "content": '{"success": true, "message": "It is sunny."}',
                        }],
                    })
                    MockEngine.return_value = mock_engine

                    result = await handler.process_voice_command_with_tools(
                        voice_command="what's the weather",
                        conversation_id="conv-frg",
                    )

            assert result["stop_reason"] == "complete"
            entry = _get_entry(timing, "final_response_generation")
            assert entry.service == "llm_proxy"
        finally:
            latency_logger.end_request("conv-frg")

    @pytest.mark.asyncio
    async def test_continue_text_mode_formatting_is_spanned(self):
        """Blocking continue (text-mode) also wraps the formatting call when
        a trace is open for the conversation."""
        handler = self._handler()
        timing = latency_logger.start_request("conv-frg2", "voice_command_continue")

        try:
            with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
                mock_cache.get_messages.return_value = [
                    {"role": "user", "content": "add milk"},
                ]
                mock_cache.get_tools.return_value = []
                mock_cache.get_node_context.return_value = {}

                result = await handler.continue_conversation_with_tool_results(
                    conversation_id="conv-frg2",
                    tool_results=[{
                        "tool_call_id": "c1",
                        "output": {"success": True, "message": "Added milk."},
                    }],
                )

            assert result["assistant_message"] == "Added milk."
            entry = _get_entry(timing, "final_response_generation")
            assert entry.service == "llm_proxy"
        finally:
            latency_logger.end_request("conv-frg2")

    @pytest.mark.asyncio
    async def test_no_trace_open_does_not_break_formatting(self):
        """nullcontext branch: no trace registered → formatting still works."""
        handler = self._handler()

        with patch("app.core.conversation_handler.conversation_cache") as mock_cache:
            mock_cache.get_messages.return_value = [
                {"role": "user", "content": "add milk"},
            ]
            mock_cache.get_tools.return_value = []
            mock_cache.get_node_context.return_value = {}

            result = await handler.continue_conversation_with_tool_results(
                conversation_id="conv-no-trace",
                tool_results=[{
                    "tool_call_id": "c1",
                    "output": {"success": True, "message": "Added milk."},
                }],
            )

        assert result["assistant_message"] == "Added milk."

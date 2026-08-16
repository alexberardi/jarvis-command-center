"""TTS client with app-to-app auth and context headers.

This client proxies TTS requests from command-center to jarvis-tts,
using app-to-app authentication and passing context headers for
household/node/user identification.
"""

import logging
import time
from typing import AsyncIterator

import httpx

from jarvis_auth_client.headers import get_app_headers, build_context_headers
from app.core.utils.latency_logger import latency_logger
from app.services.settings_service import get_settings_service

logger = logging.getLogger("uvicorn")

# Last-resort fallback (only used when settings AND service discovery fail)
_FALLBACK_TTS_URL = "http://localhost:7707"


def _get_tts_url_from_discovery() -> str | None:
    """Get TTS URL via service discovery (handles Docker URL rewriting)."""
    try:
        from app.core import service_config
        if service_config.is_initialized():
            return service_config.get_tts_url()
    except (ImportError, AttributeError, ValueError):
        pass
    return None


class TTSClient:
    """Client for interacting with the TTS service."""

    def __init__(
        self,
        household_id: str,
        node_id: str | None = None,
        user_id: int | None = None,
        conversation_id: str | None = None,
    ) -> None:
        """Initialize the TTS client.

        Args:
            household_id: The household making the request
            node_id: Optional specific node making the request
            user_id: Optional user associated with the request
            conversation_id: Optional conversation whose open latency trace
                (if any) receives ``tts_first_chunk`` / ``tts_stream_total``
                spans from streaming calls. Instrumentation only — no
                behavior change when omitted or when no trace is open.
        """
        self.household_id = household_id
        self.node_id = node_id
        self.user_id = user_id
        self.conversation_id = conversation_id

        # URL resolution priority:
        # 1. Per-household/node setting (tts.url)
        # 2. Service discovery (handles Docker URL rewriting)
        # 3. Hardcoded fallback
        settings = get_settings_service()
        url = settings.get(
            "tts.url",
            household_id=household_id,
            node_id=node_id,
        )
        if not url:
            url = _get_tts_url_from_discovery()
        self.base_url = url if url else _FALLBACK_TTS_URL

    def _build_headers(self) -> dict[str, str]:
        """Build headers including app auth and context.

        Returns:
            Dict with app-to-app auth headers and context headers
        """
        headers = {
            **get_app_headers(),
            **build_context_headers(self.household_id, self.node_id, self.user_id),
        }
        return headers

    async def speak(self, text: str, timeout: float = 30.0) -> bytes:
        """Convert text to speech.

        Args:
            text: The text to convert to speech
            timeout: Request timeout in seconds

        Returns:
            Audio bytes (WAV format)

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        url = f"{self.base_url.rstrip('/')}/speak"

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json={"text": text},
                headers=self._build_headers(),
            )
            response.raise_for_status()
            return response.content

    async def speak_stream(
        self,
        text: str,
        timeout: float = 30.0,
    ) -> tuple[AsyncIterator[bytes], dict[str, str]]:
        """Stream raw PCM audio from TTS service.

        Args:
            text: The text to convert to speech
            timeout: Request timeout in seconds

        Returns:
            Tuple of (async byte iterator, audio metadata dict with
            sample_rate, channels, sample_width from response headers)

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        url = f"{self.base_url.rstrip('/')}/speak/stream"

        # Latency spans (instrumentation only). Looked up per call — the
        # trace stays open through the whole streaming window, so every
        # sentence's synth gets its own tts_first_chunk / tts_stream_total.
        timing = (
            latency_logger.get_request(self.conversation_id)
            if self.conversation_id else None
        )
        request_start = time.perf_counter()

        client = httpx.AsyncClient(timeout=timeout)
        resp = await client.send(
            client.build_request("POST", url, json={"text": text}, headers=self._build_headers()),
            stream=True,
        )
        resp.raise_for_status()

        audio_meta = {
            "sample_rate": resp.headers.get("X-Audio-Sample-Rate", "22050"),
            "channels": resp.headers.get("X-Audio-Channels", "1"),
            "sample_width": resp.headers.get("X-Audio-Sample-Width", "2"),
        }

        async def _iter_and_close() -> AsyncIterator[bytes]:
            first_chunk = True
            total_bytes = 0
            try:
                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    if first_chunk:
                        first_chunk = False
                        if timing is not None:
                            # Time-to-first-audio-chunk from the TTS service,
                            # measured from the speak_stream() call.
                            timing.record_span(
                                "tts_first_chunk",
                                request_start,
                                time.perf_counter(),
                                service="tts",
                                metadata={"text_chars": len(text)},
                            )
                    total_bytes += len(chunk)
                    yield chunk
            finally:
                if timing is not None:
                    timing.record_span(
                        "tts_stream_total",
                        request_start,
                        time.perf_counter(),
                        service="tts",
                        metadata={
                            "text_chars": len(text),
                            "audio_bytes": total_bytes,
                        },
                    )
                await resp.aclose()
                await client.aclose()

        return _iter_and_close(), audio_meta

    async def get_audio_format(self, timeout: float = 5.0) -> dict[str, str]:
        """Get the TTS voice's audio format without synthesizing.

        Returns:
            Dict with sample_rate, channels, sample_width as strings.
        """
        url = f"{self.base_url.rstrip('/')}/audio/format"

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=self._build_headers())
            response.raise_for_status()
            data = response.json()
            return {
                "sample_rate": str(data["sample_rate"]),
                "channels": str(data["channels"]),
                "sample_width": str(data["sample_width"]),
            }

    async def generate_wake_response(self, timeout: float = 10.0) -> str:
        """Generate a dynamic wake response greeting.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Generated greeting text

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        url = f"{self.base_url.rstrip('/')}/generate-wake-response"

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            data = response.json()
            return data.get("text", "Yes?")

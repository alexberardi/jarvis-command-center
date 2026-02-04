"""Service clients for proxying requests to downstream services."""

from app.core.clients.tts_client import TTSClient
from app.core.clients.whisper_client import WhisperClient

__all__ = ["TTSClient", "WhisperClient"]

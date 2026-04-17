"""Whisper client with app-to-app auth and context headers.

This client proxies transcription requests from command-center to jarvis-whisper-api,
using app-to-app authentication and passing context headers. The household_id context
is critical for voice recognition - Whisper uses it to filter voice profiles.
"""

from typing import Any

import httpx

from jarvis_auth_client.headers import get_app_headers, build_context_headers
from app.services.settings_service import get_settings_service

# Last-resort fallback (only used when settings AND service discovery fail)
_FALLBACK_WHISPER_URL = "http://localhost:7706"


def _get_whisper_url_from_discovery() -> str | None:
    """Get Whisper URL via service discovery (handles Docker URL rewriting)."""
    try:
        from app.core import service_config
        if service_config.is_initialized():
            return service_config.get_whisper_url()
    except (ImportError, AttributeError, ValueError):
        pass
    return None


class WhisperClient:
    """Client for interacting with the Whisper transcription service."""

    def __init__(
        self,
        household_id: str,
        node_id: str | None = None,
        user_id: int | None = None,
        household_member_ids: list[int] | None = None,
    ) -> None:
        """Initialize the Whisper client.

        Args:
            household_id: The household making the request (used for voice recognition)
            node_id: Optional specific node making the request
            user_id: Optional user associated with the request
            household_member_ids: List of member IDs in household (for voice recognition)
        """
        self.household_id = household_id
        self.node_id = node_id
        self.user_id = user_id
        self.household_member_ids = household_member_ids or []

        # URL resolution priority (same pattern as TTS client):
        # 1. Per-household/node setting (whisper.url)
        # 2. Service discovery (handles Docker URL rewriting)
        # 3. Hardcoded fallback
        settings = get_settings_service()
        url = settings.get(
            "whisper.url",
            household_id=household_id,
            node_id=node_id,
        )
        if not url:
            url = _get_whisper_url_from_discovery()
        self.base_url = url if url else _FALLBACK_WHISPER_URL

    def _build_headers(self) -> dict[str, str]:
        """Build headers including app auth and context.

        Returns:
            Dict with app-to-app auth headers and context headers
        """
        headers = {
            **get_app_headers(),
            **build_context_headers(
                self.household_id,
                self.node_id,
                self.user_id,
                self.household_member_ids,
            ),
        }
        return headers

    async def transcribe(
        self,
        audio: bytes,
        filename: str,
        timeout: float = 60.0,
        **params: Any,
    ) -> dict[str, Any]:
        """Transcribe audio to text.

        The household_id context header is passed to Whisper so it can filter
        voice profiles for speaker recognition.

        Args:
            audio: Audio bytes to transcribe
            filename: Original filename (used for format detection)
            timeout: Request timeout in seconds
            **params: Additional parameters (language, task, etc.)

        Returns:
            Transcription result dict with text, optional speaker_id, etc.

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        url = f"{self.base_url.rstrip('/')}/transcribe"

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                files={"file": (filename, audio)},
                data=params if params else None,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            return response.json()

    async def enroll_voice_profile(
        self,
        user_id: int,
        audio: bytes,
        filename: str,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Upload a voice sample for speaker enrollment.

        Args:
            user_id: The user to enroll
            audio: WAV audio bytes
            filename: Original filename
            timeout: Request timeout in seconds

        Returns:
            Enrollment result dict
        """
        url = f"{self.base_url.rstrip('/')}/voice-profiles/enroll"
        params = {
            "user_id": str(user_id),
            "household_id": self.household_id,
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                files={"file": (filename, audio)},
                params=params,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            return response.json()

    async def delete_voice_profile(
        self,
        user_id: int,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Delete a voice profile.

        Args:
            user_id: The user whose profile to delete
            timeout: Request timeout in seconds

        Returns:
            Deletion result dict
        """
        url = f"{self.base_url.rstrip('/')}/voice-profiles/{user_id}"
        params = {"household_id": self.household_id}

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.delete(
                url,
                params=params,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            return response.json()

    async def list_voice_profiles(
        self,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """List voice profiles for the household.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Dict with profiles list
        """
        url = f"{self.base_url.rstrip('/')}/voice-profiles"
        params = {"household_id": self.household_id}

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                url,
                params=params,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            return response.json()

    async def check_voice_profile(
        self,
        user_id: int,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Check whether a voice profile exists for a user.

        Args:
            user_id: The user to check
            timeout: Request timeout in seconds

        Returns:
            Dict with ``exists`` boolean
        """
        url = f"{self.base_url.rstrip('/')}/voice-profiles/check"
        params = {"user_id": str(user_id), "household_id": self.household_id}

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                url,
                params=params,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            return response.json()

    async def verify_voice_profile(
        self,
        user_id: int,
        audio: bytes,
        filename: str,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Test whether an audio sample matches a user's voice profile.

        Args:
            user_id: The user to verify against
            audio: WAV audio bytes
            filename: Original filename
            timeout: Request timeout in seconds

        Returns:
            Dict with ``matched`` boolean and ``confidence`` float
        """
        url = f"{self.base_url.rstrip('/')}/voice-profiles/verify"
        params = {"user_id": str(user_id), "household_id": self.household_id}

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                files={"file": (filename, audio)},
                params=params,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            return response.json()

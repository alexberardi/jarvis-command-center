"""Whisper client with app-to-app auth and context headers.

This client proxies transcription requests from command-center to jarvis-whisper-api,
using app-to-app authentication and passing context headers. The household_id context
is critical for voice recognition - Whisper uses it to filter voice profiles.
"""

from typing import Any

import httpx

from jarvis_auth_client.headers import get_app_headers, build_context_headers
from app.services.settings_service import get_settings_service

# Default Whisper URL if not configured in settings
DEFAULT_WHISPER_URL = "http://localhost:7706"


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

        # Get URL from settings with cascade lookup (Node > Household > Default)
        settings = get_settings_service()
        url = settings.get(
            "whisper.url",
            household_id=household_id,
            node_id=node_id,
        )
        self.base_url = url if url else DEFAULT_WHISPER_URL

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

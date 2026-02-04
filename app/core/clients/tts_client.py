"""TTS client with app-to-app auth and context headers.

This client proxies TTS requests from command-center to jarvis-tts,
using app-to-app authentication and passing context headers for
household/node/user identification.
"""

import httpx
from sqlalchemy.orm import Session

from jarvis_auth_client.headers import get_app_headers, build_context_headers
from app.services.settings_service import SettingsService

# Default TTS URL if not configured in settings
DEFAULT_TTS_URL = "http://localhost:8009"


class TTSClient:
    """Client for interacting with the TTS service."""

    def __init__(
        self,
        db: Session,
        household_id: str,
        node_id: str | None = None,
        user_id: int | None = None,
    ) -> None:
        """Initialize the TTS client.

        Args:
            db: Database session for settings lookup
            household_id: The household making the request
            node_id: Optional specific node making the request
            user_id: Optional user associated with the request
        """
        self.household_id = household_id
        self.node_id = node_id
        self.user_id = user_id

        # Get URL from settings with cascade lookup (Node > Household > Default)
        settings = SettingsService(db)
        url = settings.get_setting("tts_url", household_id, node_id)
        self.base_url = url if url else DEFAULT_TTS_URL

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

import httpx
import json
import os
from typing import Optional
from app.core.interfaces.ijarvis_context_provider import ICommandInferenceSystemPromptProvider


def build_jarvis_app_headers() -> dict:
    """
    Build Jarvis app-to-app authentication headers if environment variables are set.
    """
    app_id = os.getenv("LLM_APP_ID")
    app_key = os.getenv("LLM_APP_KEY")

    headers = {}
    if app_id:
        headers["X-Jarvis-App-Id"] = app_id
    if app_key:
        headers["X-Jarvis-App-Key"] = app_key

    return headers

async def post(
    url: str,
    json_data: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = 30
) -> dict:
    json_data = json_data or {}

    # Default headers
    default_headers = {
        "Content-Type": "application/json",
        **build_jarvis_app_headers(),
    }

    # Merge with user headers (user overrides take precedence)
    merged_headers = {**default_headers, **(headers or {})}

    # Use individual HTTP client for each request (like before) but disable HTTP/2
    async with httpx.AsyncClient(timeout=timeout, http2=False) as client:
        response = await client.post(
            url,
            json=json_data,
            headers=merged_headers
        )
        response.raise_for_status()
        return response.json()


async def get(
    url: str,
    headers: Optional[dict] = None,
    timeout: int = 30
) -> dict:
    # Default headers
    default_headers = {
        "Content-Type": "application/json",
        **build_jarvis_app_headers(),
    }

    # Merge with user headers (user overrides take precedence)
    merged_headers = {**default_headers, **(headers or {})}

    # Use individual HTTP client for each request (like before) but disable HTTP/2
    async with httpx.AsyncClient(timeout=timeout, http2=False) as client:
        response = await client.get(
            url,
            headers=merged_headers
        )
        response.raise_for_status()
        return response.json()



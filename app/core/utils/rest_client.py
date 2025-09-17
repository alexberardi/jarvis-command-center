import httpx
import json
from typing import Optional 
from app.core.interfaces.ijarvis_context_provider import ICommandInferenceSystemPromptProvider

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
        # "Host": "localhost"
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



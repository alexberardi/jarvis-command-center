import httpx
import os
from typing import Optional


def build_jarvis_app_headers() -> dict:
    """
    Build Jarvis app-to-app authentication headers if environment variables are set.
    """
    app_id = os.getenv("JARVIS_APP_ID")
    app_key = os.getenv("JARVIS_APP_KEY")

    headers = {}
    if app_id:
        headers["X-Jarvis-App-Id"] = app_id
    if app_key:
        headers["X-Jarvis-App-Key"] = app_key

    return headers


# Module-level shared client — reuses connections (keep-alive) across calls.
# A fresh AsyncClient per request was paying ~40-60ms in TCP+TLS handshake
# every LLM call. Pooling eliminates that on the warm path.
# Timeout is overridden per-request via httpx.Timeout.
_shared_client: Optional[httpx.AsyncClient] = None


def _get_shared_client() -> httpx.AsyncClient:
    """Get or lazily-create the shared AsyncClient."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            http2=False,
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
                keepalive_expiry=60.0,
            ),
        )
    return _shared_client


async def close_shared_client() -> None:
    """Close the shared client. Call from FastAPI shutdown hook."""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
    _shared_client = None


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

    client = _get_shared_client()
    response = await client.post(
        url,
        json=json_data,
        headers=merged_headers,
        timeout=httpx.Timeout(timeout),
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

    client = _get_shared_client()
    response = await client.get(
        url,
        headers=merged_headers,
        timeout=httpx.Timeout(timeout),
    )
    response.raise_for_status()
    return response.json()



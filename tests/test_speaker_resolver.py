"""Tests for speaker_resolver utility."""
import pytest
import time
from unittest.mock import AsyncMock, patch

from app.core.utils.speaker_resolver import resolve_speaker_name, _speaker_cache


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the speaker cache before each test."""
    _speaker_cache.clear()
    yield
    _speaker_cache.clear()


@pytest.mark.asyncio
async def test_resolve_returns_username():
    with patch("app.core.utils.speaker_resolver.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {"users": {"42": "alice"}}

        name = await resolve_speaker_name("http://auth:7701", 42)

        assert name == "alice"
        mock_get.assert_called_once()


@pytest.mark.asyncio
async def test_resolve_caches_result():
    with patch("app.core.utils.speaker_resolver.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {"users": {"42": "alice"}}

        name1 = await resolve_speaker_name("http://auth:7701", 42)
        name2 = await resolve_speaker_name("http://auth:7701", 42)

        assert name1 == "alice"
        assert name2 == "alice"
        # Should only call the API once due to caching
        assert mock_get.call_count == 1


@pytest.mark.asyncio
async def test_resolve_returns_none_on_missing_user():
    with patch("app.core.utils.speaker_resolver.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {"users": {}}

        name = await resolve_speaker_name("http://auth:7701", 999)

        assert name is None


@pytest.mark.asyncio
async def test_resolve_returns_none_on_error():
    with patch("app.core.utils.speaker_resolver.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = Exception("Connection refused")

        name = await resolve_speaker_name("http://auth:7701", 42)

        assert name is None


@pytest.mark.asyncio
async def test_cache_expires():
    with patch("app.core.utils.speaker_resolver.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {"users": {"42": "alice"}}

        # Pre-populate cache with expired entry
        _speaker_cache[42] = ("alice", time.time() - 1)

        name = await resolve_speaker_name("http://auth:7701", 42)

        assert name == "alice"
        # Should have called API since cache expired
        assert mock_get.call_count == 1

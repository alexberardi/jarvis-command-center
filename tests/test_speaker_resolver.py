"""Tests for speaker_resolver utility."""
import pytest
import time
from unittest.mock import AsyncMock, patch

from app.core.utils.speaker_resolver import (
    resolve_member_names,
    resolve_speaker_name,
    _speaker_cache,
)


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


@pytest.mark.asyncio
async def test_member_names_batch_resolves_in_one_call():
    with patch("app.core.utils.speaker_resolver.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {"users": {"1": "Miles", "2": "Jess"}}

        names = await resolve_member_names("http://auth:7701", [1, 2])

        assert names == ["Miles", "Jess"]
        assert mock_get.call_count == 1
        # REPEATED params, not comma-joined. jarvis-auth declares
        # `user_ids: List[int] = Query(...)`, which FastAPI parses as repeated
        # `?user_ids=1&user_ids=2`. A comma-joined "1,2" fails int coercion and
        # the endpoint 422s. This assertion used to encode the bug (it asserted
        # "user_ids=1,2"), which is why the suite stayed green while prod logged
        # the failure on every conversation start for months.
        assert "user_ids=1&user_ids=2" in mock_get.call_args[0][0]
        assert "user_ids=1,2" not in mock_get.call_args[0][0]


@pytest.mark.asyncio
async def test_member_names_uses_and_fills_cache():
    with patch("app.core.utils.speaker_resolver.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {"users": {"2": "Jess"}}
        _speaker_cache[1] = ("Miles", time.time() + 100)

        names = await resolve_member_names("http://auth:7701", [1, 2])

        assert names == ["Miles", "Jess"]
        # Only the miss goes over the wire.
        assert mock_get.call_count == 1
        assert "user_ids=2" in mock_get.call_args[0][0]
        # And the miss lands in the cache for the next caller.
        assert _speaker_cache[2][0] == "Jess"


@pytest.mark.asyncio
async def test_member_names_drops_unresolvable_ids():
    with patch("app.core.utils.speaker_resolver.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {"users": {"1": "Miles"}}

        names = await resolve_member_names("http://auth:7701", [1, 999])

        assert names == ["Miles"]


@pytest.mark.asyncio
async def test_member_names_never_raises():
    # Best-effort hint signal: an auth outage degrades to no names.
    with patch("app.core.utils.speaker_resolver.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = Exception("Connection refused")

        assert await resolve_member_names("http://auth:7701", [1, 2]) == []


@pytest.mark.asyncio
async def test_member_names_empty_ids_no_call():
    with patch("app.core.utils.speaker_resolver.get", new_callable=AsyncMock) as mock_get:
        assert await resolve_member_names("http://auth:7701", []) == []
        assert await resolve_member_names("http://auth:7701", None) == []
        assert mock_get.call_count == 0


@pytest.mark.asyncio
async def test_member_names_url_never_comma_joins_ids():
    """Regression for the prod 422 (observed 2026-08-25, every conversation
    start on the kitchen node):

        Failed to batch-resolve member names: Client error '422 Unprocessable
        Entity' for url '.../internal/users/batch?user_ids=1,4'

    `resolve_member_names` fails soft by design — an auth outage degrades to no
    names — so a permanently malformed URL looked exactly like "this household
    has no other members" and never surfaced. Single-id lookups
    (`resolve_speaker_name`) were unaffected, which is why speaker naming kept
    working and hid the breakage.
    """
    with patch("app.core.utils.speaker_resolver.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {"users": {"1": "Miles", "4": "Jess", "7": "Sam"}}

        await resolve_member_names("http://auth:7701", [1, 4, 7])

        url = mock_get.call_args[0][0]
        assert url.count("user_ids=") == 3, url
        assert "," not in url.split("?", 1)[1], url

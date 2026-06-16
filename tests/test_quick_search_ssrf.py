"""SSRF-guard tests for quick_search_tool's inline scraper.

quick_search previously fetched DuckDuckGo result URLs with
``follow_redirects=True`` and no host check at all — a public result could 302
to ``127.0.0.1`` / ``169.254.169.254`` / a LAN host. These tests cover the
per-hop validation that closes that.

Hermetic: no network, no DB. Uses literal IPs (no DNS) and a stub client.
"""

import socket

import httpx
import pytest

from app.core.tools.quick_search_tool import _is_blocked_host, _safe_get


class _StubClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, headers: dict | None = None) -> httpx.Response:
        self.calls.append(url)
        if not self._responses:
            raise AssertionError(f"unexpected GET {url}")
        return self._responses.pop(0)


def _redirect(location: str, status: int = 302) -> httpx.Response:
    return httpx.Response(status, headers={"location": location})


def _ok() -> httpx.Response:
    return httpx.Response(200, content=b"<html>ok</html>")


_PUBLIC = "http://93.184.216.34/"  # literal public IP — no DNS in the hot path


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "169.254.169.254", "10.0.0.1", "192.168.0.1", "::1",
     "::ffff:127.0.0.1", "64:ff9b::7f00:1", "100.64.1.2", "0.0.0.0", "::",
     "localhost", "localhost.", "LOCALHOST"],
)
def test_blocks_internal_hosts(host: str) -> None:
    assert _is_blocked_host(host) is True


@pytest.mark.parametrize("host", ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"])
def test_allows_public(host: str) -> None:
    assert _is_blocked_host(host) is False


def test_is_blocked_host_none() -> None:
    assert _is_blocked_host(None) is True


def test_hostname_resolving_private_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))],
    )
    assert _is_blocked_host("evil.example") is True


def test_unresolvable_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a, **k):
        raise socket.gaierror("nope")

    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    assert _is_blocked_host("nope.invalid") is True


def test_safe_get_rejects_redirect_to_loopback() -> None:
    client = _StubClient([_redirect("http://127.0.0.1/")])
    with pytest.raises(ValueError, match="private or disallowed"):
        _safe_get(client, _PUBLIC, {"User-Agent": "x"})
    assert len(client.calls) == 1  # private hop never fetched


def test_safe_get_rejects_redirect_to_metadata() -> None:
    client = _StubClient([_redirect("http://169.254.169.254/latest/meta-data/")])
    with pytest.raises(ValueError, match="private or disallowed"):
        _safe_get(client, _PUBLIC, {"User-Agent": "x"})


def test_safe_get_rejects_non_http_redirect() -> None:
    client = _StubClient([_redirect("file:///etc/passwd")])
    with pytest.raises(ValueError, match="Invalid URL or redirect target"):
        _safe_get(client, _PUBLIC, {"User-Agent": "x"})


def test_safe_get_rejects_initial_private_url() -> None:
    client = _StubClient([])  # never fetched
    with pytest.raises(ValueError, match="private or disallowed"):
        _safe_get(client, "http://127.0.0.1/admin", {"User-Agent": "x"})
    assert client.calls == []


def test_safe_get_normal_chain_succeeds() -> None:
    client = _StubClient([_redirect("http://1.1.1.1/"), _ok()])
    resp = _safe_get(client, _PUBLIC, {"User-Agent": "x"})
    assert resp.status_code == 200
    assert len(client.calls) == 2


def test_safe_get_too_many_redirects() -> None:
    client = _StubClient([_redirect("http://1.1.1.1/"), _redirect("http://8.8.8.8/")])
    with pytest.raises(ValueError, match="Too many redirects"):
        _safe_get(client, _PUBLIC, {"User-Agent": "x"}, max_redirects=1)

"""Tests for the ``updates.allow_check`` global gate.

The node update check (``github_releases.latest_release``) makes an OUTBOUND
request to ``api.github.com`` to discover the newest node-setup release. That
egress is gated on the ``updates.allow_check`` setting, which defaults OFF so a
fully-local deployment never phones GitHub. As with the web_search gate this is
a *privacy claim*, so the gate must FAIL CLOSED on any settings error and must
not touch the network (or even the cache) when disabled.

Mirrors tests/test_web_search_gate.py. Hermetic: no network, no DB — settings,
httpx, and the in-process release cache are all stubbed.
"""
import pytest
from unittest.mock import MagicMock, patch

from app.services import github_releases
from app.services.github_releases import (
    ReleaseInfo,
    latest_release,
    resolve_target_version,
)

SETTINGS = "app.services.settings_service.get_settings_service"


def _settings_returning(value):
    """get_settings_service() stub whose .get(...) always returns ``value``."""
    svc = MagicMock()
    svc.get.return_value = value
    return MagicMock(return_value=svc)


def _settings_raising():
    """get_settings_service() stub that raises — exercises fail-closed."""
    return MagicMock(side_effect=RuntimeError("settings service down"))


@pytest.fixture(autouse=True)
def _clear_release_cache():
    """The release cache is module-global; reset it around every test so a
    cached value from one case can't leak into the next."""
    github_releases._cache.clear()
    yield
    github_releases._cache.clear()


def _github_release_payload():
    """A minimal GET /releases/latest body for the happy path."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"tag_name": "v0.3.0", "published_at": "2026-06-01T00:00:00Z"}
    client = MagicMock()
    client.get.return_value = resp
    client_cm = MagicMock()
    client_cm.__enter__.return_value = client
    client_cm.__exit__.return_value = False
    return client_cm, client


class TestLatestReleaseGate:
    def test_disabled_returns_none_and_no_httpx(self):
        # Setting OFF: no ReleaseInfo, and crucially no outbound request.
        with patch(SETTINGS, _settings_returning(False)), \
             patch("app.services.github_releases.httpx.Client") as client_cls:
            assert latest_release() is None
            client_cls.assert_not_called()
            # The cache must stay untouched while disabled.
            assert github_releases._cache == {}

    def test_enabled_returns_release_and_calls_httpx(self):
        client_cm, client = _github_release_payload()
        with patch(SETTINGS, _settings_returning(True)), \
             patch("app.services.github_releases.httpx.Client", return_value=client_cm) as client_cls:
            info = latest_release()
            assert isinstance(info, ReleaseInfo)
            assert info.tag == "v0.3.0"
            assert info.version == "0.3.0"
            client_cls.assert_called_once()
            client.get.assert_called_once()

    def test_settings_error_fails_closed_no_httpx(self):
        # A settings outage must DISABLE the check, never enable it.
        with patch(SETTINGS, _settings_raising()), \
             patch("app.services.github_releases.httpx.Client") as client_cls:
            assert latest_release() is None
            client_cls.assert_not_called()

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (True, True), ("true", True), ("True", True), ("1", True), ("yes", True),
            (False, False), ("false", False), ("0", False), ("no", False),
            (None, False), ("", False),
        ],
    )
    def test_coerces_setting_value(self, raw, expected):
        client_cm, _ = _github_release_payload()
        with patch(SETTINGS, _settings_returning(raw)), \
             patch("app.services.github_releases.httpx.Client", return_value=client_cm):
            result = latest_release()
            assert (result is not None) is expected

    def test_passes_household_id_to_settings(self):
        svc = MagicMock()
        svc.get.return_value = False
        with patch(SETTINGS, MagicMock(return_value=svc)), \
             patch("app.services.github_releases.httpx.Client"):
            latest_release(household_id="hh-xyz")
            svc.get.assert_called_once_with("updates.allow_check", household_id="hh-xyz")


class TestResolveTargetVersion:
    def test_explicit_version_bypasses_github_when_disabled(self):
        # An explicit version must install even with checks OFF — and without
        # any outbound request.
        with patch(SETTINGS, _settings_returning(False)), \
             patch("app.services.github_releases.httpx.Client") as client_cls:
            assert resolve_target_version("v0.3.1") == "0.3.1"
            client_cls.assert_not_called()

    def test_latest_disabled_returns_none_no_httpx(self):
        with patch(SETTINGS, _settings_returning(False)), \
             patch("app.services.github_releases.httpx.Client") as client_cls:
            assert resolve_target_version("latest") is None
            client_cls.assert_not_called()

    def test_latest_enabled_resolves_via_github(self):
        client_cm, _ = _github_release_payload()
        with patch(SETTINGS, _settings_returning(True)), \
             patch("app.services.github_releases.httpx.Client", return_value=client_cm):
            assert resolve_target_version("latest") == "0.3.0"

    def test_threads_household_id_through(self):
        svc = MagicMock()
        svc.get.return_value = False
        with patch(SETTINGS, MagicMock(return_value=svc)), \
             patch("app.services.github_releases.httpx.Client"):
            resolve_target_version("latest", household_id="hh-abc")
            svc.get.assert_called_once_with("updates.allow_check", household_id="hh-abc")

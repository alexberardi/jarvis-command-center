"""Tests for the ``web_scraping.allow_external`` household gate.

The deep_research scraper can fall back to the third-party r.jina.ai reader
proxy when a page can't be fetched directly. That fallback leaks which pages
the household reads to a third party, so it's opt-in per household via
``web_scraping.allow_external`` (default OFF). The gate must FAIL CLOSED on any
settings error and is plumbed into jarvis-web-scraper as
``FetchConfig(enable_jina_fallback=...)``.

Hermetic: no network, no DB. Settings are stubbed and WebScraper.batch_fetch is
patched to a no-op so nothing actually fetches — we only assert which
FetchConfig the scraper was constructed with.
"""
import asyncio

import pytest
from unittest.mock import MagicMock, patch

from jarvis_web_scraper import FetchConfig

from app.services import deep_research_service as drs

SETTINGS = "app.services.settings_service.get_settings_service"


def _settings_returning(value):
    """get_settings_service() stub whose .get(...) always returns ``value``."""
    svc = MagicMock()
    svc.get.return_value = value
    return MagicMock(return_value=svc)


def _settings_raising():
    """get_settings_service() stub that raises — exercises fail-closed."""
    return MagicMock(side_effect=RuntimeError("settings service down"))


def _run_scrape_capturing_config(household_id="hh-1"):
    """Call _scrape_urls and return the FetchConfig the WebScraper got built with.

    WebScraper.batch_fetch is no-op'd so no network access happens.
    """
    captured: dict[str, FetchConfig] = {}

    def _capture_init(self, config=None):
        captured["config"] = config if config is not None else FetchConfig()
        # Skip the real heavy constructor; we only need the captured config.

    async def _noop_batch_fetch(self, *a, **k):
        return []

    with patch("jarvis_web_scraper.WebScraper.__init__", _capture_init), \
         patch("jarvis_web_scraper.WebScraper.batch_fetch", _noop_batch_fetch):
        asyncio.run(drs._scrape_urls(["https://example.com"], household_id))

    return captured["config"]


class TestExternalScrapingGate:
    def test_disabled_constructs_jina_fallback_false(self):
        with patch(SETTINGS, _settings_returning(False)):
            cfg = _run_scrape_capturing_config()
            assert cfg.enable_jina_fallback is False

    def test_enabled_constructs_jina_fallback_true(self):
        with patch(SETTINGS, _settings_returning(True)):
            cfg = _run_scrape_capturing_config()
            assert cfg.enable_jina_fallback is True

    def test_settings_error_fails_closed(self):
        with patch(SETTINGS, _settings_raising()):
            cfg = _run_scrape_capturing_config()
            assert cfg.enable_jina_fallback is False

    def test_string_true_coerces_to_enabled(self):
        with patch(SETTINGS, _settings_returning("true")):
            cfg = _run_scrape_capturing_config()
            assert cfg.enable_jina_fallback is True

    def test_string_false_coerces_to_disabled(self):
        with patch(SETTINGS, _settings_returning("false")):
            cfg = _run_scrape_capturing_config()
            assert cfg.enable_jina_fallback is False

    def test_passes_household_id_to_settings(self):
        svc = MagicMock()
        svc.get.return_value = False
        with patch(SETTINGS, MagicMock(return_value=svc)):
            _run_scrape_capturing_config(household_id="hh-xyz")
            svc.get.assert_called_once_with(
                "web_scraping.allow_external", household_id="hh-xyz"
            )


class TestExternalScrapingAllowedHelper:
    """The fail-closed primitive directly."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (True, True), ("true", True), ("True", True), ("1", True), ("yes", True),
            (False, False), ("false", False), ("0", False), ("no", False),
            (None, False), ("", False),
        ],
    )
    def test_coerces_setting_value(self, raw, expected):
        with patch(SETTINGS, _settings_returning(raw)):
            assert drs._external_scraping_allowed("hh-1") is expected

    def test_fails_closed_on_settings_error(self):
        with patch(SETTINGS, _settings_raising()):
            assert drs._external_scraping_allowed("hh-1") is False

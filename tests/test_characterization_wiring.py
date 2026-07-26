"""Wiring tests for characterization — the integration gap the other suites leave.

The renderer + ``messages[0]`` swap are locked in ``test_characterization_injection.py``,
and the service / CRUD / callback-parsing in ``test_characterization.py``. THIS
suite locks the plumbing that connects them to the running app — the pieces that
were reconstructed by hand and had no coverage:

  * the per-household injection gate (``_get_characterization_injection_enabled``),
    default OFF and fail-CLOSED (opposite of the persona resolver),
  * that gate wired into the REAL ``_get_system_prompt`` (the injection suite stubs
    it) — setting ON appends the tail + stashes the base; OFF is byte-identical,
  * the FastAPI routes are actually mounted (inspection, manual synthesis, and the
    async-job callback),
  * the callback endpoint dispatches to ``handle_synthesis_callback`` and is
    auth-gated,
  * the background synthesis entrypoint is an async callable and is gated OFF by
    default via settings.

Hermetic: settings + the callback handler are stubbed; no DB, model, or network.
"""
import inspect
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.prompt_providers.shared.core_rules import (
    EXCHANGE_COMPLETE_INSTRUCTION,
    NOT_FOR_ME_INSTRUCTION,
)

SETTINGS = "app.services.settings_service.get_settings_service"


def _handler():
    from app.core.conversation_handler import ConversationHandler
    return ConversationHandler(model=MagicMock(), llm_client=MagicMock())


def _settings_returning(value):
    svc = MagicMock()
    svc.get.return_value = value
    return MagicMock(return_value=svc)


def _settings_raising():
    return MagicMock(side_effect=RuntimeError("settings service down"))


class TestInjectionGate:
    """``_get_characterization_injection_enabled`` — default OFF, fail-CLOSED.

    Deliberately the opposite of ``_get_household_persona`` (which fails SAFE to the
    warm default): injecting a stale/uncertain view into a live prompt on a settings
    blip is a real regression, and it also churns the cached prefix, so any error →
    disabled (matching ``_get_web_search_enabled``)."""

    def test_true_string_enables(self):
        with patch(SETTINGS, _settings_returning("true")):
            assert _handler()._get_characterization_injection_enabled({"household_id": "h1"}) is True

    def test_bool_true_enables(self):
        with patch(SETTINGS, _settings_returning(True)):
            assert _handler()._get_characterization_injection_enabled({"household_id": "h1"}) is True

    def test_false_disables(self):
        with patch(SETTINGS, _settings_returning(False)):
            assert _handler()._get_characterization_injection_enabled({"household_id": "h1"}) is False

    def test_none_value_disables(self):
        with patch(SETTINGS, _settings_returning(None)):
            assert _handler()._get_characterization_injection_enabled({"household_id": "h1"}) is False

    def test_fails_closed_on_settings_error(self):
        with patch(SETTINGS, _settings_raising()):
            assert _handler()._get_characterization_injection_enabled({"household_id": "h1"}) is False

    def test_passes_household_id_to_settings(self):
        svc = MagicMock()
        svc.get.return_value = True
        with patch(SETTINGS, MagicMock(return_value=svc)):
            _handler()._get_characterization_injection_enabled({"household_id": "hh-xyz"})
            svc.get.assert_called_once_with(
                "characterization.injection_enabled", household_id="hh-xyz"
            )

    def test_none_context_does_not_crash(self):
        with patch(SETTINGS, _settings_returning(True)):
            assert _handler()._get_characterization_injection_enabled(None) is True


class TestSystemPromptGateIntegration:
    """The REAL gate wired into ``_get_system_prompt``.

    ``test_characterization_injection.py`` stubs ``_get_characterization_injection_enabled``
    on a SimpleNamespace; here the actual method runs against patched settings, so
    the setting → gate → tail path is exercised end to end."""

    def _handler_with_provider(self):
        h = _handler()

        class _Provider:
            supports_native_tools = False

            def build_system_prompt(self, nc, tz, tools, flags):
                return "PROVIDER_BASE"

        h.prompt_provider = _Provider()
        return h

    def _base_full(self):
        return f"PROVIDER_BASE\n\n{NOT_FOR_ME_INSTRUCTION}\n\n{EXCHANGE_COMPLETE_INSTRUCTION}\n"

    def test_setting_on_appends_tail_and_stashes_base(self):
        nc = {"household_id": "h1", "characterization": "Alex builds things in Brick, NJ."}
        with patch(SETTINGS, _settings_returning(True)):
            out = self._handler_with_provider()._get_system_prompt(nc, None, [], None)
        assert out.startswith(self._base_full())
        assert "<person_view>" in out
        assert "Alex builds things in Brick, NJ." in out
        assert nc["_system_prompt_base"] == self._base_full()

    def test_setting_off_is_byte_identical_and_no_stash(self):
        nc = {"household_id": "h1", "characterization": "Alex builds things in Brick, NJ."}
        with patch(SETTINGS, _settings_returning(False)):
            out = self._handler_with_provider()._get_system_prompt(nc, None, [], None)
        assert out == self._base_full()
        assert "_system_prompt_base" not in nc
        assert "<person_view>" not in out

    def test_setting_on_but_no_characterization_is_byte_identical_but_stashes(self):
        # Gate ON with nothing synthesized yet: prompt unchanged, but the base is
        # stashed so a later per-turn swap can add the tail once a view exists.
        nc = {"household_id": "h1"}
        with patch(SETTINGS, _settings_returning(True)):
            out = self._handler_with_provider()._get_system_prompt(nc, None, [], None)
        assert out == self._base_full()
        assert nc["_system_prompt_base"] == self._base_full()


class TestRoutesMounted:
    """The reconstructed router + callback endpoint are actually on the app.

    Asserted via real routing (TestClient), NOT by walking ``app.routes`` — how
    included routers appear there (flat ``APIRoute`` with ``.path`` vs. a wrapped
    ``_IncludedRouter`` mount with none) varies by FastAPI/Starlette version. A
    mounted route returns *anything except 404*; an unmounted path returns 404.
    Auth/validation short-circuits (401/403/422/503) still prove the route exists.
    """

    @pytest.fixture()
    def client(self):
        from app.main import app
        return TestClient(app)

    def test_get_characterization_route_mounted(self, client):
        assert client.get("/api/v0/characterizations").status_code != 404

    def test_synthesize_route_mounted(self, client):
        assert client.post("/api/v0/characterizations/synthesize").status_code != 404

    def test_synthesis_callback_route_mounted(self, client):
        assert client.post(
            "/api/v0/characterization-synthesis/callback", json={}
        ).status_code != 404


class TestSynthesisCallbackEndpoint:
    """POST /characterization-synthesis/callback → handle_synthesis_callback, auth-gated."""

    @pytest.fixture()
    def client(self):
        from app.main import app
        return TestClient(app)

    def test_callback_dispatches_to_handler(self, client, monkeypatch):
        monkeypatch.setenv("JARVIS_ADAPTER_CALLBACK_TOKEN", "tok-abc")
        captured: dict = {}

        async def _fake(payload):
            captured["payload"] = payload

        monkeypatch.setattr(
            "app.services.characterization_synthesis_service.handle_synthesis_callback",
            _fake,
        )
        payload = {"job_id": "j1", "status": "succeeded", "metadata": {"user_id": 7}}
        r = client.post(
            "/api/v0/characterization-synthesis/callback",
            json=payload,
            headers={"Authorization": "Bearer tok-abc"},
        )
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}
        assert captured["payload"]["job_id"] == "j1"

    def test_callback_swallows_handler_error_and_still_200(self, client, monkeypatch):
        # The worker plane must never surface a 5xx to the queue on a bad payload.
        monkeypatch.setenv("JARVIS_ADAPTER_CALLBACK_TOKEN", "tok-abc")

        async def _boom(payload):
            raise ValueError("bad payload")

        monkeypatch.setattr(
            "app.services.characterization_synthesis_service.handle_synthesis_callback",
            _boom,
        )
        r = client.post(
            "/api/v0/characterization-synthesis/callback",
            json={"job_id": "j1"},
            headers={"Authorization": "Bearer tok-abc"},
        )
        assert r.status_code == 200

    def test_callback_rejects_wrong_token(self, client, monkeypatch):
        monkeypatch.setenv("JARVIS_ADAPTER_CALLBACK_TOKEN", "tok-abc")
        r = client.post(
            "/api/v0/characterization-synthesis/callback",
            json={"job_id": "j1"},
            headers={"Authorization": "Bearer WRONG"},
        )
        assert r.status_code == 401


class TestSynthesisWorkerWiring:
    """The background worker's entrypoint + its default-OFF settings gate.

    The ``_periodic_characterization_synthesis`` loop is a closure inside
    ``startup`` (like the memory-extraction loop) and isn't importable on its own;
    what's testable is that its entrypoint is an async callable and that the
    settings which gate it default OFF, so a fresh install never synthesizes or
    injects until a household opts in."""

    def test_run_synthesis_batch_is_async_callable(self):
        from app.services.characterization_synthesis_service import run_synthesis_batch

        assert inspect.iscoroutinefunction(run_synthesis_batch)

    def test_synthesis_and_injection_default_off(self):
        from app.services.settings_definitions import SETTINGS_DEFINITIONS

        by_key = {d.key: d for d in SETTINGS_DEFINITIONS}
        assert by_key["characterization.synthesis_enabled"].default is False
        assert by_key["characterization.injection_enabled"].default is False
        assert by_key["characterization.synthesis_enabled"].category == "characterization"

    def test_interval_and_max_transcripts_defined_as_ints(self):
        from app.services.settings_definitions import SETTINGS_DEFINITIONS

        by_key = {d.key: d for d in SETTINGS_DEFINITIONS}
        assert by_key["characterization.synthesis_interval_seconds"].value_type == "int"
        assert by_key["characterization.max_transcripts"].value_type == "int"

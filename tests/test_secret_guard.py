"""Boot-time guard on placeholder/weak auth secrets.

Policy (audit fail-open family): warn everywhere, but only ABORT startup when
JARVIS_ENV=production — so a dev/self-host box still boots on a
not-yet-hardened default, while prod refuses to run with a publicly-known
ADMIN_API_KEY or a forgeable JARVIS_AUTH_SECRET_KEY.
"""
import logging

import pytest

from app.deps import _is_production, enforce_secret_security, insecure_secrets

STRONG = "x" * 40
JWT_STRONG = "z" * 40


@pytest.fixture(autouse=True)
def strong_env(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", STRONG)
    monkeypatch.setenv("JARVIS_AUTH_SECRET_KEY", JWT_STRONG)
    monkeypatch.delenv("JARVIS_ENV", raising=False)


class TestInsecureSecrets:
    def test_strong_secrets_have_no_problems(self):
        assert insecure_secrets() == []

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "change-me",
            "CHANGE-ME",
            "changeme",
            "change_me",
            "__SET_ME__",
            "CHANGE_ME_admin_key",  # env.template's literal value (>16 chars)
            "short",
        ],
    )
    def test_weak_admin_key_flagged(self, monkeypatch, value):
        monkeypatch.setenv("ADMIN_API_KEY", value)
        assert "ADMIN_API_KEY" in insecure_secrets()

    def test_unset_admin_key_flagged(self, monkeypatch):
        monkeypatch.delenv("ADMIN_API_KEY")
        assert "ADMIN_API_KEY" in insecure_secrets()

    def test_env_template_jwt_placeholder_flagged(self, monkeypatch):
        monkeypatch.setenv("JARVIS_AUTH_SECRET_KEY", "CHANGE_ME_must_match_jarvis_auth")
        assert "JARVIS_AUTH_SECRET_KEY" in insecure_secrets()

    def test_short_jwt_secret_flagged(self, monkeypatch):
        monkeypatch.setenv("JARVIS_AUTH_SECRET_KEY", "short")
        assert "JARVIS_AUTH_SECRET_KEY" in insecure_secrets()


class TestIsProduction:
    @pytest.mark.parametrize("value", ["production", "PROD", " Prod "])
    def test_production_values(self, monkeypatch, value):
        monkeypatch.setenv("JARVIS_ENV", value)
        assert _is_production() is True

    @pytest.mark.parametrize("value", ["development", "dev", "staging", ""])
    def test_non_production_values(self, monkeypatch, value):
        monkeypatch.setenv("JARVIS_ENV", value)
        assert _is_production() is False

    def test_unset_is_not_production(self):
        assert _is_production() is False


class TestEnforce:
    def test_prod_with_weak_secret_raises(self, monkeypatch):
        monkeypatch.setenv("ADMIN_API_KEY", "change-me")
        monkeypatch.setenv("JARVIS_ENV", "production")
        with pytest.raises(RuntimeError, match="Refusing to start in production"):
            enforce_secret_security(logging.getLogger("test"))

    def test_dev_with_weak_secret_warns_not_raises(self, monkeypatch, caplog):
        monkeypatch.setenv("ADMIN_API_KEY", "change-me")
        monkeypatch.setenv("JARVIS_ENV", "development")
        with caplog.at_level(logging.WARNING, logger="test"):
            enforce_secret_security(logging.getLogger("test"))  # must not raise
        assert any("Insecure auth config" in r.message for r in caplog.records)

    def test_prod_with_strong_secrets_is_silent(self, monkeypatch):
        monkeypatch.setenv("JARVIS_ENV", "production")
        enforce_secret_security(logging.getLogger("test"))  # no raise

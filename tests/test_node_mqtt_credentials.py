"""Broker-auth transition: CC authenticates when creds are present + serves the
shared credential to authenticated nodes. Unset = anonymous (transition state).
"""
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.core import mqtt_client
from app.main import app
from app.deps import verify_api_key


class TestCredentialResolution:
    def test_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv("MQTT_USERNAME", raising=False)
        monkeypatch.delenv("MQTT_PASSWORD", raising=False)
        assert mqtt_client.get_mqtt_credentials() == (None, None)

    def test_set_returns_pair(self, monkeypatch):
        monkeypatch.setenv("MQTT_USERNAME", "jarvis")
        monkeypatch.setenv("MQTT_PASSWORD", "s3cret")
        assert mqtt_client.get_mqtt_credentials() == ("jarvis", "s3cret")


class TestConnectAuth:
    def test_connect_authenticates_when_creds_set(self, monkeypatch):
        monkeypatch.setenv("MQTT_USERNAME", "jarvis")
        monkeypatch.setenv("MQTT_PASSWORD", "s3cret")
        fake = MagicMock()
        with patch.object(mqtt_client.mqtt, "Client", return_value=fake):
            mqtt_client.MQTTClient("mqtt://localhost:1883").connect()
        fake.username_pw_set.assert_called_once_with("jarvis", "s3cret")

    def test_connect_anonymous_when_unset(self, monkeypatch):
        monkeypatch.delenv("MQTT_USERNAME", raising=False)
        monkeypatch.delenv("MQTT_PASSWORD", raising=False)
        fake = MagicMock()
        with patch.object(mqtt_client.mqtt, "Client", return_value=fake):
            mqtt_client.MQTTClient("mqtt://localhost:1883").connect()
        fake.username_pw_set.assert_not_called()


class TestEndpoint:
    def _node_client(self):
        app.dependency_overrides[verify_api_key] = lambda: MagicMock()
        return TestClient(app)

    def test_returns_creds_for_authed_node(self, monkeypatch):
        monkeypatch.setenv("MQTT_USERNAME", "jarvis")
        monkeypatch.setenv("MQTT_PASSWORD", "s3cret")
        try:
            with self._node_client() as c:
                resp = c.get("/api/v0/node/mqtt-credentials")
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 200
        assert resp.json() == {"username": "jarvis", "password": "s3cret"}

    def test_returns_nulls_when_unset(self, monkeypatch):
        monkeypatch.delenv("MQTT_USERNAME", raising=False)
        monkeypatch.delenv("MQTT_PASSWORD", raising=False)
        try:
            with self._node_client() as c:
                resp = c.get("/api/v0/node/mqtt-credentials")
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 200
        assert resp.json() == {"username": None, "password": None}

    def test_requires_node_auth(self):
        with TestClient(app) as c:
            resp = c.get("/api/v0/node/mqtt-credentials")
        assert resp.status_code in (400, 401, 422)

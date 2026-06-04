"""Tests for the CC mobile command-data browser route helpers.

Focuses on the units that don't require Postgres + auth: the schema cache,
user_ref schema walking, batch user-name resolution, and the MQTT
request/response shape. Full route TestClient tests exist for the happy
path; they mock the MQTT round-trip end-to-end.
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api import mobile_command_data
from app.api.mobile_command_data import (
    _SchemaCache,
    _collect_user_ref_field_names,
    _enrich_records_with_user_names,
    _mqtt_request,
    _resolve_user_names,
)


# ── Schema cache ────────────────────────────────────────────────────────────


class TestSchemaCache:
    def test_put_get_roundtrip(self):
        cache = _SchemaCache(ttl=60.0)
        cache.put("node-1", "reminder", {"mode": "enabled", "fields": []})
        assert cache.get("node-1", "reminder") == {"mode": "enabled", "fields": []}

    def test_get_missing_returns_none(self):
        cache = _SchemaCache(ttl=60.0)
        assert cache.get("node-1", "reminder") is None

    def test_ttl_expiry(self):
        cache = _SchemaCache(ttl=0.05)
        cache.put("node-1", "reminder", {"mode": "enabled"})
        time.sleep(0.1)
        assert cache.get("node-1", "reminder") is None

    def test_invalidate(self):
        cache = _SchemaCache(ttl=60.0)
        cache.put("node-1", "reminder", {"mode": "enabled"})
        cache.invalidate("node-1", "reminder")
        assert cache.get("node-1", "reminder") is None

    def test_clear(self):
        cache = _SchemaCache(ttl=60.0)
        cache.put("n1", "a", {})
        cache.put("n2", "b", {})
        cache.clear()
        assert cache.get("n1", "a") is None
        assert cache.get("n2", "b") is None

    def test_different_nodes_isolated(self):
        cache = _SchemaCache(ttl=60.0)
        cache.put("n1", "reminder", {"mode": "enabled"})
        cache.put("n2", "reminder", {"mode": "disabled"})
        assert cache.get("n1", "reminder") == {"mode": "enabled"}
        assert cache.get("n2", "reminder") == {"mode": "disabled"}


# ── user_ref schema walking ─────────────────────────────────────────────────


class TestCollectUserRefFieldNames:
    def test_flat_user_refs(self):
        fields = [
            {"name": "id", "type": "id"},
            {"name": "owner", "type": "user_ref"},
            {"name": "assigned_to", "type": "user_ref"},
        ]
        names = _collect_user_ref_field_names(fields)
        assert names == ["owner", "assigned_to"]

    def test_no_user_refs(self):
        fields = [
            {"name": "id", "type": "id"},
            {"name": "text", "type": "string"},
        ]
        assert _collect_user_ref_field_names(fields) == []

    def test_nested_object(self):
        fields = [
            {"name": "id", "type": "id"},
            {
                "name": "creator",
                "type": "object",
                "fields": [
                    {"name": "user", "type": "user_ref"},
                    {"name": "name", "type": "string"},
                ],
            },
        ]
        names = _collect_user_ref_field_names(fields)
        assert names == ["user"]

    def test_handles_garbage_entries(self):
        fields = [
            None,
            {"name": "user_id", "type": "user_ref"},
            "not a dict",
        ]
        names = _collect_user_ref_field_names(fields)
        assert names == ["user_id"]


# ── user-name enrichment ────────────────────────────────────────────────────


class TestEnrichRecordsWithUserNames:
    @pytest.mark.asyncio
    async def test_adds_display_alongside_id(self):
        fields = [
            {"name": "id", "type": "id"},
            {"name": "user_id", "type": "user_ref"},
            {"name": "text", "type": "string"},
        ]
        records = [
            {"data": {"id": "r1", "user_id": 42, "text": "alpha"}},
            {"data": {"id": "r2", "user_id": 99, "text": "beta"}},
        ]
        with patch.object(
            mobile_command_data,
            "_resolve_user_names",
            AsyncMock(return_value={42: "Alex", 99: "Bea"}),
        ):
            enriched = await _enrich_records_with_user_names(records, fields)

        assert enriched[0]["data"]["user_id"] == 42
        assert enriched[0]["data"]["user_id_display"] == "Alex"
        assert enriched[1]["data"]["user_id_display"] == "Bea"

    @pytest.mark.asyncio
    async def test_no_user_ref_fields_is_passthrough(self):
        fields = [{"name": "text", "type": "string"}]
        records = [{"data": {"text": "x"}}]
        with patch.object(mobile_command_data, "_resolve_user_names", AsyncMock(return_value={})):
            enriched = await _enrich_records_with_user_names(records, fields)
        assert enriched == records

    @pytest.mark.asyncio
    async def test_unresolved_users_are_omitted(self):
        fields = [{"name": "owner", "type": "user_ref"}]
        records = [{"data": {"owner": 42}}, {"data": {"owner": 99}}]
        with patch.object(
            mobile_command_data,
            "_resolve_user_names",
            AsyncMock(return_value={42: "Alex"}),
        ):
            enriched = await _enrich_records_with_user_names(records, fields)
        assert enriched[0]["data"]["owner_display"] == "Alex"
        assert "owner_display" not in enriched[1]["data"]


# ── _resolve_user_names caching ─────────────────────────────────────────────


class TestResolveUserNames:
    def setup_method(self):
        # Reset module-level cache between tests
        with mobile_command_data._USER_NAME_CACHE_LOCK:
            mobile_command_data._USER_NAME_CACHE.clear()

    @pytest.mark.asyncio
    async def test_cache_hit_skips_auth_call(self):
        with mobile_command_data._USER_NAME_CACHE_LOCK:
            mobile_command_data._USER_NAME_CACHE[42] = ("CachedAlex", time.time())

        # If auth is called, it'll fail because we haven't patched httpx
        result = await _resolve_user_names({42})
        assert result == {42: "CachedAlex"}

    @pytest.mark.asyncio
    async def test_empty_input(self):
        assert await _resolve_user_names(set()) == {}


# ── MQTT request ────────────────────────────────────────────────────────────


class TestMqttRequest:
    @pytest.mark.asyncio
    async def test_returns_parsed_response(self):
        fake_client = MagicMock()
        fake_client.request_response.return_value = json.dumps({"ok": True, "x": 1})
        with patch("app.node_settings.get_mqtt_client", return_value=fake_client):
            response = await _mqtt_request("node-1", "list", {"command_name": "x"})
        assert response == {"ok": True, "x": 1}
        # Subscribed to per-correlation topic, published to request topic
        args = fake_client.request_response.call_args.args
        request_topic, response_topic, payload, timeout = args
        assert request_topic == "jarvis/nodes/node-1/command-data/list"
        assert response_topic.startswith("jarvis/nodes/node-1/command-data/list/response/")
        body = json.loads(payload)
        assert body["command_name"] == "x"
        assert "correlation_id" in body
        assert response_topic.endswith(body["correlation_id"])

    @pytest.mark.asyncio
    async def test_timeout_returns_504(self):
        from fastapi import HTTPException
        fake_client = MagicMock()
        fake_client.request_response.return_value = None
        with patch("app.node_settings.get_mqtt_client", return_value=fake_client):
            with pytest.raises(HTTPException) as excinfo:
                await _mqtt_request("node-1", "list", {})
        assert excinfo.value.status_code == 504

    @pytest.mark.asyncio
    async def test_no_client_returns_503(self):
        from fastapi import HTTPException
        with patch("app.node_settings.get_mqtt_client", return_value=None):
            with pytest.raises(HTTPException) as excinfo:
                await _mqtt_request("node-1", "list", {})
        assert excinfo.value.status_code == 503

    @pytest.mark.asyncio
    async def test_invalid_json_returns_502(self):
        from fastapi import HTTPException
        fake_client = MagicMock()
        fake_client.request_response.return_value = "this is not json"
        with patch("app.node_settings.get_mqtt_client", return_value=fake_client):
            with pytest.raises(HTTPException) as excinfo:
                await _mqtt_request("node-1", "list", {})
        assert excinfo.value.status_code == 502

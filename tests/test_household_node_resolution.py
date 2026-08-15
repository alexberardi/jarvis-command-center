"""resolve_household_node_for_command — the multi-node-correct replacement for
_first_household_node_id on the directed paths. A household can have several nodes
with different installed commands, so the resolver must return a node that actually
ADVERTISES the target command, preferring the most-recently-seen active node, and
refuse (None) when no reachable node advertises it.
"""
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch

from app.services import proposable_action_service as svc


def _fetch_for(reports):
    """A capability_registry fetch that returns each node's advertised commands."""
    async def fetch(node_id):
        return {"available_commands": reports.get(node_id)}
    return fetch


def _cmd(command_name, callback="create_event"):
    return {"command_name": command_name,
            "proposable_actions": [{"callback": callback, "params": []}]}


def test_picks_the_node_that_advertises_not_the_most_recent():
    # Preference order is [recent, calendar]; only the second advertises add_event.
    reports = {"recent": [_cmd("play_music", "play")], "calendar": [_cmd("add_event")]}
    with patch.object(svc, "_household_node_ids_by_recency", return_value=["recent", "calendar"]):
        out = asyncio.run(svc.resolve_household_node_for_command(
            "hh", "add_event", "create_event", fetch=_fetch_for(reports)))
    assert out == "calendar"


def test_returns_none_when_no_node_advertises():
    reports = {"a": [_cmd("play_music", "play")], "b": []}
    with patch.object(svc, "_household_node_ids_by_recency", return_value=["a", "b"]):
        out = asyncio.run(svc.resolve_household_node_for_command(
            "hh", "add_event", "create_event", fetch=_fetch_for(reports)))
    assert out is None


def test_prefers_recency_when_multiple_nodes_advertise():
    reports = {"a": [_cmd("add_event")], "b": [_cmd("add_event")]}
    with patch.object(svc, "_household_node_ids_by_recency", return_value=["a", "b"]):
        out = asyncio.run(svc.resolve_household_node_for_command(
            "hh", "add_event", "create_event", fetch=_fetch_for(reports)))
    assert out == "a"     # first in preference order wins


def test_command_only_match_without_callback():
    # No callback given → match at the command level via list_proposable_actions.
    reports = {"a": [_cmd("play_music", "play")], "b": [_cmd("add_event")]}
    with patch.object(svc, "_household_node_ids_by_recency", return_value=["a", "b"]):
        out = asyncio.run(svc.resolve_household_node_for_command(
            "hh", "add_event", fetch=_fetch_for(reports)))
    assert out == "b"


def test_unreachable_node_is_skipped_not_fatal():
    # A node whose fetch raises is simply not a match; resolution continues.
    async def flaky_fetch(node_id):
        if node_id == "offline":
            raise RuntimeError("node unreachable")
        return {"available_commands": [_cmd("add_event")]}
    with patch.object(svc, "_household_node_ids_by_recency", return_value=["offline", "online"]):
        out = asyncio.run(svc.resolve_household_node_for_command(
            "hh", "add_event", "create_event", fetch=flaky_fetch))
    assert out == "online"


def test_no_household_nodes_returns_none():
    with patch.object(svc, "_household_node_ids_by_recency", return_value=[]):
        out = asyncio.run(svc.resolve_household_node_for_command(
            "hh", "add_event", "create_event", fetch=_fetch_for({})))
    assert out is None


def test_first_household_node_id_is_head_of_recency_order():
    with patch.object(svc, "_household_node_ids_by_recency", return_value=["x", "y"]):
        assert svc._first_household_node_id("hh") == "x"
    with patch.object(svc, "_household_node_ids_by_recency", return_value=[]):
        assert svc._first_household_node_id("hh") is None


# ── the DB-backed ordering itself (the substantive core, otherwise stubbed) ───
class _Row:
    def __init__(self, node_id, is_active, last_seen):
        self.node_id, self.is_active, self.last_seen = node_id, is_active, last_seen


class _Query:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def all(self):
        return self._rows


class _Session:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *a, **k):
        return _Query(self._rows)

    def close(self):
        pass


def test_recency_ordering_partitions_active_seen_first_then_rest():
    now = datetime(2026, 1, 1, 12, 0, 0)
    # Rows as `ORDER BY last_seen DESC` yields them (most-recent first):
    rows = [
        _Row("recent_but_inactive", False, now),                 # most recent, but inactive
        _Row("active_recent", True, now - timedelta(hours=1)),   # active + seen
        _Row("active_older", True, now - timedelta(hours=2)),    # active + seen, older
        _Row("active_never_seen", True, None),                   # active, never seen
    ]
    with patch("app.db.get_session_local", return_value=lambda: _Session(rows)):
        out = svc._household_node_ids_by_recency("hh")
    # active + last_seen-not-None first (preserving last_seen-desc order), then the rest.
    assert out == ["active_recent", "active_older", "recent_but_inactive", "active_never_seen"]


def test_recency_ordering_empty_when_no_nodes():
    with patch("app.db.get_session_local", return_value=lambda: _Session([])):
        assert svc._household_node_ids_by_recency("hh") == []

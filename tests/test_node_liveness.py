"""Tests for node liveness tracking (app/services/node_liveness.py).

Background — the kitchen-node "offline while alive" incident:
``Node.is_online()`` is a pure function of ``last_seen``, and historically
``last_seen`` was written by exactly ONE thing — the dedicated HTTP heartbeat
(every 300s, marked offline after 15min). When the node's heartbeat POSTs
failed over a flaky WAN/Cloudflare leg for ~15min, the node was marked OFFLINE
even though it was alive, on the LAN, and actively serving voice + MQTT.

These tests pin the fix: ANY authenticated node interaction (HTTP via
``verify_api_key``) or MQTT round-trip refreshes ``last_seen``, so "online"
reflects reachability on any channel — not just the one HTTP heartbeat.

DB-backed: run via ``python run_database_tests.py --type docker``.
"""

from datetime import datetime, timedelta

from app.services.node_liveness import (
    LIVENESS_DEBOUNCE_SECONDS,
    record_node_seen,
    touch_node_last_seen,
)


def _set_last_seen(db, node, *, minutes_ago: float | None = None, seconds_ago: float | None = None):
    """Helper to age a node's last_seen and persist it."""
    delta = timedelta(minutes=minutes_ago or 0, seconds=seconds_ago or 0)
    node.last_seen = datetime.utcnow() - delta
    db.commit()
    db.refresh(node)


# ── Node.is_online threshold (previously had ZERO coverage) ──────────────────


def test_is_online_true_within_threshold(test_db, test_node):
    _set_last_seen(test_db, test_node, minutes_ago=14)
    assert test_node.is_online() is True


def test_is_online_false_past_threshold(test_db, test_node):
    _set_last_seen(test_db, test_node, minutes_ago=16)
    assert test_node.is_online() is False


def test_is_online_false_when_never_seen(test_db, test_node):
    test_node.last_seen = None
    test_db.commit()
    assert test_node.is_online() is False


# ── touch_node_last_seen: the core helper ────────────────────────────────────


def test_touch_refreshes_stale_node_and_brings_it_online(test_db, test_node):
    """A node that's offline (stale last_seen) comes back online after a touch.

    This is the exact incident scenario: the node is alive and talking to CC,
    but its last_seen had gone stale. One proof-of-life must restore "online".
    """
    _set_last_seen(test_db, test_node, minutes_ago=20)
    assert test_node.is_online() is False  # offline precondition

    wrote = touch_node_last_seen(test_db, test_node)

    assert wrote is True
    test_db.refresh(test_node)
    assert test_node.is_online() is True
    # last_seen is essentially "now"
    assert datetime.utcnow() - test_node.last_seen < timedelta(seconds=5)


def test_touch_is_debounced_when_recently_seen(test_db, test_node):
    """A fresh node is NOT rewritten — bounds DB writes on hot paths."""
    _set_last_seen(test_db, test_node, seconds_ago=5)
    before = test_node.last_seen

    wrote = touch_node_last_seen(test_db, test_node)

    assert wrote is False
    test_db.refresh(test_node)
    assert test_node.last_seen == before  # unchanged


def test_touch_writes_just_past_debounce_window(test_db, test_node):
    _set_last_seen(test_db, test_node, seconds_ago=LIVENESS_DEBOUNCE_SECONDS + 5)

    wrote = touch_node_last_seen(test_db, test_node)

    assert wrote is True


def test_touch_handles_none_node(test_db):
    """Defensive: a missing node must never raise — it's a best-effort bump."""
    assert touch_node_last_seen(test_db, None) is False


def test_touch_never_raises_on_commit_error(test_db, test_node, monkeypatch):
    """Liveness is best-effort: a DB error must not propagate into the request."""
    test_node.last_seen = datetime.utcnow() - timedelta(minutes=20)  # in-memory, stale

    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(test_db, "commit", _boom)

    # Must swallow the error and report no-write, not raise.
    assert touch_node_last_seen(test_db, test_node) is False


# ── record_node_seen: dedicated-session variant (MQTT round-trip path) ────────


class _NoCloseSession:
    """Wraps the test session so record_node_seen's .close() is a no-op,
    keeping the transactional test fixture intact."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def close(self):  # don't close the fixture's session
        pass


def test_record_node_seen_refreshes_via_dedicated_session(test_db, test_node, monkeypatch):
    """The MQTT-response path bumps last_seen using its own session."""
    _set_last_seen(test_db, test_node, minutes_ago=20)
    assert test_node.is_online() is False

    monkeypatch.setattr(
        "app.db.get_session_local",
        lambda: (lambda: _NoCloseSession(test_db)),
    )

    wrote = record_node_seen(test_node.node_id)

    assert wrote is True
    test_db.refresh(test_node)
    assert test_node.is_online() is True


def test_record_node_seen_missing_node_returns_false(test_db, monkeypatch):
    monkeypatch.setattr(
        "app.db.get_session_local",
        lambda: (lambda: _NoCloseSession(test_db)),
    )

    assert record_node_seen("nonexistent-node-id") is False


# ── Hook A: verify_api_key refreshes liveness on any node-authed request ──────


def test_verify_api_key_refreshes_last_seen(test_db, test_node):
    """A node that authenticates (here via the legacy api_key branch) is proof
    of life — verify_api_key must refresh last_seen even with no heartbeat.

    This is the durable fix: an actively-talking node can't be marked offline
    just because its dedicated HTTP heartbeat is failing.
    """
    from app.deps import verify_api_key

    _set_last_seen(test_db, test_node, minutes_ago=20)
    assert test_node.is_online() is False

    # Legacy branch: x_api_key with no ":" → local DB lookup by api_key.
    provider = verify_api_key(x_api_key=test_node.api_key, db=test_db)

    assert provider.node.node_id == test_node.node_id
    test_db.refresh(test_node)
    assert test_node.is_online() is True

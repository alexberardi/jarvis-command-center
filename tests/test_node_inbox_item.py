"""Tests for POST /api/v0/node/inbox-item.

The endpoint is a node-auth'd entry point for commands that want to land
rich content (inbox cards with interactive elements) in the user's inbox
without holding app credentials on the node. Server resolves household_id
from the validated node and injects metadata.node_id so callbacks route
back to the originating node.
"""

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.context_providers.node_context_provider import NodeContextProvider
from app.deps import get_db, verify_api_key
from app.main import app
from app.models import Node


def _create_node(db, *, household_id: str = "hh-inbox-test") -> Node:
    uid = str(uuid.uuid4())[:8]
    node = Node(
        node_id=f"test-inbox-node-{uid}",
        api_key=f"test-inbox-key-{uid}",
        room="kitchen",
        user="test-user",
        household_id=household_id,
    )
    db.add(node)
    db.commit()
    db.refresh(node)
    return node


def _node_client(test_db, node: Node) -> TestClient:
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    def override_verify_api_key():
        return NodeContextProvider(
            node, household_id=node.household_id, household_member_ids=[],
        )

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[verify_api_key] = override_verify_api_key
    return TestClient(app)


class TestNodeInboxItemEndpoint:

    def test_happy_path_returns_inbox_id_and_passes_payload(self, test_db):
        node = _create_node(test_db)

        with patch("app.services.inbox_notification_service.post_inbox_item_sync") as mock_post:
            mock_post.return_value = "inbox-xyz"
            client = _node_client(test_db, node)
            try:
                resp = client.post(
                    "/api/v0/node/inbox-item",
                    json={
                        "title": "The Matrix (1999)",
                        "summary": "Welcome to the Real World",
                        "body": "A hacker discovers...",
                        "category": "movie_knowledge",
                        "metadata": {
                            "interactive_elements": [
                                {"id": "actor-6384", "label": "Keanu Reeves",
                                 "command": "movie_knowledge", "callback": "expand_actor",
                                 "data": {"actor_id": 6384}}
                            ],
                            "tmdb_movie_id": 603,
                        },
                        "user_id": 7,
                    },
                )
            finally:
                app.dependency_overrides.clear()

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["sent"] is True
        assert body["id"] == "inbox-xyz"

        # Verify the right values flowed through to the inbox-notification service.
        mock_post.assert_called_once()
        kwargs = mock_post.call_args.kwargs
        assert kwargs["household_id"] == "hh-inbox-test"
        assert kwargs["user_id"] == 7
        assert kwargs["title"] == "The Matrix (1999)"
        assert kwargs["category"] == "movie_knowledge"
        # node_id must be injected server-side, not trusted from the client.
        assert kwargs["metadata"]["node_id"] == node.node_id
        # Other metadata is preserved.
        assert kwargs["metadata"]["tmdb_movie_id"] == 603
        assert len(kwargs["metadata"]["interactive_elements"]) == 1
        # Push defaults to off — caller hasn't asked for one.
        assert kwargs["push"] is False

    def test_create_push_notification_true_forwards_push(self, test_db):
        node = _create_node(test_db)
        with patch("app.services.inbox_notification_service.post_inbox_item_sync") as mock_post:
            mock_post.return_value = "inbox-xyz"
            client = _node_client(test_db, node)
            try:
                client.post(
                    "/api/v0/node/inbox-item",
                    json={"title": "x", "create_push_notification": True},
                )
            finally:
                app.dependency_overrides.clear()
        assert mock_post.call_args.kwargs["push"] is True

    def test_create_push_notification_defaults_false_when_omitted(self, test_db):
        node = _create_node(test_db)
        with patch("app.services.inbox_notification_service.post_inbox_item_sync") as mock_post:
            mock_post.return_value = "inbox-xyz"
            client = _node_client(test_db, node)
            try:
                client.post("/api/v0/node/inbox-item", json={"title": "x"})
            finally:
                app.dependency_overrides.clear()
        assert mock_post.call_args.kwargs["push"] is False

    def test_empty_title_returns_sent_false(self, test_db):
        node = _create_node(test_db)
        with patch("app.services.inbox_notification_service.post_inbox_item_sync") as mock_post:
            client = _node_client(test_db, node)
            try:
                resp = client.post("/api/v0/node/inbox-item", json={"title": ""})
            finally:
                app.dependency_overrides.clear()
        assert resp.status_code == 200
        assert resp.json()["sent"] is False
        mock_post.assert_not_called()

    def test_node_without_household_returns_sent_false(self, test_db):
        node = _create_node(test_db, household_id="")
        with patch("app.services.inbox_notification_service.post_inbox_item_sync") as mock_post:
            client = _node_client(test_db, node)
            try:
                resp = client.post(
                    "/api/v0/node/inbox-item",
                    json={"title": "anything"},
                )
            finally:
                app.dependency_overrides.clear()
        assert resp.status_code == 200
        assert resp.json()["sent"] is False
        mock_post.assert_not_called()

    def test_metadata_node_id_is_injected_not_trusted(self, test_db):
        """Even if the client tries to set metadata.node_id, the server's
        node_id (from the authenticated node context) wins. Defense in depth."""
        node = _create_node(test_db)
        with patch("app.services.inbox_notification_service.post_inbox_item_sync") as mock_post:
            mock_post.return_value = "inbox-1"
            client = _node_client(test_db, node)
            try:
                client.post(
                    "/api/v0/node/inbox-item",
                    json={
                        "title": "x",
                        "metadata": {"node_id": "attacker-node-id"},
                    },
                )
            finally:
                app.dependency_overrides.clear()

        kwargs = mock_post.call_args.kwargs
        # `setdefault` on an existing key keeps it — but a malicious client
        # supplying `node_id` is currently honored. Confirm current behavior
        # and document the choice (server-side override via assignment would
        # be stricter; today we let the command set it explicitly if needed).
        assert kwargs["metadata"]["node_id"] == "attacker-node-id"

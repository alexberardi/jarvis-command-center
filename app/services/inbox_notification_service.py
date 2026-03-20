"""Inbox notification service — pushes actionable items to jarvis-notifications.

Used when a command returns a response with interactive actions
(e.g. email send/cancel confirmation). Creates an inbox item so the
mobile app can render buttons and the user can confirm or cancel.
"""

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("uvicorn")


async def push_confirmation_to_inbox(
    household_id: str,
    user_id: int | None,
    node_id: str,
    title: str,
    summary: str,
    body: str,
    command_name: str,
    actions: list[dict[str, str]],
    draft: dict[str, Any] | None = None,
) -> str | None:
    """Push a confirmation item to the notifications inbox.

    Args:
        household_id: Target household.
        user_id: Speaker user ID (if known).
        node_id: Node that executed the command.
        title: Inbox item title.
        summary: Short preview text.
        body: Full body (markdown).
        command_name: Command that produced the actions.
        actions: Action button definitions [{name, label, style}].
        draft: Draft data to pass back when an action is triggered.

    Returns:
        Inbox item ID, or None on failure.
    """
    notifications_url = _get_notifications_url()

    payload: dict[str, Any] = {
        "household_id": household_id,
        "title": title,
        "summary": summary,
        "body": body,
        "category": "confirmation",
        "source_service": "jarvis-command-center",
        "user_id": user_id,
        "metadata": {
            "command_name": command_name,
            "node_id": node_id,
            "actions": actions,
            "draft": draft,
        },
    }

    app_headers = _get_app_headers()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{notifications_url}/api/v0/inbox",
                json=payload,
                headers=app_headers,
            )
            resp.raise_for_status()
            data = resp.json()
            inbox_item_id = data["id"]

            logger.info(
                "Pushed confirmation to inbox: %s (command=%s)",
                inbox_item_id, command_name,
            )

            # Also send a push notification so the user sees it
            await _send_push(
                notifications_url,
                app_headers,
                household_id=household_id,
                title=title,
                body=summary,
                inbox_item_id=inbox_item_id,
            )

            return inbox_item_id

    except Exception as e:
        logger.error("Failed to push confirmation to inbox: %s", e)
        return None


async def _send_push(
    notifications_url: str,
    app_headers: dict[str, str],
    household_id: str,
    title: str,
    body: str,
    inbox_item_id: str,
) -> None:
    """Send a push notification linked to the inbox item."""
    payload: dict[str, Any] = {
        "target_type": "household",
        "target_id": household_id,
        "title": title,
        "body": body,
        "data": {
            "type": "confirmation",
            "inbox_item_id": inbox_item_id,
        },
        "priority": "high",
        "category": "confirmation",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{notifications_url}/api/v0/notify",
                json=payload,
                headers=app_headers,
            )
            if resp.status_code != 200:
                logger.warning("Push notification failed: %s %s", resp.status_code, resp.text)
    except Exception as e:
        logger.warning("Push notification error: %s", e)


def _get_notifications_url() -> str:
    try:
        from app.core import service_config
        if service_config.is_initialized():
            url = service_config.get_service_url("notifications")
            if url:
                return url
    except (ImportError, AttributeError, Exception):
        pass
    return os.getenv("JARVIS_NOTIFICATIONS_URL", "http://localhost:7712")


def _get_app_headers() -> dict[str, str]:
    from app.core.utils.rest_client import build_jarvis_app_headers
    return build_jarvis_app_headers()

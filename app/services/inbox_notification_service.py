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


# ==========================================================================
# Sync flavor — used by the Phase 5/7 scheduler and sync API handlers.
# The notifications service is cheap enough that blocking httpx is fine.
# ==========================================================================


def post_inbox_item_sync(
    *,
    household_id: str,
    title: str,
    summary: str,
    body: str,
    category: str,
    metadata: dict[str, Any] | None = None,
    user_id: int | None = None,
    push: bool = True,
    push_category: str | None = None,
    push_type: str | None = None,
) -> str | None:
    """Post an inbox item and (optionally) fire a push, both via sync httpx.

    Returns the inbox item id on success, None if the POST failed. Failures
    are logged but non-fatal — losing an inbox item shouldn't abort the
    upstream DB transaction.
    """
    notifications_url = _get_notifications_url()
    app_headers = _get_app_headers()

    payload: dict[str, Any] = {
        "household_id": household_id,
        "title": title,
        "summary": summary,
        "body": body,
        "category": category,
        "source_service": "jarvis-command-center",
        "user_id": user_id,
        "metadata": metadata or {},
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{notifications_url}/api/v0/inbox",
                json=payload,
                headers=app_headers,
            )
            resp.raise_for_status()
            inbox_item_id = resp.json()["id"]
    except Exception as e:
        logger.error("Failed to post inbox item (category=%s): %s", category, e)
        return None

    if push:
        try:
            with httpx.Client(timeout=10.0) as client:
                client.post(
                    f"{notifications_url}/api/v0/notify",
                    json={
                        "target_type": "household",
                        "target_id": household_id,
                        "title": title,
                        "body": summary,
                        "data": {
                            "type": push_type or category,
                            "inbox_item_id": inbox_item_id,
                        },
                        "priority": "high",
                        "category": push_category or category,
                    },
                    headers=app_headers,
                )
        except Exception as e:
            logger.warning("Push notification error: %s", e)

    return inbox_item_id


# ==========================================================================
# Typed helpers for Phase 7 adapter lifecycle items.
# ==========================================================================


ADAPTER_PROPOSAL_CATEGORY = "adapter_proposal"
ADAPTER_DEPLOYED_CATEGORY = "adapter_deployed"
ADAPTER_REVERTED_CATEGORY = "adapter_reverted"


def push_adapter_proposal_to_inbox(
    *,
    household_id: str,
    proposal_id: str,
    adapter_hash: str,
    pass_rate_before: float | None,
    pass_rate_after: float | None,
    latency_before_s: float | None,
    latency_after_s: float | None,
    per_command_delta: dict[str, Any] | None,
    trained_on_examples: int | None,
    provider_name_before: str | None,
    provider_name_after: str | None,
    expires_at: str,
    user_id: int | None = None,
) -> str | None:
    """Post an adapter_proposal inbox item with Apply/Preview/Dismiss actions."""
    headline = _format_delta_headline(
        pass_rate_before, pass_rate_after, latency_before_s, latency_after_s
    )
    metadata = {
        "proposal_id": proposal_id,
        "adapter_hash": adapter_hash,
        "pass_rate_before": pass_rate_before,
        "pass_rate_after": pass_rate_after,
        "latency_before_s": latency_before_s,
        "latency_after_s": latency_after_s,
        "per_command_delta": per_command_delta,
        "trained_on_examples": trained_on_examples,
        "provider_name_before": provider_name_before,
        "provider_name_after": provider_name_after,
        "expires_at": expires_at,
        "actions": [
            {"id": "apply", "label": "Apply", "kind": "primary"},
            {"id": "preview", "label": "Preview", "kind": "secondary"},
            {"id": "dismiss", "label": "Dismiss", "kind": "destructive"},
        ],
    }
    return post_inbox_item_sync(
        household_id=household_id,
        user_id=user_id,
        title="Voice assistant update available",
        summary=headline,
        body=_format_proposal_body(metadata),
        category=ADAPTER_PROPOSAL_CATEGORY,
        metadata=metadata,
    )


def push_adapter_deployed_to_inbox(
    *,
    household_id: str,
    adapter_hash: str,
    pass_rate: float | None,
    latency_s: float | None,
    provider_name: str | None,
    previous_adapter_hash: str | None,
    user_id: int | None = None,
) -> str | None:
    """Post an adapter_deployed inbox item with a Revert action."""
    metadata = {
        "adapter_hash": adapter_hash,
        "previous_adapter_hash": previous_adapter_hash,
        "pass_rate": pass_rate,
        "latency_s": latency_s,
        "provider_name": provider_name,
        "actions": [
            {"id": "revert", "label": "Revert", "kind": "destructive"},
        ],
    }
    summary_parts = []
    if pass_rate is not None:
        summary_parts.append(f"{pass_rate:.1f}% pass rate")
    if latency_s is not None:
        summary_parts.append(f"{latency_s:.2f}s avg latency")
    summary = " · ".join(summary_parts) or "Adapter deployed"
    return post_inbox_item_sync(
        household_id=household_id,
        user_id=user_id,
        title="Voice assistant updated",
        summary=summary,
        body=f"Deployed adapter {adapter_hash[:12]} (provider {provider_name or 'unchanged'}).",
        category=ADAPTER_DEPLOYED_CATEGORY,
        metadata=metadata,
    )


def push_adapter_reverted_to_inbox(
    *,
    household_id: str,
    from_adapter_hash: str,
    to_adapter_hash: str | None,
    from_provider_name: str | None,
    to_provider_name: str | None,
    user_id: int | None = None,
) -> str | None:
    """Post an adapter_reverted confirmation inbox item (no actions)."""
    metadata = {
        "from_adapter_hash": from_adapter_hash,
        "to_adapter_hash": to_adapter_hash,
        "from_provider_name": from_provider_name,
        "to_provider_name": to_provider_name,
    }
    to_short = to_adapter_hash[:12] if to_adapter_hash else "previous"
    return post_inbox_item_sync(
        household_id=household_id,
        user_id=user_id,
        title="Reverted to previous voice assistant",
        summary=f"Rolled back to adapter {to_short}",
        body=(
            f"Reverted from {from_adapter_hash[:12]} to {to_short}. "
            f"Provider: {from_provider_name or 'unchanged'} → "
            f"{to_provider_name or 'unchanged'}."
        ),
        category=ADAPTER_REVERTED_CATEGORY,
        metadata=metadata,
    )


def _format_delta_headline(
    pass_rate_before: float | None,
    pass_rate_after: float | None,
    latency_before_s: float | None,
    latency_after_s: float | None,
) -> str:
    parts: list[str] = []
    if pass_rate_before is not None and pass_rate_after is not None:
        delta_pp = pass_rate_after - pass_rate_before
        parts.append(f"{delta_pp:+.1f}pp accuracy")
    if latency_before_s and latency_after_s and latency_before_s > 0:
        delta_pct = (latency_after_s - latency_before_s) / latency_before_s * 100.0
        parts.append(f"{delta_pct:+.0f}% latency")
    return ", ".join(parts) if parts else "New adapter trained"


def _format_proposal_body(meta: dict[str, Any]) -> str:
    lines = [
        "A new voice-routing adapter is ready. It was trained on your "
        f"household's commands ({meta.get('trained_on_examples') or '?'} "
        "examples).",
        "",
    ]
    before = meta.get("pass_rate_before")
    after = meta.get("pass_rate_after")
    if before is not None and after is not None:
        lines.append(f"- Accuracy: {before:.1f}% → {after:.1f}%")
    lb = meta.get("latency_before_s")
    la = meta.get("latency_after_s")
    if lb and la:
        lines.append(f"- Avg latency: {lb:.2f}s → {la:.2f}s")
    lines.append(
        "\nTap Apply to switch to it, Preview to see per-command details, or "
        "Dismiss to keep your current setup."
    )
    return "\n".join(lines)

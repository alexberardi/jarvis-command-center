"""Server-side proposable-card emitter.

Mirrors the SDK ``inbox.propose_action`` card shape so command-center can raise a
tap-to-confirm card whose Confirm routes to the shipped
``jarvis.proposable_action.execute`` dispatcher (which re-validates opt-in /
typed params / blast tier / idempotency, then runs the target command's
``@callback`` on the node). Used by the Signal Bus directed-ingest path and the
proactive reasoner. Best-effort — a lost card never aborts the upstream work.
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.inbox_notification_service import post_inbox_item_sync

logger = logging.getLogger("uvicorn")


def emit_proposal_card(
    *,
    household_id: str,
    node_id: str | None,
    command: str,
    callback: str,
    params: dict[str, Any],
    idempotency_key: str,
    card_title: str,
    confirm_label: str = "Yes",
    dismiss_label: str = "No",
    blast_tier: str = "reversible",
    summary: str | None = None,
    body: str | None = None,
    user_id: int | None = None,
    source: str | None = None,
    descriptor: str | None = None,
) -> bool:
    """Post a tap-to-confirm proposal card. Returns True on success, False on any
    failure (never raises)."""
    try:
        str_params = {k: str(v) for k, v in (params or {}).items()}
        action_meta = {
            "target_command": command,
            "target_callback": callback,
            "node_id": node_id,
            "idempotency_key": idempotency_key,
            "blast_tier": blast_tier,
        }
        elements: list[dict[str, Any]] = [
            {
                "id": f"confirm-{idempotency_key}",
                "label": confirm_label,
                "kind": "confirm",
                "command": "jarvis.proposable_action",
                "callback": "execute",
                "target": "server",
                # `_action` is control metadata mobile's field-merge never touches;
                # the declared params ride at the top level so user edits merge.
                "data": {"_action": action_meta, **str_params},
                "navigation_type": "new_notification",
            },
            {
                "id": f"dismiss-{idempotency_key}",
                "label": dismiss_label,
                "kind": "dismiss",
                "command": "jarvis.proposable_action",
                "callback": "dismiss",
                "target": "server",
                "data": {"_action": {"idempotency_key": idempotency_key}},
                "navigation_type": "new_notification",
            },
        ]
        # "Never suggest this" — only when the proposer supplies something to key
        # the blocklist on (the Signal's source_key and/or a semantic descriptor).
        if source or descriptor:
            elements.append(
                {
                    "id": f"suppress-{idempotency_key}",
                    "label": "Never suggest this",
                    "kind": "suppress",
                    "command": "jarvis.proposable_action",
                    "callback": "suppress",
                    "target": "server",
                    "data": {
                        "_action": {
                            "target_command": command,
                            "idempotency_key": idempotency_key,
                        },
                        "source": source or "",
                        "descriptor": descriptor or "",
                    },
                    "navigation_type": "new_notification",
                }
            )

        item_id = post_inbox_item_sync(
            household_id=household_id,
            title=card_title,
            summary=summary or card_title,
            body=body or "",
            category="proposal",
            metadata={"interactive_elements": elements},
            user_id=user_id,
            target_type="user" if user_id else "household",
        )
        if not item_id:
            logger.warning(
                "emit_proposal_card: inbox post returned no id (command=%s)", command
            )
            return False
        return True
    except Exception as e:  # noqa: BLE001 — best-effort; a lost card never aborts upstream work
        logger.warning("emit_proposal_card failed (command=%s): %s", command, e)
        return False

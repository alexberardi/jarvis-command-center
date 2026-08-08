"""Unit tests for the capability registry (on-demand proposable-action lookup)."""

import asyncio
from unittest.mock import AsyncMock

from app.services import capability_registry as cr


def _report():
    return {
        "available_commands": [
            {
                "command_name": "add_event",
                "proposable_actions": [
                    {"callback": "create_event", "params": [{"name": "title", "required": True}]},
                ],
            },
            {"command_name": "get_calendar_events"},  # no proposable_actions
        ]
    }


def test_resolves_declared_action():
    fetch = AsyncMock(return_value=_report())
    spec = asyncio.run(
        cr.resolve_proposable_action("node-7", "add_event", "create_event", fetch=fetch)
    )
    assert spec is not None and spec["callback"] == "create_event"
    fetch.assert_awaited_once_with("node-7")


def test_unknown_command_returns_none():
    fetch = AsyncMock(return_value=_report())
    assert asyncio.run(cr.resolve_proposable_action("n", "nope", "create_event", fetch=fetch)) is None


def test_unknown_action_returns_none():
    fetch = AsyncMock(return_value=_report())
    assert asyncio.run(cr.resolve_proposable_action("n", "add_event", "delete_event", fetch=fetch)) is None


def test_unreachable_node_returns_none():
    assert asyncio.run(
        cr.resolve_proposable_action("n", "add_event", "create_event", fetch=AsyncMock(return_value=None))
    ) is None


def test_fetch_error_returns_none():
    fetch = AsyncMock(side_effect=RuntimeError("mqtt down"))
    assert asyncio.run(cr.resolve_proposable_action("n", "add_event", "create_event", fetch=fetch)) is None


def test_list_stamps_owning_command():
    fetch = AsyncMock(return_value=_report())
    actions = asyncio.run(cr.list_proposable_actions("node-7", fetch=fetch))
    assert actions == [{"callback": "create_event", "params": [{"name": "title", "required": True}], "command": "add_event"}]

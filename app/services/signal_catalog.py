"""Catalog of the Signal Bus kinds a household can attach an automation to — the
source of truth for the mobile "signal automations" UI.

WHY a static catalog: producers emit signal ``kind`` s as free-form string
literals with NO central registry (``kind`` is a label, never control flow), and
they do NOT declare which kinds they emit — so "what can fire for this household"
cannot be derived mechanically from installed producers. This module is that
declaration: the user-authorable kinds, each with a human label, a one-line
description, and the ``facts`` it carries (so an instruction can reference them).
A household's GET annotates each with whether it has actually been *observed*
firing (a DISTINCT-kind query over the ``signals`` table).

Keep in sync with the real emitters (the only three kinds produced today):
  presence.seen / presence.left  — signal_service.record_presence (mobile + voice)
  appt.upcoming                  — jarvis-cmd-calendar calendar_alerts agent

``leave_by.suggested`` is CC-internal (a reaction's own output), so it is NOT in
the catalog — a user doesn't author against it.

Fact-key note: the CC writer records presence facts with ``user_id`` while the SDK
convenience helper uses ``user`` — the canonical shape declared here is the CC
writer's (``user_id``); execution normalizes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalKind:
    """One user-authorable signal kind, as shown in the automations UI."""

    kind: str
    label: str  # event phrasing for the card header ("I leave home")
    description: str  # one line: when it fires
    facts: dict[str, str]  # fact key -> what it means (canonical declared shape)
    example: str  # example free-text instruction, for the placeholder
    source: str  # typical source_agent, for display


# The user-authorable catalog. List order = display order in the app.
SIGNAL_CATALOG: list[SignalKind] = [
    SignalKind(
        kind="presence.left",
        label="I leave home",
        description="Your phone leaves the home area — you've left.",
        facts={
            "user_id": "Who left",
            "state": "Always 'away'",
            "room": "Last known room, if any",
        },
        example="Lock the front door",
        source="mobile",
    ),
    SignalKind(
        kind="presence.seen",
        label="I arrive home",
        description="Your phone enters the home area — you're home.",
        facts={
            "user_id": "Who arrived",
            "state": "Always 'home'",
            "room": "Room, if known",
        },
        example="Turn on the entryway lights",
        source="mobile",
    ),
    SignalKind(
        kind="appt.upcoming",
        label="An appointment is coming up",
        description="A calendar event with a location enters its leave-by window.",
        facts={
            "title": "Event title",
            "location": "Where it is",
            "start_iso": "Start time (ISO-8601)",
            "start_display": "Start time (friendly)",
            "event_id": "Calendar event id",
        },
        example="Remind me 30 minutes before I have to leave",
        source="calendar_alerts",
    ),
]

_BY_KIND: dict[str, SignalKind] = {s.kind: s for s in SIGNAL_CATALOG}


def catalog() -> list[SignalKind]:
    """The user-authorable signal kinds, in display order (a copy)."""
    return list(SIGNAL_CATALOG)


def get_kind(kind: str) -> SignalKind | None:
    """The catalog entry for ``kind``, or None if it isn't user-authorable."""
    return _BY_KIND.get(kind)


def is_authorable(kind: str) -> bool:
    """True if ``kind`` is a signal a household may attach an instruction to."""
    return kind in _BY_KIND

"""The labeled precision corpus.

Weighted toward NEGATIVES (situations where the reasoner must stay quiet),
because over-proposing — nagging — is the failure mode that keeps proactive
proposals off by default. Every negative carries the FULL menu (so the model has
tempting options to misfire on), except where a scenario deliberately restricts
it. Positives span every ``listening_signal_types`` kind so recall isn't measured
on one command alone.

Labels are meant to be defensible: a negative is a situation where proposing ANY
of the menu actions would be a nag, not a judgment call. Ambiguous cases (a note
that could arguably be a reminder) are left out rather than mislabeled — they'd
only add noise to the false-positive rate.
"""

from __future__ import annotations

from typing import Any

from evals.signal_precision.harness import Scenario


# ── builders ──────────────────────────────────────────────────────────────────
def _param(name: str, required: bool = False, enum: list[str] | None = None) -> dict[str, Any]:
    p: dict[str, Any] = {"name": name, "required": required}
    if enum:
        p["enum_values"] = enum
    return p


def _action(
    callback: str, card_title: str, params: list[dict[str, Any]], idem: str = "idempotency_key"
) -> dict[str, Any]:
    # The idempotency param is system-injected (not offered to the model), matching
    # how real commands declare proposable actions.
    return {
        "callback": callback,
        "card_title": card_title,
        "idempotency_param": idem,
        "params": [*params, _param(idem, required=True)],
    }


def _command(
    command_name: str, actions: list[dict[str, Any]], listening: tuple[str, ...] = ()
) -> dict[str, Any]:
    return {
        "command_name": command_name,
        "listening_signal_types": list(listening),
        "proposable_actions": actions,
    }


def _obs(source_key: str, kind: str, **data: Any) -> dict[str, Any]:
    return {"source_key": source_key, "kind": kind, "data": data}


# ── the menu of proposable commands (what a well-equipped node advertises) ────
ADD_EVENT = _command(
    "add_event",
    [_action("create_event", "Add to your calendar?",
             [_param("title", True), _param("start", True), _param("end"), _param("location")])],
    listening=("appt.detected",),
)
SHOPPING = _command(
    "create_shopping_list",
    [_action("create_list", "Create a shopping list?", [_param("items", True)])],
    listening=("meal_plan.completed", "grocery.needed"),
)
REMINDER = _command(
    "set_reminder",
    [_action("create_reminder", "Set a reminder?", [_param("text", True), _param("when", True)])],
    listening=("reminder.suggested",),
)
TIMER = _command(
    "start_timer",
    [_action("start", "Start a timer?", [_param("duration", True), _param("label")])],
)
MUSIC = _command(
    "play_music",
    [_action("play", "Play music?", [_param("query", True)])],
)

FULL_MENU = [ADD_EVENT, SHOPPING, REMINDER, TIMER, MUSIC]
# A menu with NOTHING relevant to an appointment — used to prove the reasoner
# won't shoehorn an appt into a timer/music proposal when the right command is
# not installed.
IRRELEVANT_MENU = [TIMER, MUSIC]


# ── positives: a specific action SHOULD be proposed ──────────────────────────
POSITIVES: list[Scenario] = [
    Scenario(
        name="appt_email_detected",
        bundle=[_obs("appt:dentist", "appt.detected",
                     title="Dentist appointment", start="2026-08-14T09:00:00",
                     summary="Dentist appointment Thursday at 9am")],
        available_commands=FULL_MENU,
        expect=("add_event", "create_event"),
        note="the canonical email->calendar pilot",
    ),
    Scenario(
        name="appt_amid_presence_noise",
        bundle=[_obs("presence:alex", "presence.seen", subject="Alex", summary="Alex is home"),
                _obs("appt:standup", "appt.detected",
                     title="Team standup", start="2026-08-13T10:00:00",
                     summary="Team standup tomorrow 10am")],
        available_commands=FULL_MENU,
        expect=("add_event", "create_event"),
        note="must pick the appt out of a bundle that also has a no-op presence signal",
    ),
    Scenario(
        name="appt_flight",
        bundle=[_obs("appt:flight", "appt.detected",
                     title="Flight to SFO", start="2026-08-20T06:30:00",
                     summary="Your flight to SFO departs Aug 20 at 6:30am")],
        available_commands=FULL_MENU,
        expect=("add_event", "create_event"),
    ),
    Scenario(
        name="appt_with_location",
        bundle=[_obs("appt:lunch", "appt.detected",
                     title="Lunch with Priya", start="2026-08-13T12:30:00", location="Cafe Luna",
                     summary="Lunch with Priya at Cafe Luna, Tue 12:30pm")],
        available_commands=FULL_MENU,
        expect=("add_event", "create_event"),
        note="optional param (location) available to fill",
    ),
    Scenario(
        name="meal_plan_completed",
        bundle=[_obs("meal:wk33", "meal_plan.completed",
                     items=["milk", "eggs", "bread", "spinach"],
                     summary="This week's meal plan is ready")],
        available_commands=FULL_MENU,
        expect=("create_shopping_list", "create_list"),
    ),
    Scenario(
        name="grocery_needed",
        bundle=[_obs("grocery:low", "grocery.needed",
                     items=["coffee", "oat milk"],
                     summary="You're running low on coffee and oat milk")],
        available_commands=FULL_MENU,
        expect=("create_shopping_list", "create_list"),
        note="second listening kind for the same command",
    ),
    Scenario(
        name="reminder_suggested",
        bundle=[_obs("rem:trash", "reminder.suggested",
                     text="take out the trash", when="tonight at 8pm",
                     summary="Reminder to take out the trash tonight at 8pm")],
        available_commands=FULL_MENU,
        expect=("set_reminder", "create_reminder"),
    ),
]


# ── negatives: NOTHING should be proposed (the anti-nag set) ─────────────────
NEGATIVES: list[Scenario] = [
    Scenario(
        name="presence_home_only",
        bundle=[_obs("presence:alex", "presence.seen", subject="Alex", summary="Alex arrived home")],
        available_commands=FULL_MENU,
        expect=None,
        note="arriving home warrants no card from this menu",
    ),
    Scenario(
        name="presence_left",
        bundle=[_obs("presence:sam", "presence.left", subject="Sam", summary="Sam left home")],
        available_commands=FULL_MENU,
        expect=None,
    ),
    Scenario(
        name="weather_ambient",
        bundle=[_obs("weather:today", "weather", summary="Sunny, high of 78°F today")],
        available_commands=FULL_MENU,
        expect=None,
    ),
    Scenario(
        name="media_playing",
        bundle=[_obs("media:now", "media.playing",
                     summary="Now playing: Miles Davis — Blue in Green")],
        available_commands=FULL_MENU,
        expect=None,
        note="music already playing is not a request to play music",
    ),
    Scenario(
        name="diagnostic_noise",
        bundle=[_obs("sys:health", "system.status", summary="All nodes healthy; disk 42% used")],
        available_commands=FULL_MENU,
        expect=None,
    ),
    Scenario(
        name="door_opened",
        bundle=[_obs("door:front", "door.opened", summary="Front door opened")],
        available_commands=FULL_MENU,
        expect=None,
    ),
    Scenario(
        name="presence_plus_weather",
        bundle=[_obs("presence:alex", "presence.seen", subject="Alex", summary="Alex is home"),
                _obs("weather:today", "weather", summary="Cloudy, 61°F")],
        available_commands=FULL_MENU,
        expect=None,
        note="two no-op signals together are still no-ops",
    ),
    Scenario(
        name="both_home",
        bundle=[_obs("presence:alex", "presence.seen", subject="Alex", summary="Alex is home"),
                _obs("presence:sam", "presence.seen", subject="Sam", summary="Sam is home")],
        available_commands=FULL_MENU,
        expect=None,
    ),
    Scenario(
        name="appt_but_no_calendar_capability",
        bundle=[_obs("appt:dentist", "appt.detected",
                     title="Dentist appointment", start="2026-08-14T09:00:00",
                     summary="Dentist appointment Thursday at 9am")],
        available_commands=IRRELEVANT_MENU,
        expect=None,
        note="right signal, but no calendar command — must NOT shoehorn a timer/music card",
    ),
    Scenario(
        name="vague_someday_note",
        bundle=[_obs("note:fence", "note", summary="Idea: repaint the fence someday")],
        available_commands=FULL_MENU,
        expect=None,
        note="no time, no list, no appointment — nothing actionable",
    ),
    Scenario(
        name="past_tense_done",
        bundle=[_obs("note:trash", "note", summary="Took out the trash already")],
        available_commands=FULL_MENU,
        expect=None,
        note="a completed action is not a reminder to set",
    ),
]


CORPUS: list[Scenario] = [*POSITIVES, *NEGATIVES]

# Mobile Command-Data Browser

## Overview

Surface jarvis-node's per-command storage (the `command_data` table accessed via `JarvisStorage`) in the mobile app so users can browse and edit it without SSH access. Backed by a schema-driven editor declared on each `IJarvisCommand`, so any current or future command can opt in to a structured UI for its records.

## Problem Statement

Stored command data — reminders, custom routine state, Bluetooth pairings, OAuth refresh state, agent caches — is opaque from the mobile app today. The only ways for a user to inspect or correct that state are voice ("delete my reminder about X") or SSH plus a Python REPL.

Real incident, June 2026: a UTC-era bug in `ReminderService.resolve_due_at` (fixed in commit `1477e70`) left a daily reminder on the kitchen node firing at 3 PM EDT instead of the intended 7 PM EDT. The bug was fixed, but two pre-fix reminders kept firing at the wrong time with no user-accessible way to discover or delete them. Identifying and removing them required SSHing into the prod node, decrypting the SQLCipher DB, and bypassing the in-memory singleton via a service restart.

We want this kind of recovery to be a thirty-second task in the mobile app, not a forty-five-minute SSH expedition.

## Goals

- Mobile users can list, view, and edit the contents of `command_data` rows owned by any command on any node in their household.
- Commands declare a UI schema (fields, types, labels, edit affordance) so that the mobile form is structured, not a raw JSON tree.
- Sensitive or duplicated surfaces (OAuth, custom routines that already have dedicated UI) can opt out per-command.
- Forward-compatible: new field types and new "browser modes" can ship in the SDK without breaking older CC or mobile builds.

## Non-goals

- A schema or migration system for the underlying `command_data` rows.
- Editing fields the command itself doesn't expose.
- Editing or deleting rows from commands that have opted out.
- Replacing the existing `/api/v0/mobile/routines` builder. Routines remain on their dedicated screen and opt out of the browser.

## Architecture

```
Mobile (React Native)
    │  GET/POST/PATCH/DELETE /api/v0/mobile/command-data
    ▼
jarvis-command-center
    │  app/api/mobile_command_data.py
    │  - JWT auth, household scoping
    │  - publish MQTT request with correlation_id
    │  - await response on response topic (timeout 3s)
    ▼  MQTT (Mosquitto)
jarvis-node-setup (on each Pi)
    │  scripts/mqtt_tts_listener.py (or new handler module)
    │  - subscribed to command-data topics
    │  - dispatches to ReminderCommand / RoutineCommand / ... via registry
    │  - reads IJarvisCommand.editable_fields()/display_summary()/data_browser_mode
    │  - uses CommandDataRepository directly (bypasses per-command run())
    │  - publishes response with same correlation_id
```

Source of truth stays on the node. CC is a transport + auth layer with a short-lived schema cache. Matches the existing pattern used by `app/api/bluetooth.py` and `app/api/node_settings.py`.

## SDK additions (`jarvis-command-sdk`)

### `FieldSpec`

```python
from dataclasses import dataclass

@dataclass
class FieldSpec:
    name: str                                # JSON key in the stored record
    type: str                                # see vocabulary below
    label: str | None = None                 # display label; falls back to name
    description: str | None = None           # helper text under the field
    editable: bool = True                    # False → read-only display row
    required: bool = False                   # null/empty allowed?
    enum_values: list[str] | None = None     # for type="enum"
    item_type: str | None = None             # for type="array" — element type
    fields: list["FieldSpec"] | None = None  # for type="object" — nested schema
    placeholder: str | None = None
```

`type` is a string (not enum) so new types ship without rebuilding CC/mobile. Mobile maps `type → widget` and falls back to a plain text input for unknown types (degrade to permissive, since the data is already valid by virtue of being stored).

**Initial type vocabulary:**

| `type` | Widget |
|---|---|
| `string` | single-line text |
| `text` | multi-line text |
| `int`, `float` | number input |
| `bool` | toggle |
| `enum` | dropdown (uses `enum_values`) |
| `datetime` | date + time picker |
| `date` | date picker |
| `time` | time picker (`HH:MM`) |
| `duration` | minutes/hours stepper |
| `array` | list editor (element widget driven by `item_type`) |
| `object` | push-to-screen with nested `fields` |
| `id` | read-only mono-font chip |
| `user_ref` | user picker resolved via auth `/internal/users/batch` |
| _unknown_ | text input |

### `display_summary` / `editable_fields` / `data_browser_mode`

```python
from typing import Literal

DataBrowserMode = Literal["enabled", "disabled", "readonly"]


class IJarvisCommand(ABC):
    ...
    @property
    def data_browser_mode(self) -> str:
        """Controls whether this command's stored data appears in the
        mobile command-data browser.

        - "enabled"  (default): list, view detail, edit, delete
        - "disabled": not shown at all
        - "readonly": list + view detail, no edit, no delete  [reserved]

        Wire format is plain str so new modes can ship without breaking
        older CC/mobile. Unknown mode → treated as "disabled" (fail closed)."""
        return "enabled"

    def editable_fields(self) -> list[FieldSpec]:
        """Schema for the stored records this command persists.

        Default: empty list (no edit UI; rows appear as read-only JSON).
        Commands with structured data override this."""
        return []

    def display_summary(self, record: dict) -> dict:
        """Return {"title": str, "subtitle": str | None, "icon": IconName}
        for the list-row summary.

        Default: title = record's first string field, subtitle = None,
        icon = "default". Commands should override for meaningful rows."""
        ...
```

### `SUPPORTED_ICONS` / `IconName`

```python
# jarvis_command_sdk/icons.py

from typing import Literal

SUPPORTED_ICONS: list[str] = [
    # Time / scheduling
    "bell", "clock", "calendar", "hourglass",
    # Communication
    "message", "mail", "phone", "link",
    # Media
    "music", "speaker", "mic", "play",
    # Smart home
    "home", "light", "lock", "thermostat", "camera",
    # Devices
    "wifi", "battery",
    # People
    "user", "users",
    # Auth / security
    "key", "shield",
    # Weather
    "cloud", "sun",
    # Generic / fallback
    "tag", "star", "folder", "default",
]

IconName = Literal[
    "bell", "clock", "calendar", "hourglass",
    "message", "mail", "phone", "link",
    "music", "speaker", "mic", "play",
    "home", "light", "lock", "thermostat", "camera",
    "wifi", "battery",
    "user", "users",
    "key", "shield",
    "cloud", "sun",
    "tag", "star", "folder", "default",
]
```

Mobile keeps a `iconName → component` registry mirroring this list. Unknown icon → `default`. A Forge/CI lint diffs `SUPPORTED_ICONS` against the mobile registry to catch drift.

## Node MQTT contract

Topic prefix: `jarvis/{node_id}/command-data`.

| Operation | Request topic | Response topic | Request payload | Response payload |
|---|---|---|---|---|
| List records for a command | `…/list` | `…/list/response/{correlation_id}` | `{correlation_id, command_name}` | `{records: [{key, summary: display_summary(...), data}]}` |
| Get one record | `…/get` | `…/get/response/{correlation_id}` | `{correlation_id, command_name, key}` | `{record, schema}` |
| Update one record | `…/update` | `…/update/response/{correlation_id}` | `{correlation_id, command_name, key, patch}` | `{ok, record?, error?}` |
| Delete one record | `…/delete` | `…/delete/response/{correlation_id}` | `{correlation_id, command_name, key}` | `{ok}` |
| Fetch schema only | `…/schema` | `…/schema/response/{correlation_id}` | `{correlation_id, command_name}` | `{fields: [FieldSpec...], mode}` |
| List visible commands | `…/commands` | `…/commands/response/{correlation_id}` | `{correlation_id}` | `{commands: [{command_name, mode, icon}]}` |

**Correlation:** CC generates a UUID per request. Subscribes to `…/{op}/response/{correlation_id}` before publishing. Times out after 3 s. On timeout, returns `504 NODE_TIMEOUT` to mobile (with `Retry-After: 1`).

**Filtering:** Node skips commands where `data_browser_mode == "disabled"` (and any unknown mode value) before serialising the response. Defense in depth — the data never crosses the wire even if mobile's filter is buggy.

**Server-side validation:** `update` runs whatever the command's existing validation path is on the merged record before persisting. On failure, response is `{ok: false, error: {field, message}}`. No new validation framework — punt to the command's own logic.

## CC REST surface

`app/api/mobile_command_data.py`, mounted under `/api/v0/mobile/command-data`. JWT-authed via `verify_user_jwt`.

| Method | Path | Behaviour |
|---|---|---|
| `GET` | `/nodes` | List the user's nodes (household-scoped) for the node-picker |
| `GET` | `/nodes/{node_id}/commands` | List commands on that node with non-`disabled` mode |
| `GET` | `/nodes/{node_id}/commands/{command_name}/schema` | Return cached `FieldSpec[]` + mode |
| `GET` | `/nodes/{node_id}/commands/{command_name}/records` | List records (summaries only) |
| `GET` | `/nodes/{node_id}/commands/{command_name}/records/{key}` | Get one record's full data + schema |
| `PATCH` | `/nodes/{node_id}/commands/{command_name}/records/{key}` | Update; body is a patch object |
| `DELETE` | `/nodes/{node_id}/commands/{command_name}/records/{key}` | Delete |

**Schema cache:** in-process dict keyed by `(node_id, command_name)`. TTL 10 minutes; busted on node reconnect. Eliminates a round trip when mobile opens an edit form.

**Auth scoping:** every request resolves the user's household and verifies `node_id` belongs to it before publishing MQTT. Cross-household access is `403`.

**Error mapping:**

| Node response | HTTP |
|---|---|
| `{ok: true}` | `200` |
| `{ok: false, error: {field, message}}` | `400` with `{errors: [{field, message}]}` |
| Timeout | `504` |
| Unknown command_name on node | `404` |

## Mobile screen stack

1. **Data Browser home** — node picker (if >1 node) + section list of commands (`commands` endpoint). Each section header shows command icon + label + record count. Tap → section detail.
2. **Records list** — list rows from `display_summary`: icon, title, subtitle, chevron. Swipe-left = delete (with confirmation). Tap row → detail.
3. **Record detail (read-only)** — KVP list driven by the schema. Each row renders type-appropriate display (datetime → formatted local, bool → ✓/✗, enum → label, etc.). Top-right `Edit` button if `mode == "enabled"` and any field is `editable`. Nested `object` fields render as a chevron row that pushes another detail screen recursively.
4. **Record edit (form)** — same field layout but interactive widgets per the type vocabulary table. `Save` button issues `PATCH`. Server-returned field-level errors render inline.

**Widget fallback:** if mobile encounters a `type` it doesn't recognise, it renders a text input pre-filled with `JSON.stringify(value)`. The user can still inspect and (carefully) edit. A toast warns: "Field type X not supported by this app version."

**Mode handling on mobile:**

| `data_browser_mode` returned by node | UI behaviour |
|---|---|
| `enabled` | full CRUD |
| `readonly` | list + detail, no Edit button, no swipe-delete |
| `disabled` _(should not appear; node filters)_ | section hidden |
| _unknown value_ | section hidden |

## Reminder pilot

```python
class ReminderCommand(IJarvisCommand):
    # data_browser_mode defaults to "enabled"

    def editable_fields(self) -> list[FieldSpec]:
        return [
            FieldSpec("reminder_id",       "id",       label="ID",            editable=False),
            FieldSpec("text",              "string",   label="What",          required=True),
            FieldSpec("due_at",            "datetime", label="When",          required=True),
            FieldSpec("recurrence",        "enum",     label="Repeat",
                      enum_values=["daily", "weekdays", "weekly", "biweekly", "monthly"]),
            FieldSpec("snooze_until",      "datetime", label="Snoozed until"),
            FieldSpec("user_id",           "user_ref", label="Owner",         editable=False),
            FieldSpec("announced",         "bool",     label="Fired",         editable=False),
            FieldSpec("announce_count",    "int",      label="Times fired",   editable=False),
            FieldSpec("last_announced_at", "datetime", label="Last fired",    editable=False),
            FieldSpec("created_at",        "datetime", label="Created",       editable=False),
        ]

    def display_summary(self, record: dict) -> dict:
        rec = record.get("recurrence")
        subtitle = ReminderService.format_due_at_human(record["due_at"])
        if rec:
            subtitle += f" • {rec}"
        return {"title": record["text"], "subtitle": subtitle, "icon": "bell"}
```

`ReminderCommand` will also need to teach `ReminderService.update_reminder(key, patch)` how to handle `due_at` changes (re-compute internal scheduling state) and `recurrence` changes (reset `announced` and snooze). That's a new method on the service; the existing `_run_set` path is untouched.

## Opted-out commands

```python
class RoutineCommand(IJarvisCommand):
    @property
    def data_browser_mode(self) -> str: return "disabled"

class OAuthCommand(IJarvisCommand):
    @property
    def data_browser_mode(self) -> str: return "disabled"
```

## Open / deferred decisions

- **Multi-node fan-out vs picker.** First pass uses a node picker; user chooses which node's data to browse. Fan-out (single list aggregating across nodes) is deferred until we have evidence anyone has > 2 nodes with overlapping reminders.
- **Per-user vs household filter.** First pass shows all records the requesting JWT user can see (i.e. records where `user_id == me` plus legacy `user_id IS NULL`). A `?scope=household` filter for households with super-user access is deferred.
- **Bulk delete.** Not in scope. Add later if users start asking for "delete all reminders containing X."
- **Validation framework on FieldSpec.** Explicitly punted to server-side `validate_params` / command-specific checks. Will revisit if we end up duplicating regex/range logic across commands.
- **`readonly` mode implementation.** Reserved in the SDK now (so SDK consumers can declare it) but mobile renders unknown modes as hidden until we wire up the read-only widget pass.

## Out of scope

- A "create new record" flow from the browser. Records are created via the existing voice/tool path. Browser is browse + edit + delete.
- Versioning / undo. Deletes are immediate. Mirrors current behaviour.
- Audit logging of edits beyond what each command already logs.
- Exposing `IJarvisAgent` state (separate surface area; agents don't persist to `command_data`).

## Rollout

1. SDK PR — `FieldSpec`, `display_summary`, `editable_fields`, `data_browser_mode`, `SUPPORTED_ICONS`, `IconName`. No behaviour change for existing commands.
2. Node PR — MQTT handlers, registry walk, `RoutineCommand` and `OAuthCommand` opt-outs, `ReminderCommand` pilot.
3. CC PR — `app/api/mobile_command_data.py`, schema cache, MQTT publish/subscribe plumbing.
4. Mobile PR — screen stack, widget mapping, icon registry mirror, lint check against `SUPPORTED_ICONS`.

Each PR independently mergeable; mobile lights up after step 3.

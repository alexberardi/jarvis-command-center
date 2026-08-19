# Signal Bus adoption across agents — scoping & roadmap

> **Status:** scoping only — *nothing here is built yet.* This is the plan for turning
> background agents into Signal Bus **producers**. Captured 2026-08-19.

## Why

The Signal Bus is "**behavior = installed capabilities × current signals**." Today the
consumers are in place — deterministic reactions (the `appt.upcoming` → leave-by card),
the natural-language **automation builder** (a household attaches a free-text instruction
to a signal kind; when it fires, the model interprets it against the household's tools and
acts — automatically or via a tap-to-confirm card), and reactive voice render. What's
thin is **producers**: only three signals exist today.

The thesis this scoping confirms: **the bus pays off hardest in home automation.** Home
Assistant already installs the *capabilities* (`control_device`); signals supply the
*when*. Cross that with **presence** and **weather** and you get the classic
"everyone left → arm + lock + lights-off" / "freeze tonight → cover the plants" matrix —
and the shipped builder makes each of those user-authored.

## Producers that already exist (baseline)

| Signal | Producer | Drives |
|---|---|---|
| `appt.upcoming` | `jarvis-cmd-calendar/agents/calendar_alerts` | leave-by reminder card |
| `presence.seen` / `presence.left` | mobile app (`/mobile/presence`) | presence automations |
| `game.final` | `jarvis-cmd-sports/agents/sports_alerts` | ambient render + automations |

`sports_alerts` is the **reference-quality producer** — copy its shape: thread-offloaded
`emit`, upsert-by-`source_key`, re-emit only on change, `ok`/`no_backend` accounting.

## Ranked candidates

### HIGH — the payoff tier

| Agent | Proposed signal(s) | Effort | Notes |
|---|---|---|---|
| **home_assistant** (`jarvis-home-assistant-integration`) | `device.cover.left_open`, `device.lock.unlocked`, `device.offline`, `device.battery.low`, `security.alarm.state_changed`/`.triggered`, `climate.temp.out_of_range` | **Low** (steady-state) / **High** (transient) | The canonical smart-home surface. Polls a full `get_states` **snapshot** every 5 min (no diff, no `subscribe_events` today). Sensor domains (door/motion/leak/smoke, battery, alarm panel) are already fetched but excluded from voice context → the data is in hand for free. `source_key = ha:<domain>:<entity_id>` (upsert). **Push-only** (transient / life-safety, need a WS listener): `sensor.smoke/leak.detected`, `sensor.motion.detected`, `device.contact.opened`. |
| **weather — open-weather** (`jarvis-cmd-open-weather`) | `weather.alert`, `weather.freeze`, `weather.precip.starting` | **Low** | 🎯 The OneCall 3.0 command **already fetches the government `alerts` array and discards it** (`commands/get_weather/command.py`, never parsed) — `weather.alert` is a parse-and-upsert away. |
| **medication_reminders** (`jarvis-cmd-medication`) | `medication.due`, `medication.overdue` | **Low** | Health/caregiving is the perfect user-authored-automation fit ("if Mom's evening pill is overdue, call me"). Detection + dedup are **already persisted** to node storage. **Privacy:** personal meds are deliberately never spoken aloud — signals must carry the owner `user_id`/scope and consumers must honor the household-vs-owner audience. |
| **email_alerts** (`jarvis-cmd-email`) | `email.important_received{reason: vip\|urgent}` | **Low** | Deterministic VIP + urgent-keyword detection, already deduped. Maps 1:1 onto "when my boss / anything urgent emails, do Y." `source_key = email:{uid}:{message_id}`; facts = **sender + subject + reason only, never the body**. |
| **weather — meteo** (`jarvis-cmd-meteo-weather`) | same `weather.alert` | **Med-High** | Same value as open-weather, but Open-Meteo's forecast endpoint carries no gov alerts — needs a new source or run-diffing. **The two weather agents are twins** (same class name + memory keys); install one per household, and standardize a shared `weather.alert` `source_key` so whichever runs upserts one row. |

### MEDIUM — extensions

| Agent | Proposed signal | Notes |
|---|---|---|
| **calendar_alerts** (extend) | `appt.imminent` | Today `appt.upcoming` only fires for **located, timed** events → locationless & all-day events reach no signal. This closes that recall gap. |
| **drive_time_alerts** (`jarvis-cmd-drive-time`) | `travel.leave_now` | Traffic-aware departure *instant* (distinct from the pre-emptive leave-by). Carry the computed `leave_by_iso`. **Don't** re-propose a reminder card — feed action-at-departure automations ("time to leave → lock up + text my wife"). |
| **appointment_scan** (`jarvis-cmd-email`) | `email.appointment_detected` | Email-detected ≠ scheduled → keep **distinct** from `appt.upcoming` (reusing it would feed leave-by a not-yet-real event). Marginal: the `add_event` card already reacts. |
| **sports_alerts** (extend) | `game.start` | Cheap — the pre→in transition is already detected. `game.final` is done. |

### LOW / skip

- **news_alerts** — high-volume, low-salience; keep it as passive memory. Only signal *post-filter* if topic automations are wanted.
- **smart_reply** (email) — overlaps `email_alerts`' importance stream, gated off by default. If pursued, fold into `email.important_received(reason=filter)`.
- **spotify_keepalive** — not a candidate; pure node infrastructure (token refresh, daemon). No household events.

## How to build it (the patterns that matter)

1. **Steady-state ↔ upsert; transition ↔ diff.** A level condition ("garage is open right
   now") is a perfect fit for signal **upsert-by-`source_key` + TTL** — each poll re-asserts
   one live row. To fire an automation only on the *transition* (locked→unlocked), the agent
   must retain a **prior snapshot and diff** — net-new for HA (it snapshots wholesale), but
   already present in calendar/medication/sports.
2. **Emit on the agent's existing successful-detection branch** so signals inherit its dedupe.
   One `source_key` per thing: `ha:<domain>:<eid>`, `email:{uid}:{msgid}`,
   `med-due:{med}:{slot}:{date}`, `weather.alert:{lat},{lon}:{event}`.
3. **Passive vs event.** Today's forecast and the device snapshot stay **memories** (context) —
   only *threshold-crossing events* become signals.
4. **Privacy in facts.** Keep email bodies, personal-med scope, and appointment detail out of
   `facts`; honor the per-user/household audience the agent already enforces.
5. **Poll vs push.** 5–30 min polling is fine for slow/steady conditions (left-open, offline,
   battery, freeze). Life-safety (smoke, leak, motion) is latency-critical and not reliably
   poll-detectable → needs a true event source (an HA WebSocket `subscribe_events` listener or
   HA-side webhooks). That's a bigger architectural upgrade worth its own milestone.
6. **Producer contract:** `JarvisSignals("<agent>").emit(kind=, source_key=, summary=, facts={},
   scope={"user_id": …}, ttl_seconds=, cacheable=, salience=)`
   (`jarvis-command-sdk/jarvis_command_sdk/signals.py`). Adding a new producer *kind* needs
   **zero** CC ingest/dispatcher change — the reaction registry is generic.

## Suggested first slice (cheapest × highest value)

1. **OpenWeather `weather.alert`** (data already fetched) + **HA steady-state safety**
   (`device.cover.left_open`, `device.lock.unlocked`-at-night, `device.offline`) — both low
   effort, both compose with presence.
2. **`medication.overdue`** + **`email.important_received`** — low effort, high user intent.
3. Then the **HA `subscribe_events` push upgrade** to unlock the transient / life-safety tier
   (motion, smoke, leak, momentary contact).

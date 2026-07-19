# Attention Broker

> **Status:** Scoped, not yet built. All load-bearing code claims below were **verified against the actual source on 2026-07-18** (jarvis-command-center, jarvis-node-setup, jarvis-notifications, jarvis-node-mobile, jarvis-command-sdk). Where a doc (CLAUDE.md) contradicts the code, the code citation wins and the discrepancy is noted.

## Overview

A deterministic, LLM-free governance layer in jarvis-command-center that every proactive notification passes through before it reaches a human. It persists each attention event to a journal, deduplicates across nodes, and routes down an escalation ladder — **journal → inbox → push (default) → LED invite → speak (urgent-only)** — through four gates: consent ceiling, trust tier, daily budget, and context (quiet hours). Anything demoted lands in the journal and coalesces into a daily digest card instead of disappearing.

The broker governs **retroactively**: it interposes the three CC endpoints all node-originated notifications already flow through, so the ~24 existing Pantry packages are governed without modification. It is off by default (`attention.enabled`, per-household); with the flag off, every interposed endpoint behaves byte-for-byte as today.

This PRD covers the broker itself (program phases 1–3: skeleton, feedback/tiers, consent surface). The LED-invite rung, the digest worker (see `prds/daily-briefing.md`), and the standing-orders judgment layer are separate work items that plug into the broker's ladder and are referenced only where their future schema needs anticipating now.

## Problem statement

Proactive output today is ungoverned at every layer:

- The news agent that got proactivity disabled pushes **directly** — its noise channel is `/node/push-notification`, not the alert queue — and nothing between it and the phone can say no.
- **Inbox items are never deduplicated** (`jarvis-notifications/app/api/inbox.py:221-238` is a straight passthrough to `inbox_service.create_item`; no dedup exists on the inbox path). Push dedup is a 60-second in-memory window (`DEDUP_WINDOW_SECONDS = 60`) keyed on `(source_service, target_id, title, body, category)` — `target_type` is **not** in the key, and a deduped send is logged as `delivery_status="skipped"`, indistinguishable from no-tokens and no-relay.
- The node's alert queue is in-memory and node-local (`alert_queue_service.py:49-52`, no persistence anywhere in the module): restart drops everything, and the same agent on two nodes double-alerts.
- Node-side dedup is by **case-insensitive title**, not the SDK's new `dedupe_key` field (`alert_queue_service.py:74-83`; zero references to `dedupe_key` in the file).
- There is no budget, no per-source preference, no quiet hours, no record of what was sent or suppressed, and no way to ask "why did you tell me that?"

## Goals

- Every humanward proactive event is journaled with the gate trail that produced its outcome — delivered or withheld, and why.
- Hard per-household daily caps and per-source daily caps, enforced server-side where no package can out-shout them.
- Cross-node, cross-restart dedup on stable keys.
- All ~24 existing packages governed **unmodified** via endpoint interposition.
- **Zero LLM calls in the routing path.** Routing is always explainable and always terminates at the journal floor — never a silent drop.
- Off by default; per-household opt-in; with the flag off, interposed endpoints are behavior-identical to today.

## Non-goals

- **No spoken or LED delivery in this PRD.** The broker's ladder tops out at push until the verified in-room rung ships (separate work; schema anticipated below).
- **No LLM relevance filtering or standing orders.** The judgment layer is a separate consumer that will *propose* tiers to the broker; the broker only demotes.
- **No new mobile screens in phases 1–2.** The journal card renders through the existing generic InboxDetail path (verified below); the consent matrix screen is phase 3.
- **No governance of live voice responses.** Only unsolicited (proactive) output is in scope — tool results inside a conversation the user initiated are not attention events.

## The delivery ladder (owner decision, 2026-07-18)

| Rung | Channel | When |
|---|---|---|
| 0 | Journal only | Failed a gate; visible in the daily journal card and console |
| 1 | Inbox card | Default floor for delivered content |
| 2 | **Push (the default)** | Anything worth proactive attention — durable, personal, can't be missed by leaving the room |
| 3 | LED invite | *Future rung.* Purple "Jarvis has something to tell you"; redeemed via button / "what's up" |
| 4 | Speak unprompted | *Future rung.* Urgent/safety categories with explicit per-category user grant only — never a default |

Push is deliberately ranked **below** LED/speak on the intrusion scale but is the default *delivery*: the in-room rungs are supplements gated on presence, and a wrong presence guess costs nothing because the push already landed.

## Interposition — the core mechanism

All node-originated notification egress funnels through three node-auth endpoints in `app/api/node_commands.py`, each delegating to `app/services/inbox_notification_service.py`, which POSTs app-to-app to jarvis-notifications `/api/v0/inbox` and `/api/v0/notify`. The broker wraps these handlers: with `attention.enabled` on for the household, the request becomes an `attention_event`, gates run, and delivery (if any) goes through the **same** downstream services. Verified surface:

### `POST /node/push-notification` (`node_commands.py:239-292`)

- Request: `PushNotificationRequest {title: str, body: str, priority: str = "default", category: str = "alert", user_id: int | None, target_type: Literal["user","household"] = "household"}` (`:239-249`).
- ⚠️ **`priority` and `category` are accepted but never used** — the handler hardcodes `command_name="reminder"`, `actions=[]` and forwards only title/body/user_id/target_type to `push_confirmation_to_inbox` (`:277-287`), which in turn **hardcodes `category="confirmation"`** and **always fires a push** with `priority: "high"` (`inbox_notification_service.py:81, 110-120, 152`). The broker starts *reading* the request's `category` — that is the phase-1 routing key. Existing callers' categories must be audited before caps are enforced (see Open questions).
- Response `{sent: bool, inbox_item_id: str | None}` — always HTTP 200, `sent=False` on downstream failure (`:252-254, 289-292`). The broker preserves this shape; a withheld event returns `sent=false` plus a new additive `withheld_by: str | null` field.
- Auth: `Depends(verify_api_key)`; `household_id` comes from the validated node row (`:261-264, :274`) — callers cannot spoof household. The broker keys everything on this server-derived household_id.

### `POST /node/inbox-item` (`node_commands.py:414-494`)

- Request: `NodeInboxItemRequest {title, summary="", body="", category="general", metadata, user_id, create_push_notification=False, target_type="household"}` (`:414-439`). Handler is **sync def** (`:447`). Blank title or node without household → 200 with `sent=false` (silent drop, `:465-475`).
- CC injects `metadata.setdefault("node_id", node_context.node.node_id)` (`:481`) so inbox buttons route callbacks to the origin node — the broker must preserve this injection on every delivery it forwards.
- This is also the SDK path: `JarvisInbox.post()` targets `POST {cc}/api/v0/node/inbox-item` and returns discriminated tags `"ok"|"no_backend"|"no_cc_url"|"http_error"|"invalid"`, never raises (`jarvis-command-sdk/inbox.py:108-142`). Interposing here governs every SDK-using package.

### `POST /node/send-link` (`node_commands.py:365-407`)

- `SendLinkRequest {user_id: int (required), url, title?, body?}`; rejects non-http(s) URLs with `sent=false` (`:393-395`). ⚠️ Calls the **sync** helper `send_link_push_sync` inside an `async def` handler — blocking httpx on the event loop (`:401-407`). Interpose for journaling + caps; fixing the blocking call is a drive-by candidate, not a requirement.

### Downstream facts the broker inherits

- `post_inbox_item_sync` is synchronous (blocking `httpx.Client`, timeout 10.0), keyword-only, `push=True` **by default**, push priority hardcoded `"high"` (`inbox_notification_service.py:192-264`). Callers from async contexts must use `asyncio.to_thread` (same constraint `prds/daily-briefing.md` documents).
- Push target resolution: `target_type="user"` with `user_id=None` **silently falls back to household broadcast** (`inbox_notification_service.py:25-43`). The broker makes this explicit in the delivery record rather than inheriting it silently.
- Inbox visibility: `user_id IS NULL` means household-wide; the filter `(user_id == me) OR (user_id IS NULL)` is applied uniformly (`jarvis-notifications/app/services/inbox_service.py:60-63`).
- notifications `category` is free-form ≤ String(50); `title` is String(500) and **Postgres errors (5xx), not truncates**, on overflow — the create models have no Pydantic max_length (`jarvis-notifications/app/models.py:56-69`, `app/api/inbox.py:19-44`). The broker enforces length caps *before* forwarding.

### What interposition does NOT cover (accepted, documented)

- **Node-local alert announcements.** `get_alerts()` output never reaches CC today — it feeds the node's own queue/LED/announcer only (`agent_scheduler_service.py:316-320` is the only non-test `add_alert` caller). The node's existing priority-3 quiet-moment speech path (`alert_announcer.py`) continues ungoverned until the in-room rung migrates it, source by source, with a per-source suppression flag. This is deliberate: reminders/medication keep their current, proven local path (safety-class law below).
- **Direct app-to-app posts to jarvis-notifications** from other services bypass CC entirely. Phase-1 scope is node-originated traffic; CC's own producers (deep research, memory extraction, callback fan-outs) adopt the broker internally in phase 2.

### Node-side forward shim (phase 2, with the feedback command's node release)

A forward hook where the scheduler already collects `get_alerts()` (`agent_scheduler_service.py:316-320`) ships alerts upstream to `POST /api/v0/attention/events` so the journal sees silent node-local alerts too (LED/`"what's up"` behavior unchanged). Note the SDK default `priority=2` sits **below** the node's announce gate `ALERT_ANNOUNCE_PRIORITY = 3` (`alert_queue_service.py:37,43`) — most alerts are already silent-queue-only; the shim makes them *visible* in the journal, which is the "weather agent does nothing" fix's first half.

## Data model (CC Postgres, alembic at repo root — NOT `app/alembic`; CLAUDE.md's tree is wrong. New migration's `down_revision = "d9c8b7a6e5f4"`, hand-written 12-char pseudo-hex revision id via `./make_migration.sh`)

### `attention_events`

| column | type | notes |
|---|---|---|
| `id` | String(36) PK | uuid |
| `household_id` | String(255), indexed | from the validated node row, never client-supplied |
| `source` | String(100) | phase 1: the request `category`; later: SDK-declared source name |
| `category` | String(50) | mirrors notifications' String(50) cap |
| `title` / `summary` | String(500) / Text | broker-truncated to fit notifications' hard caps |
| `dedupe_key` | String(255), nullable, indexed | from SDK Alert / request; fallback = SHA-256 of normalized title |
| `target_user_id` | Integer, nullable | NULL = household |
| `origin_node_id` | String(255), nullable | for `metadata.node_id` re-injection |
| `payload_json` | Text | the original request, for replay/debugging |
| `created_at` | DateTime | |

Dedup rule: an event with the same `(household_id, source, dedupe_key)` inside `attention.dedupe_window_hours` (default 24) is journaled as `duplicate` and not re-delivered — replacing both the notifications 60s push window and the node's title-matching as the authority.

### `attention_deliveries`

| column | type | notes |
|---|---|---|
| `id` | String(36) PK | |
| `event_id` | FK → attention_events | |
| `rung` | enum: `journal` / `inbox` / `push` (later `led_invite` / `speak`) | outcome rung |
| `gate_trail` | Text (JSON) | ordered list of `{gate, result, detail}` — the "why did you tell me that?" substrate |
| `withheld_by` | String(50), nullable | gate name when rung = journal |
| `inbox_item_id` | String(36), nullable | the durable artifact id |
| `request_id` | String(36), nullable | reserved for the future verified in-room rung — **DB-backed by design**, because `NodeCommandService`'s pending store is an in-memory per-process dict with a 5-minute TTL and one-time-use verify that a CC restart wipes (verified; see Security) |
| `outcome` | String(30), nullable | reserved: `delivered` / `redeemed` / `expired` / `failed` |
| `created_at` | DateTime | |

### `attention_source_tiers` (phase 2)

`(household_id, source, category)` → `tier` (T0 journal-only … T3 push-eligible; T4/T5 reserved for LED/speak), `score` (decayed float), `state_reason` (String(255) — human-readable, mirroring the agent auto-disable culture), `updated_at`. New sources start at **T1 inbox-only probation**.

### `attention_consents` (phase 3)

`(household_id, source, category)` → `max_rung`, granted by a household **admin**. Speech (`max_rung = speak`) is grantable only per-category and never defaulted — the `allow_updates` lineage.

### `attention_feedback` (phase 2)

`(id, delivery_id, user_id, verb: useful|not_useful|mute|why, created_at)` — written by the feedback endpoint; consumed by the scoring worker.

TTL: events/deliveries swept after `attention.journal_ttl_days` (default 30) by a `_periodic_*` cleanup worker.

## The four gates (deterministic, in order)

1. **Consent ceiling** — `min(requested rung, consent max_rung)`. Absent row = default ceiling `push` for built-in sources, `inbox` for anything else. Third-party manifest requests can never raise this above `inbox` (future SDK phase); only a user grant can.
2. **Trust tier** — the source's earned rung. Phase 1: all sources T3 (no behavior change from tiers alone); phase 2 activates scoring: useful/acted promote, dismissed/ignored decay, `mute` hard-floors to T0 with a `state_reason`. **Demote-only law:** any future LLM consumer (salience judge, standing orders) may *propose* a rung; gates only lower it. Enforcement lives in this code, never in a prompt.
3. **Budget** — per-household daily counters from the settings DB: `attention.daily_push_budget` (default 8), `attention.daily_inbox_budget` (default 30), per-source `attention.source_daily_cap` (default 4). Exhausted budget demotes one rung, never drops.
4. **Context** — quiet hours (`attention.quiet_hours`, household tz): during quiet hours push demotes to inbox unless the category is safety-class.

**Safety-class exemption (law):** categories in `attention.safety_categories` (default `["medication", "reminder", "security", "safety"]`) bypass gates 2–4 entirely and cannot be muted by feedback — only an explicit consent-row change by a household admin can lower them. They are also excluded from dedup suppression beyond exact `dedupe_key` match. The LLM is never in this path.

**Personal-scope law:** an event with `target_user_id` set caps at push (never a future household-audible rung) — generalizing the medication push-only precedent.

## Settings (`app/services/settings_definitions.py`, new category `attention`)

| key | type | default | purpose |
|---|---|---|---|
| `attention.enabled` | bool | `false` | master switch, per-household scope |
| `attention.daily_push_budget` | int | `8` | household/day |
| `attention.daily_inbox_budget` | int | `30` | household/day |
| `attention.source_daily_cap` | int | `4` | per source/day |
| `attention.dedupe_window_hours` | int | `24` | |
| `attention.quiet_hours` | string | `"22:00-07:00"` | household tz |
| `attention.safety_categories` | string | `"medication,reminder,security,safety"` | comma-separated (no `json` value_type precedent in this service — see daily-briefing PRD's warning) |
| `attention.journal_ttl_days` | int | `30` | |
| `attention.journal_card_enabled` | bool | `true` | the daily journal card |
| `attention.journal_card_cron` | string | `"0 21 * * *"` | evening summary |

> Adding the `attention` category **breaks `tests/test_settings.py:55-62`**, which asserts the 17-category set by exact equality (`llm, tool_classifier, tool_router, transcription, prompt, model, conversation, admin, memory, network, oauth, smart_home, adapter, voice, routines, web_search, updates`). Add `attention` to `expected_categories` in the same change.

Consent reads/writes use the **household-admin pattern, not the superuser `/settings/*` router**: copy `app/api/mobile_household_settings.py` — explicit allowlist dict (`:31-34`), `_WRITE_ROLE = "admin"` / `_READ_ROLE = "member"` (`:38-39`), 404 on non-allowlisted keys (`:87-92`), `verify_household_role(...)` (`:94`). ⚠️ `verify_household_role`'s **default role is `"power_user"`** (`app/deps.py:343-347`) — pass roles explicitly. It fails closed (503) when `JARVIS_APP_KEY` is unset (`deps.py:352-358`).

## Worker + scheduling conventions

- Broker cleanup + journal-card workers follow the `_periodic_*` pattern exactly (`app/main.py:378-400`): inner async fn, `asyncio.create_task` in startup, re-read interval setting each tick, `SessionLocal()` per tick in try/finally, swallow-and-log all exceptions, sleep `max(10, interval)`.
- Enabled-gates use the string-tolerant truthiness idiom (`main.py:387-389`) — settings values are strings.
- Journal-card cadence reuses `routine_scheduler.is_due(schedule: dict, now_utc, last)` (`app/services/routine_scheduler.py:56-89`, duck-typed dict, lazy croniter import, returns False on malformed input). ⚠️ Its interval branch **baselines instead of firing on first sight** (`:64-65`) — correct for us; don't "fix" it.
- Household enumeration: the broker enumerates from its **own rows** (`attention_events` distinct household_id for cleanup; explicit household-scoped `attention.enabled` Setting rows for the journal card), following the direct-query precedent (`adapter_scheduler.py:379-387`) and the daily-briefing PRD's finding that the settings cascade cannot enumerate.

## The daily journal card (phase 1, zero mobile work — verified render path)

Posted via `post_inbox_item_sync` with `category="attention_journal"`:

- Unknown categories fall back to the generic InboxDetail screen (`InboxListScreen.tsx:43-54` special-cases only `adapter_proposal`, `adapter_deployed`, `interactive_list`; default `'InboxDetail'`); the category renders as a chip with underscores→spaces.
- Body renders as **Markdown**: `content_format` is a phantom field — mobile types it, but jarvis-notifications has no such column, so it is always null and `useMarkdown` is true (`InboxDetailScreen.tsx:201-202`; `models.py:56-69`). Write the card body in markdown; plain-text opt-out does not exist without backend work.
- Do **not** emit literal `<think>` tags (stripped into a "Show reasoning" collapsible, `InboxDetailScreen.tsx:51-59`); do not reuse `metadata.elapsed_seconds` (triggers a hardcoded research footer that renders `undefined` for missing keys).
- Card content: delivered items with rungs, withheld items with `withheld_by`, counts ("delivered 3, withheld 11 — 9 over source cap, 2 duplicates"). The **withheld count is the product**: it is what makes suppression visible instead of feeling dead.
- **Phase 1 ships the card view-only (no buttons).** Verified constraint: `interactive_elements` buttons dispatch `POST {cc}/api/v0/callbacks` → CallbackJob (5-min TTL from tap) → MQTT `callback` nudge → the **node named in `metadata.node_id`** GETs the payload and dispatches to a `@callback` on a **node-installed command** (`callbacks.py:54,130-173`; `mqtt_tts_listener.py:300-420`). A server-side broker has no node command to host the handler, and items without `node_id` show "Missing node context" on tap. Useful/Mute buttons therefore arrive in phase 2 alongside a new built-in node command (`attention_feedback`) whose `@callback` relays the verb to `POST /api/v0/attention/feedback` — set `navigation_type` explicitly and return success with no `context_data["inbox"]` title for a silent ack (CC only fans out a follow-up card when the title is a non-empty string, `callbacks.py:279-307`). Avoid `category="confirmation"` — it engages the legacy `metadata.actions` → `/api/v0/nodes/{id}/actions` path instead (`InboxDetailScreen.tsx:96-127`).

## Security

- **No new inbound MQTT trust.** Ingest is HTTP-only on existing auth (node X-API-Key via `verify_api_key`, or app-to-app). Household identity always comes from the validated node row, never the payload.
- **The future in-room rung must be nudge-then-pull, DB-backed.** Verified traps this PRD explicitly forbids inheriting:
  - `handle_tts` is completely unverified — anyone who can publish to the broker makes the node speak (`mqtt_tts_listener.py:111-112` drives TTS + LEDs with no `_verify_command`). Never extend it.
  - Only 3 of 15 node command handlers verify request_ids; "commands are verify-gated per the hardening work" is **not** an invariant — the dominant hardened pattern is nudge-then-pull.
  - `handle_action`'s verify can be bypassed by `details["trusted"]=true` **read from the untrusted payload itself** (`mqtt_tts_listener.py:203`). Do not copy.
  - `NodeCommandService`'s pending-request store is an in-memory per-process singleton with a 5-minute TTL and one-time-use verify; CC restart (or multi-worker uvicorn) silently fails all in-flight verifies. Broker invites use `attention_deliveries.request_id` + an authed GET, mirroring the DB-backed `CallbackJob` pattern (`callbacks.py`), not `NodeCommandService`'s dict.
  - New MQTT suffix topics must be added to the node's `.endswith()` dispatch chain **before** the commands-array fallthrough or payloads get parsed as command arrays and dropped.
- **Length caps enforced broker-side** (title ≤ 500 post-truncation with ellipsis, category ≤ 50) because notifications 5xxes rather than truncating.
- **Blast-radius limiter:** an absolute per-household per-day delivery ceiling (`attention.daily_push_budget + attention.daily_inbox_budget`) independent of per-source caps, so a compromised or buggy node cannot spam even inside per-source limits. Per-node rate limit on `/attention/*` ingest.
- Broker failure **fails open to legacy behavior**: if broker evaluation raises, the interposed endpoint logs and falls through to today's direct delivery. Governance is a feature; losing a medication push to a broker bug is not acceptable. (Inverse of the web-search fail-closed choice, deliberately: that gate controls egress; this one controls a safety-relevant delivery path.)

## Failure modes

| Failure | Behavior |
|---|---|
| `attention.enabled` false (default) | Interposed endpoints behave exactly as today; no events recorded |
| Broker evaluation raises | Log + fall through to legacy direct delivery (fail-open, by design — see Security) |
| notifications service down | Same as today: `post_inbox_item_sync` returns None, endpoint returns `sent=false`; delivery row records `failed` |
| Budget exhausted | Demote one rung (push→inbox→journal); `withheld_by="budget"`; visible in journal card |
| Duplicate within window | Journaled as `duplicate`, not delivered; safety-class exempt beyond exact-key match |
| `target_type="user"` with no `user_id` | Recorded explicitly in gate_trail, then household fallback (today's silent behavior, made visible) |
| CC restart | Events/deliveries durable in Postgres; no in-memory broker state to lose in phases 1–3 |
| Journal card worker raises | Swallow-and-log per worker convention; card skipped this cycle |
| Clock-skewed duplicate `run_date` for the card | Cron `is_due` + a `last_fired_at` column; a duplicate card is cosmetic, not correctness-critical (unlike daily-briefing's once-per-day constraint) |

## Phasing (patch releases, `v0.1.X`)

1. **Broker skeleton (CC only).** Migration (4 tables minus tiers activation); `POST /api/v0/attention/events` (node + app-to-app auth); interposition of the three endpoints behind `attention.enabled`; dedup; budgets + quiet hours from settings; gate_trail journaling; view-only daily journal card; TTL cleanup worker; `attention` settings + `test_settings.py` category fix. **Demo: install the unmodified news agent, watch it capped and journaled.**
2. **Feedback + tiers (CC + node patch).** `attention_feedback` + `POST /api/v0/attention/feedback`; built-in `attention_feedback` node command hosting the `@callback` for Useful/Not useful/Mute buttons on cards; voice verbs ("stop telling me about this", "why did you tell me that?" answered from gate_trail) as pre-routed no-LLM patterns — they must set `context_data["message"]` (pre-route rule); node forward-shim shipping `get_alerts()` output to the journal; scoring worker + T0–T3 tier activation with `state_reason`.
3. **Consent surface (CC + mobile).** `attention_consents` + household-admin allowlisted routes (mobile_household_settings pattern); mobile Attention screen (Agents-tab sibling): source × category ceiling picker, budget sliders, quiet hours, journal browser with withheld-items view, mute management. Until the screen lands, InteractiveList cards (SDK caps: 6 sections / 100 rows / 6 actions, `interactive.py:26-31`) cover mute/ceiling operations.

Later consumers (separate PRDs/tasks): verified LED-invite rung; daily-briefing digest reading the journal; standing-orders judgment proposing tiers.

## Testing

- Interposition equivalence: with `attention.enabled` false, byte-identical behavior for all three endpoints (golden-request tests against the current response shapes, including `sent=false` silent-drop paths).
- Gate order and demote-only: budget exhaustion demotes exactly one rung; nothing ever drops below journal; gate_trail records every gate in order.
- Dedup: same `(household, source, dedupe_key)` within window → `duplicate`; fallback title-hash path; safety-class exemption.
- Safety-class: categories in the list bypass tiers/budget/quiet-hours; feedback `mute` on a safety category is refused.
- Caps: title truncated to 500 before forwarding (no 5xx); category coerced ≤ 50.
- Fail-open: broker evaluation raising → legacy delivery still happens (assert the log line, per worker-swallow convention).
- Budget counters reset on household-timezone day boundary; DST fall-back day does not double-grant.
- `tests/test_settings.py` `expected_categories` includes `attention`.
- Phase 2: feedback verbs are pre-routed (no LLM), set `context_data["message"]`; callback silent-ack (no `inbox.title`) produces no follow-up card; tier decay is conservative (no thrash on a single dismiss).

## Open questions

- ~~Existing-caller category audit~~ **RESOLVED 2026-07-18:** the reminder agent explicitly sends `category: "reminder"` (`jarvis-node-setup/agents/reminder_agent.py:141`) and medication sends `category="medication"` (`jarvis-cmd-medication`) — both covered by the default safety list. Other verified `JarvisInbox` callers (jarvis-cmd-email's connection_health / email_alerts / reply_drafts, the export shopping/todo list commands, `mqtt_tts_listener.py`) are non-safety categories and correctly subject to caps. The request-default `"alert"` is not used by any safety caller; it stays out of the safety list.
- **Source identity is category-grade in phase 1.** Two packages sharing a category share a tier/cap. The clean fix (an SDK-declared `source` name validated against the installed-package registry) belongs to the SDK phase; decide whether category-grade is acceptable until then.
- **`/node/push-notification`'s dead `priority`/`category` fields** become load-bearing under the broker. That is a silent behavior change for any caller that set them expecting nothing — the audit above covers it.
- **Per-user budgets** are out of scope (household-level only); the personal-scope law covers the privacy edge until then.
- **Web/browser chat surface** (`jarvis-web`) inbox parity — does the journal card render acceptably there? (Not verified this pass.)

## Changelog

**2026-07-18 — initial, code-verified.** Notable verification findings baked in: `/node/push-notification` silently discards `priority`/`category` and hardcodes `category="confirmation"` + unconditional high-priority push downstream; inbox creation has zero dedup and push dedup is a 60s in-memory window missing `target_type`; node alert dedup is title-based (SDK `dedupe_key` unread); `content_format` is a mobile-side phantom (everything renders markdown); inbox buttons require a node-installed `@callback` command + `metadata.node_id` (hence view-only phase-1 card); `NodeCommandService` pending store is in-memory/5-min/one-shot (hence DB-backed invites); only 3/15 node handlers verify MQTT request_ids and `handle_action` has a payload-controlled bypass (hence "never cite verify-gating as an invariant"); `verify_household_role` defaults to `power_user`; alembic lives at repo root (CLAUDE.md tree wrong) with hand-written revision ids, head `d9c8b7a6e5f4`; `tests/test_settings.py` asserts exact category-set equality.

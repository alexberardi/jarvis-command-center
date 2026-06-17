# Daily Briefing

> **Status:** Scoped, not yet built. All load-bearing code claims below were **verified against the actual source on 2026-06-16** (jarvis-command-center, jarvis-llm-proxy-api, jarvis-web-scraper, jarvis-notifications, jarvis-node-mobile). Several reuse assumptions from the first draft were wrong and have been corrected — see **Changelog** at the bottom.

## Overview

A server-owned, per-household scheduled feature in jarvis-command-center. Once a day, it reads a configured list of topics, web-searches the top N results for each topic, scrapes those articles, summarizes each topic with the LLM, assembles the per-topic digests into a single briefing, and delivers it as one consolidated "Daily briefing" card (plus a push) in the jarvis-notifications inbox.

It **reuses the deep-research I/O primitives** (web search, SSRF-hardened scrape, async LLM-job queue, inbox/push delivery, the `Setting` cascade, the periodic-worker convention) but **the orchestration is genuinely new**: deep-research is strictly single-shot (one job → one callback → one inbox item), and there is **no map-reduce / fan-in / row-locking pattern anywhere in command-center today** (repo-wide grep for `with_for_update`/`FOR UPDATE` over `app/` returns nothing). So the copyable part is the per-topic *search → scrape → enqueue* loop and the callback Bearer-guard shape; the **completion barrier, the two tracking tables, the two-phase callback, and the once-per-day/idempotency machinery are net-new, correctness-critical code** and deserve their own design and concurrency tests.

The genuinely new code is one service module, two small tracking tables, one callback endpoint, one background worker, and a `briefing.*` settings group.

## Problem Statement

A user who wants to start the day informed has no first-party way to get a curated, personalized digest from Jarvis. Today they would have to ask voice questions one at a time, and voice answers are ephemeral and single-topic. There is no scheduled, multi-topic, glanceable morning summary.

We want: the user lists the things they care about ("AI policy", "Arsenal", "mortgage rates", "my industry"), and every morning Jarvis has already gone out, read the latest, and left a tidy briefing in their inbox.

## Goals

- Per-household scheduled daily run that produces one consolidated briefing in the inbox.
- Topics are **household-scoped and editable** in v1. **Per-user topic overrides are Phase 2** — see "Topics" below for why the existing `Setting` cascade cannot express a household-wide per-user override and why the settings router blocks user-self-editing.
- The pipeline fits the real deployment's **8192-token, single-slot, serial** LLM constraints — no design that assumes a large or dedicated context window. The char budget targets the **worst-case smallest plausible background context**, not a number that only holds on dev/Mac (see Key constraint).
- Reuse the existing deep-research / queue / scheduler / inbox machinery wherever the shape genuinely matches; treat the fan-in orchestration as new.
- Off by default; opt-in per household via a setting.
- **Fail-closed on auth and fail-safe on content**: the callback refuses unauthenticated calls; scraped content and emitted links are treated as untrusted (see Security).

## Non-goals

- **Live / on-demand briefing that the user waits for.** Generation is multi-second-to-minutes and serial-queued; it is always delivered async (notification when ready). A future "brief me now" trigger is fire-and-forget + notify, never a blocking call.
- **Spoken delivery in v1.** Delivery is the inbox card. Voice narration of the stored briefing is a later phase, gated on node presence.
- **Curated newsroom-quality headline selection.** v1 sources "top N" from web search results (see Open Questions for the RSS upgrade path).
- **A general scheduling/automation builder.** This is a single core feature with its own settings, not a new Routine type.
- **Per-user topic overrides in v1.** Deferred to Phase 2 (requires a dedicated resolver + a self-scoped mobile endpoint — the `/settings/*` router is superuser-only).

## Key constraint (why this is map-reduce)

Verified against jarvis-llm-proxy-api:

- The context window default is **8192** (`model.main.context_window`, default 8192 — `services/settings_service.py:158-166`).
- Queued jobs run on the `background` model alias. **On the default config** this resolves to the *same* llama.cpp instance and the *same* `n_ctx` as the live voice path: `managers/model_manager.py:217` gates sharing on `should_share = (bg_backend == live_backend and bg_model_path == live_model_path)`. ⚠️ This is **conditional** — a self-hoster (e.g. Linux/GPU prod) can point `model.background` at a separate model/URL/context window. **Do not assume 8192 is guaranteed by the alias.** Size the char budget to the worst-case smallest plausible context, and ideally read the effective background context at run time.
- The queue is **serial** — one RQ job at a time (single `rq.Worker` in `scripts/queue_worker.py:162`; a single worker process in every deploy config), sharing the single model slot with live voice. Background jobs briefly block voice while they run.
- There is **no server-side prompt-size guard**; llama.cpp silently truncates on overflow (`services/chat_runner.py:486-590` validates model/image/JSON-format only — no token or prompt-length check; the only truncation logic in that file is JSON *output* repair). Char/token budgets MUST be enforced client-side, or content is silently lost.

⟹ Putting ~15 scraped pages into one prompt is impossible. The briefing is summarized **map-reduce**: one map job per topic (each fits the context), then a small reduce job that assembles the digests.

## Architecture

```
_periodic_daily_briefing()            app/main.py startup worker (~60s tick, settings-gated)
    │  households = SELECT DISTINCT household_id FROM settings        ← see "Household enumeration"
    │               WHERE key='briefing.enabled' AND value truthy
    │                 AND node_id IS NULL AND user_id IS NULL
    │  for each household:
    │     + small per-household jitter (hash(household_id) → 0..N min) ← avoid 07:00 thundering herd (MVP)
    │     is_due(synthetic cron schedule, now_utc, last_fired_at)?     ← routine_scheduler.is_due
    ▼
run_briefing(household_id)            app/services/daily_briefing_service.py
    │  resolve topics (household-scoped Setting; per-user is Phase 2)
    │  if topics empty → log-and-skip, NO run, NO card                 ← empty-topics guard
    │  validate topics (list[str], trim, cap count + per-topic chars)
    │  INSERT DailyBriefingRun (unique (household_id, user_id, run_date)) ← once/day authority;
    │       on IntegrityError → "already ran today", skip                  catch it, don't rely on last_fired_at
    │  collect URLs across topics: _search_web(topic, N) → dedup by normalized URL (MVP, not Phase 2)
    │  scraper.batch_fetch(urls, max_concurrent=3, max_chars=briefing.max_chars_per_page)
    │  create N DailyBriefingTopic rows; for each: enqueue MAP job (≤3 pages → short digest)
    │       job_id = uuid4(), idempotency_key = uuid4()  (queue dedup is NOT the authority — see Idempotency)
    ▼  llm-proxy /internal/queue/enqueue  (model="background", serial)
POST /api/v0/daily-briefing/callback   app/main.py  (Bearer-gated, FAIL-CLOSED; phase via metadata)
    ├─ MAP callback (metadata.phase=="map", metadata.topic_index):
    │     authorize: payload job_id must match a known DailyBriefingTopic.map_job_id
    │     store digest on the topic row, mark done (or failed)
    │     atomic gate: UPDATE DailyBriefingRun SET status='reducing'
    │         WHERE id=:run AND status='mapping'
    │           AND (SELECT count(*) FROM topic WHERE run_id=:run AND status IN ('done','failed')) = topic_count
    │         → if this UPDATE affected 1 row AND ≥1 topic is 'done', THIS caller enqueues the reduce job
    │         → if 0 'done' topics, mark run 'failed', no reduce, no card
    └─ REDUCE callback (metadata.phase=="reduce"):
          idempotency guard: if run.inbox_item_id IS NOT NULL or status=='completed' → return, post nothing
          assemble consolidated markdown → SANITIZE links → post_inbox_item_sync(category="daily_briefing")
              (post_inbox_item_sync is SYNC → call via asyncio.to_thread from the async callback)
          set run.inbox_item_id + status='completed' in ONE committed transaction
          fire household push "Your daily briefing is ready"
```

Source of truth: the **inbox item** is the durable user-facing artifact (same as deep_research). The CC-side tables hold generation state only, for idempotency / once-per-day / stale-job recovery — not the briefing body.

> **Why an atomic conditional `UPDATE` instead of `SELECT … FOR UPDATE`:** the DB layer is **synchronous psycopg2** (`app/db.py:37,43`), and callbacks are `async def` endpoints that run blocking psycopg2 queries **directly on the single Uvicorn event loop** (there is no `run_in_executor`/threadpool offload anywhere in CC). A contended `FOR UPDATE` that waits on a held lock would stall *all* voice traffic. A single atomic conditional `UPDATE … WHERE status='mapping' AND <all terminal>` serializes the reduce-election without holding a lock across the async handler. Row locking is used nowhere in CC today, so either approach is new code — prefer the non-blocking one.

## Household enumeration

The worker loop "for each household with `briefing.enabled`" is **not achievable via the settings client cascade** and was the single biggest unspecified mechanism in the first draft:

- CC has **no Household table** — `household_id` is a denormalized `String` column on ~20 feature tables.
- The settings client `get(key, household_id=…)` is a *per-household* cascade lookup requiring you to already know the `household_id`; `list_all()` collapses all rows into a dict keyed by `setting.key` (last row wins), discarding the household grouping. Neither can answer "which households set this key truthy."
- Auth's only full-list endpoint (`/superuser/households`) is superuser-JWT-gated and uncallable by a headless worker.
- Precedent: the existing per-household workers enumerate from **CC-local domain data**, never from "who set a setting" — `adapter_scheduler.list_active_households()` does `db.query(ConversationTranscript.household_id).distinct()` (`adapter_scheduler.py:379`); `routine_scheduler` iterates `Routine` *rows*. Briefings are a *setting on a non-enumerable key*, not rows.

**Decision:** enumerate by querying CC's own `settings` table directly (the `Setting` model lives in `app/models.py:289`, so this is a plain ORM query, not the settings client):

```sql
SELECT DISTINCT household_id FROM settings
WHERE key = 'briefing.enabled'
  AND node_id IS NULL AND user_id IS NULL      -- household scope only
  AND value IN ('true','1','yes')              -- truthy, matching bool coercion
```

This **intentionally bypasses the cascade** and **does not honor a system-default enable** — a household must have an explicit household-scoped `briefing.enabled=true` row to be picked up. Document this; it is a deliberate constraint, not a bug.

## Data model

### Topics — `Setting` table (household scope only in v1)

Stored as a `value_type="json"` setting on the existing table (`app/models.py:289`; `value_type` column at `:307`, scope columns `household_id`/`node_id`/`user_id` at `:315-317`, unique constraint on `(key, household_id, node_id, user_id)` at `:323`).

- `briefing.topics` — JSON list of topic strings, **household default**: `household_id` set, `node_id` NULL, `user_id` NULL. Resolved at run time via `settings_service.get("briefing.topics", household_id=…)`.
- ⚠️ **`briefing.topics` would be the first `value_type="json"` setting in this service** (all 50 existing definitions are string/int/float/bool). The library supports it (`coerce_value`/`serialize_value` handle JSON), but there is no in-repo precedent — test the coerce/serialize round-trip explicitly.

**Per-user override is Phase 2, not v1**, because the existing machinery cannot deliver it:

1. **The cascade can't express it.** The settings cascade resolver (`jarvis-settings-client/.../service.py:167-244`) has exactly four levels and its *user* level **requires `household_id` AND `node_id` AND `user_id` all set** — there is no "household + user, `node_id` NULL" level. So a household-wide per-user override is never matched by `get(household_id=H, user_id=U)`. And `list_all()` keys by `setting.key`, so multiple `briefing.topics` rows (default + N user overrides) *collide*.
2. **The settings router blocks user self-editing.** `/settings/*` is wired with `create_superuser_auth` for writes **and** combined-auth that routes Bearer JWTs to the superuser dependency for reads (`main.py:188-192`, `auth.py:41-123`) — a normal user cannot even *read* their own `briefing.topics`, let alone write it.

Phase 2 delivers per-user topics with (a) its own resolver — a direct `Setting` query for `(household_id=H, user_id=U, node_id NULL)` falling back to the household row — and (b) a self-scoped `/api/v0/mobile/briefing-topics` endpoint with `mobile_memories`-style role gates (`_can_read`/`_can_write`), **not** the superuser settings router.

### `DailyBriefingRun` (new table + alembic migration)

| column | type | notes |
|---|---|---|
| `id` | PK | |
| `household_id` | str | |
| `user_id` | int, nullable | reserved for Phase 2 per-user runs; **always NULL in v1** |
| `run_date` | str `YYYY-MM-DD` | in household timezone |
| `status` | enum | `pending` → `mapping` → `reducing` → `completed` / `failed` |
| `topic_count` | int | expected number of map digests (the fan-in target) |
| `reduce_job_id` | str, nullable | |
| `inbox_item_id` | str, nullable | set on reduce callback; **also the double-card idempotency guard** |
| `attempt_count` | int, default 0 | hard cap so stale-reset can't loop a deterministically-failing run forever |
| `last_fired_at` | datetime | feeds `is_due` cadence — **NOT** the dedup authority |
| `started_at` | datetime | for the 30-min stale-job reset |

**Unique constraint on `(household_id, user_id, run_date)` is the once-per-day authority.** The fire path INSERTs and catches `IntegrityError` as "already ran today, skip" — do not rely on `last_fired_at` for dedup (it's only for cadence; `is_due` reads it via its `last` arg). This makes the guard DST-safe: a local `run_date` never repeats, so even a cron time inside a fall-back-repeated hour is deduped by the unique constraint.

### `DailyBriefingTopic` (new child table)

One row per topic per run — avoids the read-modify-write race on a shared JSON column during fan-in.

| column | type | notes |
|---|---|---|
| `id` | PK | |
| `run_id` | FK → DailyBriefingRun | |
| `topic` | str | |
| `topic_index` | int | carried in callback `metadata.topic_index` to route the digest back |
| `map_job_id` | str, nullable | **authorization check**: an inbound map callback's `job_id` must match a row here |
| `status` | enum | `pending` / `done` / `failed` |
| `digest` | text, nullable | the per-topic summary; may be nulled after reduce (it's duplicated in the durable inbox item) |
| `source_urls` | json | `[{title, url}]` actually scraped, for citations. Read from `ScrapedPage.title`/`.url`; skip `page.ok == False` |

## Settings (`app/services/settings_definitions.py`, category `briefing`)

| key | type | default | purpose |
|---|---|---|---|
| `briefing.enabled` | bool | `false` | master switch (per-household; must be a household-scoped row to be enumerated) |
| `briefing.time_cron` | string | `"0 7 * * *"` | daily fire time |
| `briefing.timezone` | string | household tz | cron evaluated in this tz |
| `briefing.topics` | json | `[]` | the topic list (household scope in v1) |
| `briefing.urls_per_topic` | int | `3` | "top N" headlines (lower = safer token budget) |
| `briefing.max_chars_per_page` | int | `3000` | per-page scrape cap (tightened from the first draft's 4500 — see Token budget) |
| `briefing.max_topics` | int | `8` | hard cap on topic count |
| `briefing.max_topic_chars` | int | `120` | per-topic length cap (DoS / budget protection — the first draft named "length" but never bounded it) |
| `briefing.max_prompt_chars` | int | `10000` | total assembled map-prompt char ceiling (conservative total-prompt guard — see Token budget) |
| `briefing.scheduler_interval_seconds` | int | `60` | worker tick (floor 30) |
| `briefing.jitter_minutes` | int | `20` | per-household fire spread to avoid the 07:00 herd (MVP, not Phase 4) |
| `briefing.delivery_mode` | string | `"inbox"` | `inbox` \| `voice` \| `both` (voice = later phase) |
| `briefing.run_ttl_days` | int | `7` | TTL for run/topic cleanup |

> Adding the `briefing` category breaks `tests/test_settings.py:55-62`, which asserts the category set by **exact equality** (this is what commit `76d5cd1` updated for `routines`). Add `briefing` to `expected_categories` in the same change or CI fails.

## Token budget (per map job — must fit the worst-case background context)

**The first draft's 3.75 chars/token assumption was unsafe.** Measured chars/token on representative scraper output: clean English prose ~6.3, realistic news with URLs/markdown/tables/tickers ~2.58, non-English/CJK ~1.62. At 4500 chars/page × 3, a URL-dense or CJK-heavy day produces a map prompt that **overflows 8192 → silent truncation** (no server-side guard, per Key constraint).

Budget conservatively at **~1.6–2.0 chars/token**, not 3.75:

- Reserve: output `max_tokens` 300 + fixed overhead ~200 tokens (system prompt — the real deep-research one is ~85 tokens, not 250; + llama-3 chat-template special tokens ~40; + per-source `## Source / URL:` headers ~60). → ~500 tokens reserved.
- Safe input ceiling ≈ 7000 tokens (leaves margin under 8192).
- At **`max_chars_per_page=3000` × `urls_per_topic=3` = 9000 input chars**: worst-case 1.6 chars/token → ~5,600 tokens; + ~500 reserved → **~6,100 total. Safe** even on CJK-heavy content.
- **Per-page char cap alone is NOT sufficient** (chars→tokens varies 4×; long source URLs and headers aren't counted; a future bump to `urls_per_topic` removes the protection). **Enforce a total-prompt guard before enqueue:** assemble the full message, and if its char length exceeds `briefing.max_prompt_chars` (10000, ≈6250 tokens at the conservative 1.6 ratio), trim/drop sources until it fits. (Tokenizing exactly would be better if a cheap local tokenizer is available; the char ceiling is the pragmatic guard.)
- **Two full topics in one job would overflow → silent truncation.** Per-topic is the map unit; do not batch topics.

Reduce job: digests are *generated* at `max_tokens≈300` each (not ~120), so with `max_topics=8` the reduce input is **up to 8 × 300 = 2,400 tokens** (the first draft's "well under 1k" was wrong). Plus a small assembly prompt + `max_tokens≈700` output → still comfortably under 8192. Cap stored `digest` length so reduce input stays bounded regardless of a map overrun.

## Idempotency & scheduling

**The DB row is the dedup/once-per-day authority — not the queue.** The first draft inverted this:

- The queue dedup key is **composite**: `f"llmproxy:dedupe:{job_id}:{idempotency_key}"` (`jarvis-llm-proxy-api/queues/redis_queue.py:30`), set with a short TTL (deep-research uses `ttl_seconds=600`). The clone target sets `job_id == idempotency_key == str(uuid4())` (`deep_research_service.py:231,249`) and so **never dedups at the queue layer at all**. A deterministic `idempotency_key` alone buys nothing unless `job_id` is *also* deterministic — and even then the 600s dedup TTL is far shorter than the 30-min stale reset, so it neither reliably blocks a double-fire nor permits a retry. The first draft's "never use a random UUID (documented deep_research gotcha)" was backwards.
- **Use random `uuid4()` for both `job_id` and `idempotency_key`** (matching the proven deep-research/memory-extraction pattern), and make correctness rest entirely on:
  1. **`UNIQUE(household_id, user_id, run_date)`** — the once-per-day guard (INSERT + catch `IntegrityError`).
  2. **The atomic conditional `UPDATE … WHERE status='mapping' AND <all terminal>`** — elects exactly one reduce.
  3. **The `run.inbox_item_id` guard** in the reduce callback — prevents a double card.

Scheduler reuses `routine_scheduler.is_due(schedule, now_utc, last)` (`app/services/routine_scheduler.py:56`, fully duck-typed — reads only `schedule.get("type"/"cron"/"timezone"/"enabled")`) with a synthetic `{type:"cron", cron, timezone, enabled:true}` dict per household. **Pass `last_fired_at` as the 3rd `last` argument** — a `last_fired_at` key *inside* the dict is inert (`is_due` ignores it). croniter is imported lazily with an `ImportError` guard that returns `False` (worker stays alive); it uses ZoneInfo. There is a `_mark_fired` helper (`:92`) but it persists into the `Routine` JSON column — the new typed `last_fired_at` column mirrors the concept, not the mechanism, so write it directly.

**Stale-job recovery:** a run stuck in `mapping`/`reducing` with no progress >30 min (by `started_at`) is reset on the next tick so it can retry that day — *bounded by `attempt_count`*. The cited mirror, `transcript_service.reset_stale_jobs` (`transcript_service.py:115`, 30-min default), detects staleness by `created_at` and just clears the in-flight job id; it gives **no** guidance for resetting a half-done fan-in. Reset logic must: re-enqueue only `pending`/`failed` topic map jobs (not `done` ones), with fresh `uuid4()` jobs; increment `attempt_count`; mark the run `failed` once the cap is hit.

## Concurrency & fan-in (net-new — no precedent in CC)

- **Gate on "all topics TERMINAL", not "all done".** Topic status is `pending`/`done`/`failed`; gating on `done` alone means one permanently-failed topic hangs the run forever. The reduce is elected only when `count(done)+count(failed) == topic_count`, and requires **≥1 `done`** (else mark run `failed`, mirroring deep-research's "could not scrape any" path).
- **Reduce election is a single atomic `UPDATE`** (see Architecture note) — no held lock, so it can't block the event loop.
- **The reduce callback is idempotent**: it checks `inbox_item_id`/`status=='completed'` before posting and writes `inbox_item_id` + `status` in one committed transaction. The notifications inbox has **zero idempotency** (`jarvis-notifications/app/api/inbox.py:221-238` inserts a fresh row every call) and push dedup is only 60s (`notification_service.py:25`), so without this guard a stale reset or a crash-after-post-before-status-write yields a duplicate card.

## Delivery

The REDUCE callback stores an inbox item via `inbox_notification_service.post_inbox_item_sync` (`app/services/inbox_notification_service.py:192`):

- ⚠️ `post_inbox_item_sync` is **synchronous** (blocking `httpx.Client`). The callback is an `async def` endpoint — **call it via `asyncio.to_thread`** so it doesn't block the event loop. (Note: deep-research does *not* use this helper; it has its own private async `_store_inbox_item`/`_send_notification`, the latter hardcoded to household-scoped push. We deliberately use `post_inbox_item_sync` because it bundles inbox + push + failure-swallow — this is a divergence from the "clone", not reuse.)
- `category = "daily_briefing"`, `title = "Daily briefing — {date}"`. Category is free-form server-side (notifications stores any string ≤50 chars — no allowlist), so no registration is needed there.
- `summary` = first ~200 chars (think-block stripped) for the card preview.
- `body` = full sectioned markdown (one section per topic, 2–3 bullets + inline `[Title](URL)` citations) — **after link sanitization** (see Security).
- `metadata`: render-shape is **net-new mobile work, not free reuse.** The existing `InboxDetailScreen.tsx:183` reads `metadata.sources` as a **flat array** `[{title,url}]`, and there is no per-topic collapsible card component. Either emit `metadata.sources` as a flat array for v1 (degrades into the existing renderer; unknown category falls through to the generic `InboxDetail` screen with a default chip color — no crash) and defer the per-topic collapsible card to Phase 2, or build the mobile component. Do **not** assume `metadata = {sources: {topic: [...]}}` renders for free.
- Push: **household-targeted** in v1 (`target_type='household'`). Per-user push is Phase 2 (would use `post_inbox_item_sync(target_type='user', user_id=…)`).
- Scoping correctness: a stored inbox item with `user_id` NULL is visible to **every** household member (`inbox_service.list_items:60` filters `user_id == me OR user_id IS NULL`). v1 is household-wide, so NULL is correct; when Phase 2 adds per-user runs, setting `user_id` is what prevents cross-user leakage.

## Security

Input-side network posture is strong; the gaps are **content-side and auth-side**.

### Network / SSRF (verified solid — keep)
- All scraping goes through `WebScraper.batch_fetch` with `block_private_hosts=True` (default, `jarvis-web-scraper/.../models.py:17`). Every redirect hop is validated and DNS-resolved against the private/loopback/link-local/RFC1918/CGNAT/IPv6 blocklist (`fetcher.py:145,175`). Keep `block_private_hosts=True`. Keep `max_concurrent=3` (the scraper has no robots.txt compliance; bumping risks 429/IP bans).
- If any non-batch fetch is added, gate with `quick_search_tool._is_blocked_host()` (the hardened helper added in commit `244ff3b`, `app/core/tools/quick_search_tool.py:54`).

### Callback auth — FAIL CLOSED (changed from the first draft)
- `/api/v0/daily-briefing/callback` validates `Authorization: Bearer {JARVIS_ADAPTER_CALLBACK_TOKEN}`. The sibling guards (`/deep-research/callback` at `main.py:1479`, check at `:1485`, wrapped in `if callback_token:` at `:1483`) are **fail-open**: an unset token = an open endpoint that accepts forged callbacks. Since the callback payload directly controls the inbox card body/title and a household-wide push fan-out, a missing-token prod deploy lets any unauthenticated caller inject inbox content + push spam to every household.
- **This feature's callback must fail CLOSED:** if `JARVIS_ADAPTER_CALLBACK_TOKEN` is unset, refuse the callback (503) and/or refuse to enable briefing. Do not clone the `if callback_token:` opt-in. (Worth retrofitting the existing deep-research and memory-extraction callbacks too.)
- **Authorize the payload, not just the bearer:** the map callback's `job_id` must match a known `DailyBriefingTopic.map_job_id`, and the reduce's must match `DailyBriefingRun.reduce_job_id` — reject callbacks for jobs this service never enqueued. The `DailyBriefingRun` row is a **required authorization check**, not merely a double-fire guard.
- Enqueue calls carry `X-Internal-Token = LLM_PROXY_INTERNAL_TOKEN`.

### Untrusted scraped content — prompt injection (NEW; was unaddressed)
- Scraped page text flows **verbatim** into the map user prompt (the clone concatenates `page.text_content` into the user message — `deep_research_service.py:62-78`; the extractor strips `<script>`/`<style>` but preserves all visible prose). News pages are attacker-influenceable (planted article, compromised site, SEO spam, surviving comment text), so "ignore previous instructions…" reaches the model. Blast radius: poisoned digest → reduce → inbox `body` → daily push to the whole household, no human in the loop.
- Mitigations (defense-in-depth; input trust is impossible for arbitrary web pages, so containment is **output-side**): (1) wrap each source in clearly delimited, labeled boundaries and instruct the system prompt to treat everything inside as DATA, never instructions; (2) prefer the search-result `snippet` over full scrape where it suffices, to shrink the surface; (3) sanitize the LLM output before it becomes a card (below); (4) document prompt injection as an accepted residual risk with output-side containment.
- (The first draft's claim that topics are "never interpolated into the system prompt as instructions" is moot here — the real surface is scraped *content* in the user message, not the topic string.)

### Output / link injection into the inbox card (NEW)
- The reduce emits `[Title](URL)` where `Title`/`URL` come from untrusted search results and LLM output, with **zero sanitization** at every hop (CC → notifications stores raw → mobile renders markdown). Mobile defaults to markdown rendering (`InboxDetailScreen.tsx:202`, `content_format` is null) and the default link rule calls `Linking.openURL(href)` directly — **no confirmation, no scheme/host allowlist**. A poisoned `[Your bank — verify now](https://evil/phish)` becomes a tappable phishing link delivered daily.
- **Required:** before storing the body, sanitize citations — allow only `http(s)` schemes, restrict link hosts to the actually-scraped source hosts (cross-check `DailyBriefingTopic.source_urls`), reject `javascript:`/`data:`. Set `content_format` deliberately. Coordinate a mobile-side `onLinkPress` confirm handler (a footgun for *all* markdown inbox items, not just briefings).

### Privacy — Jina.ai fallback (NEW emphasis)
- On 401/403 **or timeout** (a common path), the scraper proxies the full target URL to `https://r.jina.ai/{url}` (`fetcher.py:215-222`), leaking which articles a household reads — and by inference its configured topics (health, finance). For a project whose Core Principle #1 is "fully private, no cloud dependencies by default", this contradicts the promise.
- **Make external proxying opt-in, default OFF** (a `briefing.allow_external_proxy` / scraper flag). When off, a fetch that would fall back to Jina instead marks the topic source failed (the failure-mode table already handles "all scrapes fail" gracefully). At minimum, document the leak prominently rather than asserting it "acceptable" on the user's behalf.

### Topic input validation (NEW)
- Topics are untrusted free-form JSON. Enforce: list-of-strings typing, `max_topics` count cap, `max_topic_chars` length cap, whitespace trim — before a topic enters the search call or the prompt. Keep topics in the **search-query position**; they are not instructions.

### Blast-radius limiter
- Consider a per-household per-day delivery cap independent of idempotency, so a logic bug or accepted injection can't spam.

## Reuse map

| New piece | Reuses / clones | Notes |
|---|---|---|
| per-topic `_search_web` + `batch_fetch` + enqueue loop | `deep_research_service.py` (`_search_web:194`, `_scrape_urls:216` calling `batch_fetch(max_concurrent=3)`, `_enqueue_summarization:224`) | Copyable. `batch_fetch` returns `list[ScrapedPage]`; read `.text_content`/`.title`/`.url`, check `.ok`. deep-research hardcodes `max_chars=6000` — we wire `max_chars_per_page` (default 3000) instead. |
| map/reduce fan-in, completion barrier, 2 tracking tables, 2-phase callback | **nothing — net-new** | No `FOR UPDATE`/map-reduce anywhere in CC; deep-research is single-shot. Budget as fresh work + concurrency tests. |
| `/api/v0/daily-briefing/callback` Bearer shape | `/deep-research/callback`, `app/main.py:1479` (check `:1485`) | But **fail-closed**, unlike the clone. Callback envelope is `{job_id,job_type,finished_at,status,result,error,timing,metadata}`; digest is `result.content`; arbitrary `metadata` round-trips (`queue_models.py:77`, `tasks.py:324-333`). |
| `_periodic_daily_briefing()` worker | `_periodic_routine_scheduler`, `app/main.py:360` | Same pattern (per-tick interval read, exception swallow, `asyncio.create_task` in startup). |
| TZ-aware daily cron | `routine_scheduler.is_due`, `app/services/routine_scheduler.py:56` | Synthetic cron dict; pass `last` as arg 3. |
| household enumeration | **net-new** — direct `Setting` query | Cascade/`list_all` can't enumerate; see Household enumeration. |
| topic storage (household) | `Setting` cascade, `app/models.py:289` | First `json` setting; per-user override is Phase 2. |
| TTL cleanup worker | `_periodic_transcript_cleanup:215` / `_periodic_trace_cleanup:234` / `_periodic_routine_execution_cleanup:385` | Same daily/hourly delete-by-cutoff loop for `run_ttl_days`. |
| inbox + push | `inbox_notification_service.post_inbox_item_sync:192` | **Sync** → wrap in `asyncio.to_thread`. NOT deep-research's private helpers. |
| stale-job reset | `transcript_service.reset_stale_jobs:115` (concept only) | Mirror gives no fan-in guidance; new half-done-reset logic + attempt cap. |
| Phase-2 mobile topic CRUD | `mobile_memories.py` role gates (`_can_read:97`/`_can_write:106`) | Self-scoped, JWT — not the superuser settings router. |

## Failure modes

| Failure | Behavior |
|---|---|
| Topics list empty (default `[]`) | Log-and-skip: no run, no card, no push. (Surface "no topics configured" at most once, not daily.) |
| A topic returns 0 search results / all scrapes fail | Skip the topic; pass a note so reduce can say "No notable headlines for X today." Mark the topic row `failed`. |
| ALL topics yield no scrapeable content (0 `done`) | Mark run `failed`, no inbox item (mirror deep_research's "could not scrape any"). |
| ddgs search rate-limited (`RatelimitException`) | Catch **distinctly** (don't lump into the bare `except` that returns `[]`); back off; treat as a **retryable run failure**, not a per-topic "no news" (a throttle otherwise masquerades as empty results across the whole feature). |
| One topic permanently fails | Reduce still fires once all topics are **terminal** and ≥1 is `done`. Does not hang. |
| Map job callback never arrives | Stale-job reset after 30 min re-enqueues only the missing map jobs (fresh uuids) → retry same day, bounded by `attempt_count`. |
| Reduce callback retried / re-enqueued | `inbox_item_id`/`status=='completed'` guard returns early → **no duplicate card**. |
| Concurrent map callbacks racing to reduce | Atomic conditional `UPDATE` elects exactly one (no held lock → no event-loop stall). |
| Same-day re-fire | `UNIQUE(household,user,run_date)` INSERT raises `IntegrityError` → skip. |
| notifications service down | `post_inbox_item_sync` returns None (logged, non-fatal); run still completes its tables. **MVP must log a per-run terminal-state line** so this isn't silent (workers swallow exceptions — CLAUDE.md invariant #10). |
| Background model points at a sub-8192 context (custom prod) | Total-prompt char guard + worst-case budget protect against silent truncation; ideally read effective context at run time. |
| Worker raises | Loop logs and continues (CC convention) — needs explicit log assertions/tests so a logic bug isn't silent. |

## Phasing

1. **MVP — scheduled household inbox briefing.** `briefing.*` settings + household-scoped `Setting` topics; `daily_briefing_service.py` (household enumeration → topic loop → search top N → cross-topic URL dedup → `batch_fetch` → **total-prompt-guarded** per-topic map enqueue → terminal-gate fan-in → reduce); `DailyBriefingRun` + `DailyBriefingTopic` + migration; **fail-closed** Bearer callback with map/reduce fan-in + reduce idempotency guard; `_periodic_daily_briefing()` worker **with per-household jitter**; consolidated inbox card (flat `metadata.sources`) + household push; **link sanitization + topic validation + scraped-content containment**; **minimal observability** (per-run terminal-state log + `last_status`/`last_completed_at`); run/topic TTL cleanup. Default `enabled=false`. Tests below.
2. **Configuration UX + per-user.** Per-user topic resolver (direct `Setting` query) + `/api/v0/mobile/briefing-topics` GET/PUT (role-gated, self-scoped — **not** the superuser router); per-user runs + per-user push targeting; mobile screen to edit topics/toggle/time; per-topic collapsible card component + nested `metadata.sources`.
3. **On-demand + voice.** Optional `daily_briefing` server tool ("brief me now") — fire-and-forget + notify; optional scheduled voice narration to the household primary node over MQTT, gated on `Node.is_online()`; `briefing.delivery_mode`.
4. **Hardening / observability.** Latency/trace logging (scrape-vs-LLM time) like deep_research; smarter per-household fire-time staggering; extract shared web-synthesis helpers into `app/services/_web_synthesis.py` shared with deep_research; RSS upgrade for known categories.

## Testing

- Topic-loop scrape with mocked `_search_web` + `batch_fetch` (mock at the `rest_client.post` boundary for enqueue).
- **Household enumeration**: the direct `Setting` query returns only households with a truthy household-scoped `briefing.enabled` row; ignores system-default and per-user rows.
- **Once-per-day**: `UNIQUE(household,user,run_date)` INSERT on a same-day re-fire raises `IntegrityError` and is handled as skip. Cover a **non-UTC** household tz across a day boundary and a DST fall-back day.
- **Fan-in**: N map callbacks → reduce enqueued **exactly once** via the atomic conditional `UPDATE`; one permanently-`failed` topic still reduces; **0 `done` topics → run `failed`, no card**.
- **Reduce idempotency**: a second reduce callback (or stale re-enqueue) posts **no** second inbox item (`inbox_item_id` guard).
- **Callback fails CLOSED**: unset `JARVIS_ADAPTER_CALLBACK_TOKEN` → callback rejected (not accepted). Bad/missing bearer → 401. Payload `job_id` not matching a known job → rejected.
- **Token budget**: assert the **total assembled map prompt** (chars, conservative ratio) stays under the ceiling for adversarial content (CJK, URL/table-dense) — not just the per-page cap.
- **Security**: emitted citation links with non-`http(s)` schemes or off-source hosts are stripped/neutralized; scraped content with injection strings is wrapped/contained and does not alter card structure.
- **Empty/invalid topics**: `[]` → no run, no card; non-string / over-long topics rejected or trimmed.
- **`value_type="json"` round-trip** for `briefing.topics` (coerce + serialize) — no in-repo precedent.
- **`tests/test_settings.py` `expected_categories`** updated to include `briefing`.
- **Worker resilience**: a raising tick logs and continues; assert the log line (so a silent logic bug is caught).
- Inbox item shape (category, title, summary, body, flat `metadata.sources`) on reduce.

## Open questions / deferred

- **Headline quality + summary quality (ship gate).** v1 "top N" = top N DuckDuckGo results for the topic string — flexible but not curated/dated. Compounding risk: a local 3B/8B model summarizing 3 arbitrary results, with first-`max_chars` truncation that often captures nav/boilerplate. A daily *push* that reads as low-signal trains users to ignore it (worse than deep-research, which is user-pulled). **Define a concrete pre-GA quality bar** (manual eval of N real topics, or a dogfood week) and a rollback if it's noise. Upgrade path: route known categories (tech/sports/business/science/health/general) through the existing RSS `fetch_news_headlines` (`agent_service.py`), web-search the rest.
- **Cost/latency budget.** Add an estimate: jobs/day = `(topics + 1) × households`, est. tokens + wall-clock, total daily background-inference minutes — so the serial-queue-vs-voice-contention tradeoff is quantified before GA.
- **MVP scale ceiling.** Even with jitter, the serial single-slot queue bounds how many households × topics can fire before voice contention is user-visible. State the ceiling; pull smarter staggering forward if exceeded.
- **Migration rollback.** Specify the alembic downgrade for both new tables (alembic lives at the **repo root**, not under `app/`, despite the CLAUDE.md tree).
- **Voice delivery presence rules** (which node, empty-room avoidance) — Phase 3.

## Changelog

**2026-06-16 — code-grounded review.** First draft was well-cited but had load-bearing errors corrected here:
- *Wrong claims fixed:* the idempotency/"deep_research gotcha" was backwards (queue dedup key is `{job_id}:{idempotency_key}`; deep-research uses random uuids and never dedups) → DB row is now the authority; the `Setting` cascade cannot express a household-wide per-user override and the settings router is superuser-only → per-user moved to Phase 2; "for each household with briefing.enabled" had no implementation path → explicit `Setting`-table enumeration added; "background = same n_ctx" is only true on default config; "topics never interpolated into the prompt" missed that scraped *content* is the real injection surface.
- *Net-new (was framed as reuse):* the map-reduce fan-in, completion barrier, two tables, and two-phase callback — no precedent in CC.
- *Correctness added:* terminal-state fan-in gate (not "all done"), atomic conditional `UPDATE` instead of event-loop-blocking `FOR UPDATE`, reduce-callback `inbox_item_id` idempotency guard, `IntegrityError` once-per-day handling, `attempt_count` cap, empty-topics short-circuit, ddgs rate-limit handling, `asyncio.to_thread` for the sync inbox helper.
- *Security added:* fail-closed callback + payload authorization, scraped-content prompt-injection containment, citation-link sanitization (+ mobile `onLinkPress`), Jina.ai exfiltration opt-in/default-off, topic input validation, total-prompt token guard.
- *Budget:* chars/token re-derived at ~1.6–2.0 (not 3.75); `max_chars_per_page` 4500 → 3000; reduce math corrected.
- *Pulled into MVP:* per-household jitter, cross-topic URL dedup, minimal observability, TTL cleanup, total-prompt guard.

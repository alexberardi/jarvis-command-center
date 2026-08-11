# PRD: Signal Bus + Situation Matcher

**Owner:** jarvis-command-center · **Status:** Draft for build · **Date:** 2026-08-08

---

## 1. Summary, Goals, Non-goals

**Summary.** Today jarvis-command-center pivots behavior off hardcoded, per-scenario handlers (the weather/reminder/calendar ambient categories at `conversation_handler.py:2789`; the email→calendar proposal pilot in `proposal_matcher.py`). This PRD introduces a **generic** mechanism: behavior is computed as **{installed capabilities} × {current signals}**. A **Signal** is a small, typed, deduplicated fact ("Alex is home", "meal plan finished", "package delivered") written into a dedicated `signals` table by any producer — a node agent, a phone geofence, one of Jarvis's own microservices, or an external system holding a household-scoped API key. Two triggers consume signals: a **reactive** turn-time renderer (generalizes `_assemble_ambient_bundle` so every non-expired Signal is visible to the LLM, cache-safely) and a **proactive** event-driven reasoner (generalizes `match_proposals` into `match_situation`, runs off the live inference slot on the background model, and only ever proposes a tap-to-confirm card). Presence is just the first producer, not a special case; genericity is proven by ≥2 producers (voice-derived presence + synthetic curl) before any phone work.

**Goals.**
- One core abstraction (the Signal) and one dedicated store, replacing ad-hoc `UserMemory` category rows for situational facts.
- Reactive rendering that is byte-stable and preserves the prefix-cache / TTFS wins from commits #100 and `35c9b8d`.
- A generic proactive reasoner that reuses the shipped A–F confirm chokepoint verbatim — the reasoner *physically cannot* execute a tool.
- An ingress that lets external systems and internal services PROPOSE (never execute) via node-auth or app-to-app — **no per-source registration in core** (a third party integrates by POSTing, zero core update); the command boundary comes from the default node's live advertisement, plus in-memory rate limiting.
- Ship value incrementally: **reactive-only first** (high value, cache-safe, no LLM tick, no autonomy).

**Non-goals.**
- **No auto-execute — propose-only for now.** Every inbound signal, from any source, lands as a tap-to-confirm card. Autorun (letting trusted internal producers skip the tap for reversible/low-blast actions) is an explicit **future** extension we design toward but do not build here (Section 7).
- **Home Assistant is not a dependency.** HA is one optional producer among many; the design must work with zero HA present.
- **Acoustic / fall detection is PARKED.** Audio is a weak sensor for falls; accelerometer/mmWave is the real tool. No acoustic producer ships here.
- Not a Redis/streaming bus, not a durable event log. The `signals` table is the source of truth; TTL-cleanup mirrors the memory worker.

---

## 2. The Signal

### 2.1 Schema (the core abstraction)

A Signal is `{ kind, source_key, scope, subject, summary, facts, observed_at, ttl_seconds, cacheable, salience?, source_agent }`:

| Field | Type | Meaning |
|---|---|---|
| `kind` | dotted label, e.g. `presence.seen`, `meal_plan.completed` | **A descriptive label, NEVER control flow.** The matcher and renderer read it; no code branches on specific kind values (the one weather special-case at `conversation_handler.py:2803-2813` is *deleted*, not ported). |
| `source_key` | stable string id | **Dedup + suppression key.** UPSERT target. Same logical fact re-observed → same `source_key` → one row. This is the exact key `_handle_suppress` blocklists (`proposable_action_service.py:346`) and `record_suppression` stores (`proposal_suppressions.py:37`). |
| `scope` | `{household_id, user_id?, node_id?, room?}` | `household_id` required and server-trusted; `user_id` NULL = household-wide (mirrors UserMemory's NULL convention, `memories.py:186`). |
| `subject` | string | Sub-key within a `kind` so multiple facts per kind survive (e.g. `kind=weather subject=current` vs `subject=forecast`). Stable-sort secondary key. |
| `summary` | NL line | The human/LLM-readable line the reactive renderer prints and the matcher reasons over. |
| `facts` | typed JSON payload | Structured data copied verbatim into proposed action args by `match_situation`. |
| `observed_at` | timestamp | When the producer observed the fact. |
| `ttl_seconds` | int | Expiry horizon; NULL/absent = producer default. Drives `expires_at`. |
| `cacheable` | bool, **producer-owned, default false** | May this Signal ride the cached prefix `messages[0]`? Enforced at the write boundary (Section 2.4). |
| `salience` | float, optional, **capped** | A *hint* to ranking. Core caps and re-ranks; a producer cannot evict rivals (Section 7). |
| `source_agent` | string | Which agent/service produced it (self-declared; used for audit + rate-limit bucketing). Not a trust boundary — the confirm chokepoint is. |

### 2.2 The dedicated `signals` table (NOT UserMemory)

**Rationale (privacy — do not compromise):** persisting presence/location into `UserMemory` would leak it into the recall tool and the embedding sweep — a real privacy hole. Signals get a dedicated table with no `Vector` column, never touched by memory recall or extraction.

Model added to `app/models.py`, copying the `UserMemory` shape (`models.py:369-389`):

- `id Integer PK autoincrement` (`models.py:377`)
- `household_id String(255) NOT NULL` indexed; `user_id Integer NULL` indexed; `node_id String(255) NULL`; `room String(255) NULL` (scope)
- `kind String(255) NOT NULL` indexed; `subject String(255) NULL`
- `source_key String(512) NOT NULL` — **the UPSERT key**
- `summary Text`; `facts Text` (JSON-serialized); `source_agent String(255)`
- `cacheable Boolean NOT NULL default False`; `salience Float NULL`
- `observed_at DateTime`; `expires_at DateTime NULL` indexed (`ix_signals_expires_at`, TTL sweep) — NULL = never expires (`models.py:389`)
- `is_active Boolean NOT NULL default True` (soft-delete, `models.py:384`)
- `created_at`/`updated_at` with `onupdate=datetime.utcnow` (`models.py:387-388`)

**DB-enforced uniqueness (unlike UserMemory):** add
`__table_args__ = (UniqueConstraint('household_id', 'source_key', name='uq_signals_household_source'),)`
(copying `PersonCharacterization.__table_args__`, `models.py:416-420`). This makes flaps physically collapse to one row and lets a DB-backed test prove the constraint rejects dupes.

### 2.3 UPSERT-on-`source_key`

There is no Postgres `ON CONFLICT` in this codebase; upsert is application-side (`memory_service.py:114-158`). New `SignalService.save_signal(...)`:
- Query existing by `household_id == AND source_key == AND is_active == True` (NULL-safe on `user_id` per the branch at `memory_service.py:121` — `.is_(None)` when None else `==`).
- Found → mutate fields, `updated_at = utcnow()`, `commit()`, `refresh()` (mirrors `memory_service.py:128-142`). A flap updates `summary/facts/observed_at/expires_at` in place; the row identity (and thus any suppression keyed on `source_key`) is stable.
- Else → construct, `add`/`commit`/`refresh` (`memory_service.py:144-157`).

### 2.4 Invariants + the cacheable write-boundary rule

1. **`kind` is never control-flow.** No `if signal.kind == "presence.seen"` anywhere in core. Behavior comes from the LLM reading `summary`/`facts` against the capability menu.
2. **Cacheable ⇒ quantized + absolute + household-scoped.** Only `cacheable=True` Signals may be promoted into the cached prefix `messages[0]`. **Enforce by VALIDATION AT THE WRITE BOUNDARY:** `save_signal` rejects (422/400) any Signal marked `cacheable=True` whose `summary`/`facts` contain a live float, a relative time string, or a sub-15-min timestamp — a non-quantized cacheable signal would poison the prefix cache and destroy TTFS (the exact regression fixed in `#100` / `35c9b8d`). Everything else defaults `cacheable=False` and rides the trailing block. This is defense-in-depth: re-checked at the prefix builder (Section 4).
3. **TTL sweep mirrors memory.** A `_periodic_signal_cleanup` worker (Section 6/10) deletes expired rows.
4. **Server-trusted scope.** `household_id` comes from validated auth, never the client body when they conflict (Section 3).

---

## 3. Ingress — `POST /api/v0/signals`

New router `app/api/signals.py` (mirrors `app/api/memories.py`), registered in `main.py` next to `memories.router` (`main.py:722-723`): `from app.api import signals` / `app.include_router(signals.router, prefix="/api/v0")` → `POST /api/v0/signals`.

### 3.1 Auth — existing credentials only, no per-source registration

Inline `SignalsAuthContext(auth_type, household_id)` dataclass + `_verify_signals_auth(...)` dependency, cloned from `_verify_inject_auth` (`memories.py:205-258`). **Two existing credential types only — no new `signal_sources` table, no per-key allowlist.** A per-source allowlist in core would reintroduce the "third party needs a core update" trap; instead a third party integrates purely by POSTing with a household credential:

- **Node auth** (`X-API-Key: node_id:node_key`): call `verify_api_key` (`deps.py:162`); household = server-trusted `NodeContextProvider.household_id` (`node_context_provider.py:9`). Nodes, self-hoster glue (cron/n8n/Shortcuts given the household's node key), and external systems all use this.
- **App-to-app** (`X-Jarvis-App-Id`/`X-Jarvis-App-Key`, internal services): reuse the inline app-ping fallback (`memories.py:237-256`); household comes from the body (`app-ping` returns no household). `jarvis-recipes-server` posts this way.

**The command boundary is NOT stored anywhere — it is the default node's advertised capabilities.** A signal can only ever produce a proposal for a command the household's node has installed and advertised as proposable (`capability_registry.list_proposable_actions(node_id)` for open; `resolve_proposable_action` for directed, which returns `None` → refused for anything not advertised). So "allowed commands" is populated live by the node, not a core table — a newly installed command becomes proposable instantly, with zero CC change. **Which node?** For a node-auth signal it's the poster's node; for an external/app signal (no node in scope) the menu comes from the household's **default node** resolved via `_first_household_node_id` (`proposable_action_service.py`, already `is_active` + most-recent-aware from the shipped fix).

**Anti-spoof rule (load-bearing, copy `memories.py:280-293`):**
```
household_id = body.household_id or auth.household_id
if not household_id: 400
if auth.auth_type == "node" and body.household_id and auth.household_id and body.household_id != auth.household_id: 403
```
No credentials → 401; invalid credential → 401; auth service unreachable → 502.

> **Future (not built here):** if a revocable, mobile-mintable external token is ever wanted, add a **household-scoped auth token only** (no allowlist, no per-command config) — commands stay bounded by the node. That is an additive auth mode, not the `signal_sources` allowlist rejected here.

### 3.2 Two payload shapes — OPEN vs DIRECTED

Inline pydantic models (route-local convention, `memories.py:180-201`), bounded fields (`Field(..., max_length=...)`, `ttl_seconds` `gt=0, le=…`), batch cap enforced twice (pydantic `max_length` + explicit `len()>N → 400`, per `memories.py:276`):

- **OPEN** `{signal, data}` — the matcher picks the command. `signal` carries `kind/source_key/subject/summary/scope/ttl_seconds/cacheable/salience`; `data` becomes `facts`. Persisted as a Signal; the proactive matcher (`match_situation`) chooses which advertised command, if any, fits.
- **DIRECTED** `{signal, command, data}` — the producer names the command, exactly like `propose_action`. Persisted as a Signal AND emits a directed proposal for `command` (skips `list_proposable_actions`/matcher; goes straight to the A–F chokepoint, matching the shipped directed path).

Discriminate with a `Literal` field or a `model_validator(mode="after")` (open requires no `command`; directed requires it).

### 3.3 The propose-not-execute safety boundary

**An API key grants the right to PROPOSE, never to EXECUTE.** Every inbound — open or directed, node or key — flows through the **same generic dispatcher and the same confirm chokepoint** (`proposable_action_service._handle_execute`, the A–F gates at `:237-328`). Concretely, an inbound signal produces at most a tap-to-confirm inbox card; execution only happens when a household member taps it, at which point A–F re-validate opt-in/params/idempotency on the node plane. Additional per-inbound gates before a card is ever raised: **node-bounded commands** (a proposal can only name a command the household's node advertises — non-advertised → refused at `resolve_proposable_action`), **rate limit** (per `household`+`source_agent` — see 3.4), **blast-tier**, **occurrence idempotency**, **cooldown**, **daily cap**, **suppression** (Section 7). *A leaked household credential can at worst raise a capped, suppressible card for an already-installed command; it cannot fire `unlock_door` unless that command is installed AND a member taps to confirm.*

### 3.4 Rate limiting

**There is no rate-limiting anywhere in CC today** — this is net-new, but light:
- An **in-memory token bucket keyed on `(household_id, source_agent)`** (no persistence, no per-source row — mirrors jarvis-pantry `rate_limiter.py`). Over budget → **429 + Retry-After**. This caps a noisy or runaway producer without any registration; `source_agent` is self-declared, which is fine because the real safety boundary is the confirm chokepoint, not the rate limiter.
- Cheap short-circuit gates that also throttle expensive work (copy `memories.py:296-317`): a `signals.enabled` per-household setting → 409/early-return when off; the proactive reason pass additionally re-checks `proposals.enabled` at entry (Section 7).

Flag as new pattern: the in-memory token bucket is the only genuinely new mechanism; everything else reuses inject's soft protections.

---

## 4. Reactive path (ships FIRST)

**Why first:** high value, cache-safe, no LLM tick, no autonomy. Renders current Signals into the LLM's turn context so voice answers reflect "who's home / what just finished" with zero new inference.

### 4.1 Generalize `_assemble_ambient_bundle`

Today `_assemble_ambient_bundle` (`conversation_handler.py:2762-2826`) hardcodes three categories (`:2789`), does latest-per-category recency (`:2794-2815`), and has a hand-rolled weather `current`-vs-`forecast` `ilike` hack (`:2803-2813`).

**Change:** replace the `categories` tuple + per-category loop (`:2789-2819`) with a query over non-expired `signals` for the household, then **stable-sort by `(kind, subject)`** (`sorted(signals, key=lambda s: (s.kind, s.subject))`) and render **latest-per-`(kind, subject)`**. The weather hack becomes the general case — `subject` *is* the disambiguator, so `current` and `forecast` both render and neither shadows the other; delete the `ilike` special-case. **Preserve the label-prefix grammar** at `:2816-2819` per rendered line (content kept verbatim if it already leads with its label, else `f"{label}: {content}"`) so multi-fact-per-kind content survives. First line stays the 15-min quantized absolute clock (`:2780-2781`). Fail-open on DB error → `""` (unchanged).

### 4.2 Prefix-vs-trailing cache discipline (the load-bearing invariant)

Two homes for a Signal, and validation gates the split:

- **Trailing per-turn block** (default, `cacheable=False`): re-derived every turn, appended as a `role=system` message after `messages[0]` at the three voice sites (`:718-720`, `:1005`, `:1350`), stripped+rebuilt via `_is_transient_system_block` (`:129`, strip at `:699`) so it never accumulates. New renderer `render_signal_block(signals)` in `core_rules.py` (sibling to `build_ambient_context_block`, `core_rules.py:338-370`), fenced with a stable tag (reuse `<ambient_context>` or add `<signals>`). No cacheability gate needed here.
- **Cached prefix `messages[0]`** (`cacheable=True` only): quantized, household-scoped Signals may be concatenated into the system prompt built at `conversation_handler.py:483-490` / `build_context_header` (`ijarvis_prompt_provider.py:36-85`). **Re-check the cacheable flag here** and drop/raise on any non-quantized Signal before it reaches the frozen string (`:490`). This is the second half of the write-boundary rule (Section 2.4) — defense in depth so the prefix stays byte-stable per household and the llama.cpp prefix cache is never cold-prefilled.

**Invariant test (must hold):** `build_context_header(with_trailing_signals) == build_context_header(base)` and the fence tag is absent from the header — mirroring `TestAmbientStaysOutOfCachedPrefix` (`test_ambient_context.py:114-120`).

---

## 5. `match_situation` — bundle generalization of `match_proposals`

Generalize the shipped OPEN-mode matcher (`proposal_matcher.py:74-129`) from one `data: dict` to a **bundle of Signals**, keeping every existing guard. `match_proposals` becomes a one-line adapter (`bundle=[{"source_key":..., "data": data}]`) so the email pilot is untouched.

New signature:
```
async def match_situation(*, bundle: list[dict], node_id, llm_client=None, fetch=None) -> list[dict]
# bundle item: {"source_key": "<hard key>", "data": {...facts...}}
```

Mechanical changes, skeleton intact:
1. **Menu unchanged** — `list_proposable_actions(node_id)` + `_build_menu` (`:87-91`); `idempotency_param` still excluded from the menu (`:44`).
2. **Prompt** — `_build_match_prompt` takes the whole bundle; render each item as `#<i> (source_key=…): {data}`; ask for `{"matches":[{command,action,args,"source_keys":[<i>…]}]}`. Only new field: `source_keys`.
3. **`by_key` anti-hallucination drop unchanged** (`:107-111`).
4. **Idempotency key scoped to CONTRIBUTING items** — feed only the cited items' data into `_stable_idempotency_key` (`:31-35`), i.e. `_stable_idempotency_key({"items":[bundle[i] for i in contributing]}, command, action)`, so two situations that trigger the same action from different sources don't collapse.
5. **Validation unchanged** — `validate_against_params` (`:117`).
6. **RETURN contributing `source_keys`** — map the LLM's cited indices back to `source_key` values, add `"source_keys":[...]` to each result. **Drop any match citing an out-of-range index** (same never-trust-the-LLM philosophy as the `by_key`/validate drops).

**Directed** signals bypass the matcher entirely (agent-supplied `command`/`args`) and go straight to `_handle_execute` — identical to the shipped `propose_action` path. So `match_situation` is purely the OPEN-path replacement; `proposable_action_service.py` + `capability_registry` are shared verbatim by both, unchanged.

**Downstream reuse is free:** `source_keys` rides onto the card's `_action`, so `_handle_suppress` (`:346-348`) already knows the exact hard key to blocklist and `_already_completed` DB dedup (`:132-156`) works on the new key.

---

## 6. Proactive path (event-driven, off the live slot, only proposes)

### 6.1 Event-driven + debounced worker

**Not polled.** A Signal write that changes a `(kind, subject)` key flips a module-level `asyncio.Event`; co-arriving edges coalesce into ONE reason pass → ONE enqueue → ONE card. Structural template from `_periodic_characterization_synthesis` (`main.py:298-316`): defined in `startup_event`, launched via `asyncio.create_task`, settings-gated, `try/except` that only logs (invariant #10).

Replace the blind `asyncio.sleep(interval)` with: `await _situation_wake.wait()` (zero cost when idle), then an inner debounce loop — `asyncio.wait_for(_situation_wake.wait(), timeout=_DEBOUNCE_SECONDS)` collapsing a burst into one batch, with a `_MAX_LATENCY_SECONDS` ceiling so a steady drip still fires. `SignalService.save_signal` calls `signal_situation_edge()` (cheap, non-blocking) only when a `(kind,subject)` value actually changed (true edge). The DB queue (unprocessed signals + `reset_stale_jobs`, mirroring `memory_extraction_service.py:77-79`) is the source of truth; the event is an optimization so a restart loses nothing.

### 6.2 Background-model enqueue

`situation_matcher_service.run_match_batch()` builds the bundle and enqueues via `POST {llm_proxy}/internal/queue/enqueue`, cloning `_enqueue_extraction` (`memory_extraction_service.py:108-212`):
- `request.model == "background"` — **the single off-slot selector** (`:182`; asserted like `test_characterization.py:192`).
- `job_type:"chat"`, `job_type_version:"v1"`, `ttl_seconds:600`, `job_id == trace_id == idempotency_key`.
- `metadata:{type:"situation_match", household_id, node_id, contributing_source_keys, occurrence_ids}` — round-tripped into the callback.
- `sampling:{temperature:0.0}` and **`/no_think`** suppression (short prompt), for slot-safety (6.4).
- `callback.url` = `…/api/v0/situation-matcher/callback`, `auth_type:"bearer"`, `token = JARVIS_ADAPTER_CALLBACK_TOKEN` (`:155-164`).
- Enqueue POST at the `app.core.utils.rest_client.post` boundary with `X-Internal-Token: LLM_PROXY_INTERNAL_TOKEN` (`:193-206`). **Ordering invariant:** mark edges in-flight only after a successful enqueue (`:204-206`).
- **`proposals.enabled` checked at REASON ENTRY** (fail-closed, `_proposals_enabled`, `proposable_action_service.py:50`) so opted-out households are never reasoned over — the batch short-circuits before building any prompt.

### 6.3 Callback → card

New `POST /api/v0/situation-matcher/callback`, cloned from `/memory-extraction/callback` (`main.py:1834-1851`): `_verify_callback_auth(request)` FIRST (`main.py:1727-1755`, fail-closed: 401 on bad Bearer, 503 if token unset unless `JARVIS_ALLOW_INSECURE_CALLBACKS`), then `handle_match_callback(payload)` in a try/except, always return 200. Handler mirrors `handle_extraction_callback` (`memory_extraction_service.py:215-305`): on `status!="succeeded"` log+return (leave un-marked for stale-retry); on success parse `result.content` with the tolerant `<think>`/fence stripper (`:308-354`), then feed the parsed matches through **`match_situation`'s validated output into the A–F chokepoint** to raise cards. **The reasoner ONLY proposes** — it hands validated `{command,action,args,source_keys}` to `_handle_execute`'s upstream card-creation; it physically cannot call a tool (the chokepoint owns execution).

### 6.4 Slot-safety (concrete risk + mitigation)

Even the `"background"` lane can share the single llama.cpp process / 8192 serial queue with live voice. Two real hazards: **prefix-cache eviction** (a long background prompt evicts the warmed voice prefix → next voice turn re-encodes ~9k tokens, blowing the 1.4s CC budget — the exact concern behind `35c9b8d`) and **head-of-line blocking** (voice waits behind an in-flight generation — the reason `_enqueue_extraction` forces `/no_think`, `memory_extraction_service.py:142-152`). Mitigations that hold: (a) verify llm-proxy's `"background"` priority truly deprioritizes/preempts, not just labels; (b) the debounce minimizes job count; (c) keep the prompt short + `temperature:0.0` + `/no_think` so occupancy is brief; (d) `ttl_seconds:600` drops stale jobs rather than piling them behind voice.

---

## 7. Guardrails (all reused)

- **Opt-in, fail-closed.** `proposals.enabled` (default False, `settings_definitions.py:312-324`) gates card dispatch at `_handle_execute` (`proposable_action_service.py:252`) AND is re-checked **at reason entry** (6.2) so opted-out households are never reasoned over. `signals.enabled` gates ingress. Both use the fail-closed `_proposals_enabled` shape (`:50-66`, any exception → False).
- **Confirm chokepoint.** The A–F dispatcher (`proposable_action_service._handle_execute`, `:237-328`) is the single execution path for open, directed, node, and API-key inbounds. The reasoner cannot bypass it.
- **Blast-tier.** SDK `BlastTier` (`proposable.py:34`, presentation-only) → irreversible actions get a stronger ack; multi-step actions route through `errand_planner`/`workflow_engine.widens_envelope` (`workflow_engine.py:468`) which pauses-and-replans when the envelope widens (more `is_risky` steps than approved, `:499`).
- **Anti-nag** (four layers):
  1. **Edge-triggering** — fire on false→true only; never while a condition merely persists. `signal_situation_edge()` fires only when a `(kind,subject)` value changes (6.1).
  2. **Occurrence-scoped idempotency** — the idempotency key is `hash(contributing source_keys + their coarse observed_at bucket + command + action)`, EXCLUDING ticking `time.now`, built via `match_situation`'s contributing-items scoping (Section 5, step 4). The `observed_at` bucket distinguishes "this arrival home" from "next Tuesday's" without a recurrence engine; a single persisting edge stays one card. (Bucket size is the one remaining tuning knob — Section 13.)
  3. **Per-`(household, source_key, command)` cooldown.**
  4. **Per-household daily cap.**
  Cooldown + daily cap are **backstops, not the primary control** (Section 8).
- **Suppression reuse (zero new UI).** "Don't suggest this again" → `record_suppression(household, user, command, source_key, descriptor)` (`proposal_suppressions.py:18`), upsert-keyed on `(household,user,command,source_key)` (`:37-51`). The Signal's `source_key` is the hard key. Injection contract `suppression_signals(...)` (`:103`) returns distinct `source_keys` (deterministic hard-skip) + `descriptors` (negative prompt examples). The existing mobile Suppressed-Suggestions CRUD (`proposals.py` mobile_router `:54-84`) lists/deletes — no new screen.
- **Core-owned ranking.** Producer `salience` is a **capped hint**; core ranks the bundle on recency + importance so an adversarial producer can't evict rivals. Ranking lives in `run_match_batch`/the renderer, not the producer.
- **Egress fail-closed.** `web_search.enabled` gates unchanged. A signal writer can **assert**, never **open egress**. Nothing in this subsystem un-gates outbound web.

### PROPOSE-only now; autorun is a designed-for future (Section 7 continued)

**Everything proposes a card — for every source, including internal services.** No autorun ships here. But the design leaves the seam open: autorun would be a *later* additive gate at the dispatcher letting a trusted producer skip the tap for **reversible + low-blast** actions only (keyed on `BlastTier`; irreversible/high-blast always card). Because we rejected `signal_sources`, any future trust signal lives as a **household / per-command setting** (e.g. `signals.autorun_commands`), never a per-key table. Candidate first case: `meal_plan.completed → create_shopping_list`. Build propose-only, but keep the dispatcher's single-path shape so autorun is a switch, not a rewrite.

---

## 8. Producers

Genericity is proven by **≥2 producers before any phone work**, so presence is never a special-case:

1. **Voice-derived presence (FIRST, free).** `speaker_user_id` resolved at a node during a voice turn IS a `presence.seen` Signal (`scope.node_id`, `scope.room`, `user_id`). Written from the existing hot-path speaker resolution — no new sensor. Zero-cost, real, and immediately validates the reactive renderer.
2. **Synthetic (tests / self-hoster).** `curl POST /api/v0/signals` with the household node key (or app creds). Validates the whole spine with no phone and no live model — Phase 1 ships on this alone.
3. **Jarvis's own microservices.** `jarvis-recipes-server` posts `meal_plan.completed` (app-to-app) → proposes `create_shopping_list` (a future-autorun candidate). Proves internal-service ingress.
4. **Self-hoster glue.** cron / n8n / Shortcuts POST with the household node key. The propose-not-execute boundary + node-bounded commands make this safe by construction — no core registration.
5. **Phone geofence (LATER — Phase 5).** The high-quality presence producer (node agent + phone geofence). Deferred until the spine is proven by producers 1–4.

**HA-independent** — Home Assistant is at most ONE optional producer; nothing in the design depends on it. **Acoustic/fall detection is PARKED** — audio is a weak fall sensor; accelerometer/mmWave is the real tool; no acoustic producer here.

---

## 9. Settings surface — the `signals.*` group

Append to `SETTINGS_DEFINITIONS` (`settings_definitions.py`, copying `proposals.enabled` at `:312-324`), `category="signals"`, `value_type="bool"`, `default=False`:
- `signals.enabled` — master ingress + reactive toggle (fail-closed).
- `signals.proactive_enabled` — gates the proactive reasoner independently of reactive (so reactive can ship while proactive is dark).
- `signals.debounce_seconds`, `signals.max_latency_seconds` — worker coalescing window + ceiling.
- `signals.daily_cap`, `signals.cooldown_seconds` — anti-nag backstops.

Household toggles: add `signals.enabled` (and `signals.proactive_enabled`) to `HOUSEHOLD_CONTROLLABLE_SETTINGS` (`mobile_household_settings.py:37`) — member read (`_READ_ROLE`), admin write (`_WRITE_ROLE`, `verify_household_role`). Any non-allowlisted `signals.*` key 404s (`:139-144`). Proactive also inherits `proposals.enabled` at reason entry.

---

## 10. Data model + migration

**New table (`app/models.py` + one alembic migration, template `alembic/versions/a7w8x9y0z1a2_add_request_traces.py`):**

**`signals`** — columns/indexes/constraint per Section 2.2. Migration: `op.create_table("signals", …sa types…, server_default=sa.func.now() on timestamps)`; `op.create_index("ix_signals_expires_at", …)`, plus indexes on `household_id`, `kind`, `user_id`; declare `uq_signals_household_source`. `downgrade` drops the table. **Find head first** (`alembic heads`), chain `down_revision`.

**No `signal_sources` table.** Per-source registration + allowlists were rejected — they'd force a core update per third party. Auth reuses existing node/app credentials (Section 3.1); the command boundary is the node's *live* advertisement; rate-limiting is an in-memory `(household, source_agent)` bucket. Nothing per-source is persisted, so a third party integrates with zero CC change.

No new UI for suppression (reuses `proposal_suppressions` + the mobile CRUD).

---

## 11. THE BUILD PLAN (5 phases, each TDD RED→GREEN)

### Phase 1 — Signal contract + `signals` table + migration + `POST /api/v0/signals` [node + API-key, open + directed]
**Independently shippable with synthetic curl signals — no phone, no live model.**

**(a) RED tests:**
- `tests/test_signal_service.py` (new, SQLite `Base.metadata.create_all` fixture mirroring `test_memory_service.py:13-24`; no embedding column so SQLite is fine): `save_signal` basic insert; **upsert-on-`source_key`** (same key updates + bumps `updated_at`, no dup row); NULL `user_id` household-wide branch; `cleanup_expired` deactivates/deletes only `expires_at <= now`, leaves NULL-`expires_at`; **cacheable write-boundary rejection** (a `cacheable=True` signal carrying a live float/relative-time string → raises).
- `tests/test_signals_ingest.py` (new, `TestClient(app)` + `_verify_signals_auth`/`get_db` overridden, house style from `test_memories_authz.py:37-47`): node no-body-household → derived+200; node body-household matches → 200; **node body-household mismatch → 403**; app-to-app with body household → 200; no creds → 401; invalid credential → 401; auth service unreachable → 502; **directed naming a command the node does NOT advertise → refused** (`resolve_proposable_action → None`, no card, no execution); **directed** (`command` set, advertised) persists + emits directed proposal (assert `_handle_execute`-bound card path invoked, patched); **open** persists, matcher deferred; batch `>N` → 400 + zero rows; over-length field → 422; `signals.enabled` false → 409 + nothing persisted; **rate-limit over budget (per household+source_agent) → 429**.
- `tests/test_signals_migration.py` (DB-backed, `test_db` fixture, run via `run_database_tests.py --type postgres`): row round-trips; `uq_signals_household_source` rejects a second insert with the same `(household, source_key)`.

**(b) GREEN:** add the `Signal` model to `app/models.py`; `app/services/signal_service.py` (`save_signal` upsert copying `memory_service.py:114-158`, `cleanup_expired` copying `:582+`, cacheable validation); `app/api/signals.py` (inline pydantic open/directed models, `SignalsAuthContext`, `_verify_signals_auth` node→app fallback, anti-spoof block `memories.py:280-293`, in-memory `(household,source_agent)` token bucket, endpoint); register in `main.py:722`; **one** alembic migration; `_periodic_signal_cleanup` in `startup_event` copying `main.py:359-376`.

### Phase 2 — Generalized reactive renderer [REACTIVE-ONLY, voice-derived presence as first real producer]
**(a) RED tests** (`tests/test_signal_render.py` new, + extend `tests/test_ambient_context.py`, SQLite `StaticPool` sessionmaker per `test_ambient_context.py:25-38`):
- `render_signal_block([])` → `""`; fenced block with stable tag; **stable ordering** (two differently-ordered inputs → byte-identical output, proving `(kind,subject)` sort); **multi-fact grammar preserved** (label-leading content not double-prefixed; `current`+`forecast` both render, neither shadows); determinism/quantization (same 15-min bucket → byte-equal; no live seconds leak).
- **Cache-split invariant:** `build_context_header(with_trailing_signals) == build_context_header(base)`, fence absent from header (mirror `test_ambient_context.py:114-120`); a non-quantized signal promoted to prefix is rejected before `messages[0]`.
- **Transient-strip:** `_is_transient_system_block` True for the signal block; append twice via strip path → exactly one survives.
- Fail-open: DB error → `""`, warmup unbroken.
- Voice-derived presence: a resolved `speaker_user_id` at a node writes a `presence.seen` Signal (assert `save_signal` called with the right scope from the hot path).

**(b) GREEN:** add `render_signal_block` to `core_rules.py` (sibling of `:338-370`); refactor `_assemble_ambient_bundle` `:2789-2819` to the Signal iteration; wire the renderer into the three trailing sites (`:718`, `:1005`, `:1350`); optional prefix promotion behind the cacheable gate at `_get_system_prompt`/`build_context_header`; emit the voice-derived presence Signal from the speaker-resolution point in the hot path.

### Phase 3 — `match_situation` (bundle generalization returning contributing `source_keys`)
**(a) RED tests** (`tests/test_match_situation.py` new, reusing `_fetch_report()`/`_llm()` verbatim from `test_proposal_matcher.py`, `asyncio.run`, no DB/network):
- `test_returns_contributing_source_keys` (2-item bundle, LLM cites `#0` → result `source_keys==["cal@x"]`, other absent).
- `test_idempotency_key_scoped_to_contributing_items` (same contributing item + different noise → same key).
- `test_out_of_range_source_index_dropped` (`#5` in a 2-item bundle → dropped).
- Port the four shipped guards to bundle input: hallucinated action drop, invalid-args drop, empty matches, no advertised actions.
- `test_prompt_lists_source_keys_and_omits_injected_param`.
- `test_match_proposals_adapter` — `match_proposals` still passes the entire existing `test_proposal_matcher.py` suite (proves the one-line adapter untouched the email pilot).

**(b) GREEN:** implement `match_situation` in `proposal_matcher.py` (bundle-aware `_build_match_prompt`, index→`source_key` map with range-drop, contributing-items idempotency scoping); keep `_build_menu`/`by_key`/`validate_against_params`/`_run_planner` unchanged; make `match_proposals` a one-line adapter.

### Phase 4 — event-driven `_situation_matcher_loop` + background enqueue + `/situation-matcher/callback` [behind `proposals.enabled` + precision harness]
**(a) RED tests:**
- `tests/test_situation_matcher_enqueue.py` (new, patch `app.core.utils.rest_client.post` AsyncMock): assert `request.model=="background"` (the key assertion, mirror `test_characterization.py:192`), `job_type=="chat"`, `job_type_version=="v1"`, `ttl_seconds==600`, `job_id==trace_id==idempotency_key`, callback URL suffix, bearer token present iff `JARVIS_ADAPTER_CALLBACK_TOKEN` set, `X-Internal-Token` header; **ordering** (post raises → edges NOT marked; succeeds → marked); **reason-entry gate** (`proposals.enabled` false → no enqueue); **debounce** (event set N times in-window → `run_match_batch` awaited once; `_MAX_LATENCY_SECONDS` ceiling forces a fire under sustained drip).
- `tests/test_situation_matcher_callback.py` (new, monkeypatch `app.db.get_session_local`, `asyncio.run(handle_match_callback(payload))`, mirror `test_characterization.py:56-62`): `succeeded` + valid content → cards raised via chokepoint (assert `_handle_execute`-upstream invoked); `failed` → nothing, un-marked for retry; unparseable/empty content → nothing, no throw; metadata round-trip.
- Extend `tests/test_async_job_callback_auth.py`: token set + wrong/missing Bearer → 401; unset + insecure off → 503; unset + insecure on → 200, hitting the new route.

**(b) GREEN:** `app/services/situation_matcher_service.py` (`run_match_batch` enqueue cloning `memory_extraction_service.py:108-212`, `handle_match_callback` cloning `:215-305`, `signal_situation_edge`/`_situation_wake`); `_situation_matcher_loop` + `asyncio.create_task` in `startup_event`; `POST /api/v0/situation-matcher/callback` on `v0_router` (clone `main.py:1834-1851`); anti-nag backstops (cooldown/daily-cap) in `run_match_batch`. **Gate user-visibility behind the offline precision harness passing** (Section 12).

### Phase 5 — node presence agent (phone geofence) as the high-quality producer
**(a) RED tests:** node-side agent unit tests (in `jarvis-node-setup`) that a geofence enter/exit posts a `presence.seen`/`presence.left` Signal to `/api/v0/signals` with the right `source_key` (stable per user+home) and `ttl_seconds`; CC-side: geofence signal renders in the reactive block and, on a false→true edge, enqueues exactly one reason pass (reuse Phase 4 debounce test harness).

**(b) GREEN:** node presence agent (node auth) + phone geofence integration; no CC schema change (rides the Phase 1 ingress).

---

## 12. Test & coverage strategy

- **Unit (fast) vs DB-backed.** No `tests/database/` split — flat `tests/*.py`; a test is DB-backed iff it requests `test_db`. Model/service RED→GREEN runs on in-memory SQLite (`StaticPool` sessionmaker, `test_ambient_context.py:25-38` / `test_memory_service.py:13-24`) — clean because `signals` has no `Vector` column. Migration + `UniqueConstraint` + `server_default` semantics proven with `test_db` (real pgvector Postgres) via `run_database_tests.py` (docker port 5544 default, `--type postgres` → 5433).
- **Mock boundaries.** LLM-proxy / enqueue mocked at `app.core.utils.rest_client.post` (CLAUDE.md testing note; `main.py:31`). Matcher boundaries injected: `fetch=AsyncMock`, `llm_client.chat_completion=AsyncMock` (`test_proposal_matcher.py`). Auth faked via `dependency_overrides` + `patch("app.api.signals.verify_household_role"/…)` (`test_memories_authz.py`, `test_mobile_household_settings.py:23`). Gated-dispatcher tests stack `patch.object(svc, "_proposals_enabled"/"_already_completed"/"_execute_target_callback_on_node", …)` (`test_proposable_action_service.py`).
- **Offline precision harness (the primary control for the proactive reasoner).** Before Phase 4's cadence trigger is user-visible, run the reasoner on **replayed real bundles** on the **9B worst case (background == live)** and measure **NONE-rate** (fraction of bundles that correctly propose nothing) and **false-positive-rate** (unwanted cards). Daily cap + cooldown are backstops, not the primary control. The harness is a standalone script over captured `signals` bundles, asserting `match_situation` output against a labeled gold set. **Metrics + gating threshold are an OPEN decision** (Section 13) — do not ship proactive to users until the threshold is set and met.
- **Coverage target** 80%+ (RULES.md); RED→GREEN→IMPROVE per phase.

---

## 13. Risks + Open decisions

**Risks:**
- **Slot contention** (Section 6.4) — background job evicts the voice prefix or blocks the serial queue. Mitigated by debounce + short `/no_think` prompt + `ttl_seconds:600`; **must verify llm-proxy's `"background"` lane truly preempts**, not just labels.
- **Reasoner false positives / nagging** — mitigated by edge-triggering + occurrence idempotency + cooldown + daily cap, but the real gate is the precision harness.
- **Leaked household credential** — bounded by propose-not-execute + node-bounded commands + rate limit + suppression; worst case is a capped, suppressible card for an already-installed command.
- **Cache poisoning** — a mis-marked `cacheable` Signal blows TTFS; mitigated by write-boundary validation + prefix re-check (defense in depth).

**Resolved (previously open):**
1. **Occurrence-identity for the idempotency key** → `hash(contributing source_keys + their coarse observed_at bucket + command + action)`, excluding live `time.now` (Sections 6–7).
2. **Per-source allowlist / `signal_sources`** → **dropped.** No per-source registration in core; auth reuses node/app credentials and the command boundary is the node's live advertisement (Section 3.1).
3. **Propose-vs-autorun** → **propose-only now**; autorun designed-for-later as a household/per-command setting, never a per-key table (Section 7).

**Still open:**
1. **Precision-harness metrics + gating threshold** — the exact NONE-rate / false-positive thresholds (on the 9B worst case) that must be met before the proactive cadence goes user-visible (Section 12).
2. **Coarse `observed_at` bucket size** for the idempotency key (e.g. 1h vs per-day) — tune so a genuine re-occurrence re-proposes but a flapping edge does not.

---

## 14. Post-development manual acceptance test plan (for the user)

A by-hand runbook so the live system can be validated end-to-end on the dev node after development. The automated tests (Section 12) prove correctness in isolation; **this proves the running system behaves.** Run a phase's block once that phase is GREEN.

**Prereqs.** CC up (`curl :7703/health`); a paired dev node with its household node key; `signals.enabled=true`; for proactive: the background model configured in llm-proxy + `proposals.enabled=true`. DB rows checked via the jarvis-mcp `query_logs`/psql; cards checked in the mobile inbox.

### Phase 1 — the Signal spine (no phone, no live model)
1. **Ingest (open).** `curl -X POST :7703/api/v0/signals -H "X-API-Key: <node_id>:<node_key>" -d '{"signal":{"kind":"presence.seen","source_key":"presence:alex","subject":"user:alex","summary":"Alex heard in the kitchen","scope":{"user_id":<alex>},"ttl_seconds":900},"data":{"user":"alex","room":"kitchen"}}'` → **200**; the `signals` table has exactly one row, `expires_at ≈ now+900s`.
2. **Upsert.** Re-POST the same `source_key` with a new summary → still **one** row, `summary` updated, `updated_at` bumped.
3. **TTL.** POST `ttl_seconds:1`; after the cleanup tick the row is gone / filtered as expired.
4. **Cacheable rejection.** POST `cacheable:true` with a live float or relative-time string in `summary` → **422/400**, no row.
5. **Directed, advertised.** POST `{...,"command":"<an installed proposable command>",...}` → **200** and a tap-to-confirm **card in the inbox**.
6. **Directed, NOT advertised.** POST `command:"unlock_door"` with no such command installed → **refused** (`resolve→None`), no card, no execution.
7. **Auth.** no creds → 401; wrong node key → 401; node key for household A with `body.household_id=B` → **403**.
8. **Gating.** `signals.enabled=false` → POST **409**, nothing persisted.
9. **Rate limit.** Hammer past budget with one `source_agent` → **429 + Retry-After**.

### Phase 2 — reactive "who's home" (voice-derived presence)
1. POST a `presence.seen` for Alex, then ask a node **"who's home?"** → the spoken answer reflects Alex **with no tool call** (verify the request trace shows no tool iteration).
2. **Voice-derived producer.** Speak any command to a node as an enrolled speaker → a `presence.seen` row for that user appears automatically (the hot path emitted it — presence with zero phone work).
3. **Prefix cache intact.** Compare TTFS / prompt-processing on the latency trace before vs after signals exist → **unchanged** (signals ride the trailing block, never the cached prefix).
4. **Expiry.** Let a signal's TTL pass → "who's home?" stops reporting that person.
5. **Opt-out.** `signals.enabled=false` → the ambient block omits signals entirely.

### Phase 3/4 — proactive proposals (the reasoner)
1. **Slot isolation (load-bearing).** With a voice command in flight, trigger a reasoner pass (POST an edge signal) → `docker logs llm-proxy-model` shows the reasoner ran as a **background** job and the voice command's TTFS was **not** delayed (latency trace).
2. **Edge → card.** POST a false→true edge that plausibly matches an installed command → within the debounce window, **one** card appears; tap it → the command executes (verify the effect).
3. **Idempotency.** Re-POST the same edge (same `source_key`, same occurrence bucket) → **no second card**.
4. **Re-occurrence.** POST the same `kind`/`source_key` at a genuinely later occurrence (past the bucket) → a **new** card.
5. **Persisting condition.** Keep a condition asserted across ticks → still **one** card (edge-triggered, not re-nagged).
6. **Cooldown / daily cap.** Exceed the per-household daily cap → further proposals suppressed for the day (logs).
7. **Suppression.** Tap **"Don't suggest this again"** on a card → it shows in the mobile **Suppressed Suggestions** screen; re-POST the edge → **no card**; delete it from that screen → suggestions resume.
8. **Reasoner opt-out.** `proposals.enabled=false` → no reasoner passes at all (no enqueue in logs).
9. **NONE behaviour.** POST a bundle where nothing installed fits → **no card** (returns NONE, never invents).

### Phase 5 — phone presence (real-world)
1. Cross the home geofence → a presence transition posts; "who's home?" updates; on a genuine arrival edge an arrival-shaped proposal appears (if a matching command is installed).
2. Reliability: confirm the geofence still fires after the app has been backgrounded for hours (the real iOS/Android check).

### External-producer smoke test (the extensibility proof)
1. From **cron/curl** (self-hoster) with the node key, POST `meal_plan.completed` → it lands and, if a matching command is installed, proposes a card. **No CC change was needed to add this producer.**
2. From **jarvis-recipes-server** (app-to-app), POST `meal_plan.completed` → proposes `create_shopping_list` as a card.

### Gate before enabling proactive for real users
Run the offline precision harness (Section 12) on replayed real bundles against the **9B worst case**; confirm NONE-rate / false-positive-rate meet the agreed threshold **before** turning `proposals.enabled` on for daily use.
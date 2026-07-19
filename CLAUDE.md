# jarvis-command-center

The brain. Takes voice (or text) input from nodes / mobile, runs it through the LLM with a curated toolset, executes tools, streams back audio or returns JSON. Also owns: memory, transcripts, provisioning, smart home, routines, node updates, and the inbox-actions flow.

> **Largest service in the stack** by surface area — 24 route files, ~50 core/service modules, 7 background workers. If a task touches "voice", "memory", "tools", "routines", or "the node-mobile chat surface", it almost certainly lands here.

---

## Vocabulary (read this first)

| Term | What it is |
|---|---|
| **Tool** / **Command** | Interchangeable. An LLM-callable function. May execute server-side (in command-center) or client-side (on the node). The node sends its available client-tool list at `conversation/start`. |
| **Agent** | Node-side background task that supplies data to LLM context — e.g., a Home Assistant agent that periodically fetches device state. NOT an LLM agent. The node ships this data in `node_context.agents` on `conversation/start`. |
| **Routine** | An automation, like Home Assistant automations. A bundle of commands triggered by some condition. Built in mobile, persisted here. |
| **Prompt provider** | A model-family-specific system prompt + chat-format strategy. Examples: `qwen3-8b`, `llama-3b`, `hermes-2`. Selected via the `llm.interface` setting; auto-enriched in `list_all` to surface available providers. |
| **Adapter** | A LoRA adapter intended for per-household fine-tuning. **Dormant — third-party (llama.cpp #21125) lib issue blocks rollout.** Code exists but don't develop against it. Treat as code-frozen until the upstream story exists. |
| **Conversation** | One voice session keyed by `conversation_id`. Cached state: warmed prompt, tools list, node context. Lifecycle: `conversation/start` → 0..N `voice/command(/stream)` → maybe `voice/command/continue(/stream)` for tool results. |
| **Node context** | Authoritative session info the LLM sees: `node_id`, `room`, `household_id`, `speaker_user_id`, `speaker_name`, `agents`, optionally `room_hierarchy`. Built server-side from node properties — client-provided context is mostly ignored for security. |

---

## Topology

```
                    ┌─────────────────────────────────────────┐
                    │              jarvis-command-center      │
                    │                                          │
   Pi Zero node  ──▶│  /voice/command/stream     (audio out) │
   (mic, speaker)   │  /voice/command/continue                │
                    │  /conversation/start                    │
                    │                                          │
   jarvis-node-     │  /mobile/chat              (JSON out)   │
   mobile (JWT)  ──▶│  /mobile/audio                          │
                    │  /mobile/routines                       │
                    │  /mobile/traces                         │
                    │                                          │
                    │  /admin/*                  (admin token)│
                    │  /settings/*               (super JWT)  │
                    │                                          │
                    └────────┬──────────────┬──────────────┬──┘
                             │              │              │
                             ▼              ▼              ▼
                       llm-proxy-api    whisper-api      tts
                       (sync inference  (STT proxy)  (TTS streamed
                        + async queue                 audio back to
                        for adapter                   the caller)
                        train, deep
                        research,
                        memory ext)
                             │
                             ▼
                       jarvis-auth (node + JWT + app validation, speaker resolution)

   MQTT (server → node async push) ──▶  Pi Zero subscribers
   (node settings, package install, bluetooth, node updates)
```

---

## Quick Reference

```bash
# Local dev (Docker includes Postgres on jarvis-net)
bash run-docker-dev.sh

# Tests (DB-backed → spins Postgres in Docker)
python run_database_tests.py --type docker

# Health
curl http://localhost:7703/health
```

---

## Dependency graph

**Upstream (CC depends on):**
- **PostgreSQL** (required) — own DB with ~20 tables
- **jarvis-llm-proxy-api** (required, port 7704) — `/api/v1/chat` for live inference + `/internal/queue/enqueue` for async (adapter train, deep research, memory extraction)
- **jarvis-auth** (required, port 7701) — node validation, app auth (forwarding for media proxy), `/internal/users/batch` for speaker name resolution
- **jarvis-tts** (used by streaming voice paths, port 7707)
- **jarvis-whisper-api** (proxied through media routes, port 7706)
- **jarvis-config-service** (required for service discovery, port 7700)
- **jarvis-notifications** (port 7712) — for the inbox-actions flow
- **jarvis-mcp** (optional, port 7709) — MCP tools via `jarvis_mcp_client`; degrades gracefully if down
- **Mosquitto MQTT broker** (required for server→node async push)
- **MinIO/S3** (used for camera streaming via go2rtc — also proxied here)

**Downstream (depends on CC):**
- **jarvis-node-setup** (Pi Zeros) — the primary voice client
- **jarvis-node-mobile** — the chat / control client
- **jarvis-llm-proxy-api** — sends callbacks to CC's `/adapters/jobs/callback`, `/deep-research/callback`, `/memory-extraction/callback` when queued jobs finish

**Impact if down:**
- All voice + mobile chat fail. This is the user-facing brain.

---

## The hot path: voice command from a node

```
[Pi node]                  [command-center]                 [llm-proxy / tts / auth]
   │
   │ POST /conversation/start
   │   X-API-Key: node_id:node_key
   │   { conversation_id, client_tools[], available_commands[],
   │     node_context: { timezone, speaker_user_id, agents } }
   │
   │       ├── verify_api_key (deps.py)
   │       │     ├── node_context_provider validates via auth /internal/validate-node
   │       │     │      (with service_id="jarvis-command-center")
   │       │     └── caches result; returns NodeContextProvider with household_id
   │       │
   │       ├── builds node_context (server-trusted) from node properties + speaker resolution
   │       │     └── resolves speaker_user_id → display name via auth /internal/users/batch
   │       │
   │       ├── model_service.warmup_conversation_with_tools(...)
   │       │     ├── system_prompt_builder injects: speaker, memories, room, hierarchy, agents data
   │       │     ├── conversation_cache stores tools[], node_context, warm prompt
   │       │     └── optional pre-warm inference to populate llm.cpp prefix cache
   │       │
   │       └── returns 200 { conversation_id }
   │
   │ POST /voice/command/stream   (the steady-state hot path)
   │   { voice_command, conversation_id, speaker_user_id? }
   │
   │       ├── auth (verify_api_key)
   │       ├── tool router classifier predicts intent (fastText)
   │       │
   │       ├── ┬── FAST path (try_stream_voice_response):
   │       │   │   - router predicts conversational/search → bypass tool loop
   │       │   │   - stream LLM tokens → sentence chunker → TTS → audio bytes
   │       │   │   - returns 200 audio/raw PCM + X-Audio-* headers
   │       │   │
   │       │   ├── TOOL-STREAM path (try_stream_voice_response_with_tools):
   │       │   │   - gated by JARVIS_STREAM_TOOL_RESPONSES env flag
   │       │   │   - runs iter 1 blocking (tool exec) then streams iter 2 prose → TTS
   │       │   │
   │       │   └── STANDARD path (process_voice_command_with_tools):
   │       │       - blocking iter 1 → tool_calls returned
   │       │       - if assistant_message has prose: stream_text_as_audio → 200 audio/raw
   │       │       - if tool_calls (no prose): return 202 JSON; node executes tools,
   │       │         then POSTs /voice/command/continue(/stream) with results
   │
   │ POST /voice/command/continue/stream   (if tool calls happened)
   │   { conversation_id, tool_results[] }
   │
   │       ├── auth
   │       ├── _maybe_push_actions_to_inbox(tool_results, node)
   │       │     └── if any tool returned context.actions, POST to notifications inbox
   │       ├── continue_conversation_with_tool_results → stream prose → TTS audio
   │       └── 200 audio/raw   OR   202 JSON {"fallback":"use_blocking_continue"}
   │
   │ [Node plays audio. Done.]
```

**Latency-critical:** Every step in the hot path is instrumented through `latency_logger` (see `core/utils/latency_logger.py`). A trace per request goes to the `request_traces` table — viewable through `/admin/traces` (admin) or `/mobile/traces` (user).

**Parallel pings:**
- `POST /voice/acknowledge` — node calls this in parallel with `/voice/command/stream`. Returns an instant keyword-matched acknowledgment ("Sure", "On it") to speak while the main pipeline runs. **No LLM call** — avoids contending for llama.cpp's slot or evicting the prefix cache.

---

## The other hot path: mobile chat

`/api/v0/mobile/chat` and `/api/v0/mobile/audio` — JWT-authenticated, used by `jarvis-node-mobile`. Same model service under the hood but returns JSON (not audio streams). Mobile renders chat UI from the response. Tool results with `actions` go to the inbox the same way.

`/api/v0/mobile/voice-profiles` — manage voice enrollment from mobile (proxies to whisper).
`/api/v0/mobile/routines` — CRUD on routines, the automation bundles.
`/api/v0/mobile/node-tools` — mobile-side inspection of what tools a node has.
`/api/v0/mobile/traces` — request trace viewer scoped to the user.

---

## Background workers (started in `app.main:startup_event`)

| Task | Cadence | What |
|---|---|---|
| Provisioning token cleanup | 1h | Removes expired pre-claim tokens |
| Passive memory extraction | settings `memory.extraction_interval_seconds` (default 300s) | Reads recent transcripts, queues an LLM job to extract memories; controlled by `memory.extraction_enabled` setting |
| Transcript TTL cleanup | 24h | Deletes transcripts older than `memory.transcript_ttl_days` (default 7) |
| Trace TTL cleanup | 1h | Deletes request traces older than `tracing.retention_days` (default 7) |
| Expired-memory cleanup | 30min | Removes memories whose TTL has passed (agent-injected memories may have TTLs) |
| Node-task timeout sweep | 2min | Marks update tasks as `failed` if no progress 30 min after creation (covers offline node, dead installer, OOM) |
| Adapter auto-training | settings `adapter.auto_train_interval_seconds` (default 1h) | Off by default. Dormant pipeline — see "Adapter status" below. |

All workers are settings-driven where possible. If you add a new background worker, follow this pattern: `asyncio.create_task` in startup, read interval from `settings_service.get(...)` each iteration, catch & log all exceptions to keep the loop alive.

---

## "How to..." recipes

### Add a new server-side tool/command

1. Implement `IJarvisCommand` (interface in `jarvis-node-setup/core/ijarvis_command.py` and SDK-mirrored in `jarvis-command-sdk`).
2. Register it in `app/core/tool_registry.py` so the tool builder includes it in the prompt-tools list.
3. If it should also be invocable from a node, ship the same `IJarvisCommand` to the node (typically via Pantry package).
4. Add tests covering the tool's `validate_params`, `execute`, and error shape — see existing tools for pattern.
5. **If the tool returns actions** (interactive confirmations like "send this email?"), populate `context.actions`, `context.draft`, `context.preview`, and optionally `context.inbox_title`/`inbox_summary`. The `_maybe_push_actions_to_inbox` helper will route it to the notifications inbox automatically.

### Add a new node-callable API endpoint

Add a router under `app/api/`. Use `node_context_provider: NodeContextProvider = Depends(verify_api_key)` for node auth. Wire it up in `main.py`'s `include_router` block (search for "Include").

### Add a new mobile-only endpoint

Add to `app/api/mobile_*.py`. JWT auth — use the standard FastAPI Bearer dependency from `app/deps.py`. Mount under `/api/v0/mobile`.

### Add a new prompt provider

Implement under `app/core/prompt_providers/`. Register in `app/core/prompt_provider_factory.py:PromptProviderFactory`. Provider must implement: `build_system_prompt`, `build_training_prompt`, `build_training_completion`, `build_training_system_prompt`, and own its chat-format rules (think-block stripping, special tokens, etc.). The factory's `get_available_providers()` is auto-exposed in the `llm.interface` setting's options.

### Add a new setting

Append to `app/services/settings_definitions.py:SETTINGS_DEFINITIONS`. The settings router uses `jarvis-settings-client` with **combined-auth reads** (JWT or app-creds) and **superuser-only writes** — same pattern as auth and logs.

### Push something async to a node

Send via MQTT. The CC publishes to topics the node subscribes to. Existing patterns to copy: `node_settings.mqtt_client`, `package_install` (Pantry → install on node), `bluetooth` (scan/pair commands), `node_updates` (version pinning). **Topic conventions** live in the relevant route file — don't invent new ones without checking these.

### Queue an async LLM job

POST to llm-proxy's `/internal/queue/enqueue` with `job_type`, `request`, `callback`. Include `LLM_PROXY_INTERNAL_TOKEN` in `X-Internal-Token`. Implement a callback endpoint in `main.py` (see existing `/adapters/jobs/callback`, `/deep-research/callback`, `/memory-extraction/callback` for shape). Idempotency via `idempotency_key`.

---

## Invariants & gotchas

1. **Client-provided `node_context` is mostly ignored for security.** The authoritative context is built server-side from `NodeContextProvider` (which validates via auth). The client can provide `timezone`, `speaker_user_id`, and `agents` data — everything else (room, household_id, node_id, voice_mode) comes from the validated node row. If you add a new context field, decide which side is authoritative before merging it in.
2. **Adapter subsystem is dormant.** llama.cpp #21125 (Qwen3 GGUF LoRA conversion) blocks rollout. The training pipeline, scheduler, and callback handler are all wired but the `node_context["adapter_hash"]` injection is hard-disabled with `if False`. Don't expand this surface — if you need to change something here, ask first. The "orphaned 2026-04-25" comment in `main.py` captures the call-time cost reasoning.
3. **Three voice paths are stable architecture, not a transition.** Fast/tool-stream/blocking each serves a different traffic shape. Don't collapse them. The streaming paths are for the voice (Pi node) flow; mobile chat uses the blocking pipeline because it renders JSON, not audio.
4. **`/voice/command/stream` returns 200 OR 202 depending on outcome.** 200 = audio body. 202 = JSON body (tool calls, validation, error, or empty assistant message). The node logic must handle both content-types. Same split for `/voice/command/continue/stream`.
5. **MQTT is the only async server→node channel.** No SSE-to-node, no WebSocket push. If you need to send the node a message outside an HTTP response, use MQTT. The node has a long-lived MQTT subscription.
6. **`speaker_user_id` resolves to a name via `auth /internal/users/batch`**, cached briefly (`speaker_resolver`). Don't query auth per-request inline; use the resolver. If the user is unknown, the LLM gets `speaker_name = None` and the prompt provider handles it.
7. **Conversation cache is in-process only.** No Redis. `conversation_cache.py` holds warmed prompts, tools, and node context per `conversation_id`. Restarting command-center drops all warm conversations. Nodes are expected to call `/conversation/start` again if they get an error.
8. **The "Tool data:" prefix is internal history context.** The LLM sometimes prepends `[Tool data: ...]\n\n` to `assistant_message`. `main.py:769` strips it before returning to clients. Don't expose this to the LLM in further turns either.
9. **`JARVIS_TEST_MODE=1` enables adapter override on `conversation/start`.** Lets the eval harness force a specific adapter hash (or baseline = no adapter). Off in normal operation. Don't ship code that depends on this being on in production.
10. **Background workers swallow exceptions to stay alive.** Every `_periodic_*` loop logs and continues. **One worker failing doesn't kill any others.** If a worker is silently failing, grep its log line. Don't add `raise` to these loops.
11. **Memory extraction is opt-in.** Default `memory.extraction_enabled` is false-ish (the setting check looks for explicit truthy). The worker still wakes up on its interval but skips the LLM call. This is fine — extraction is privacy-sensitive.
12. **Provisioning tokens self-expire and are swept hourly.** If you see "expired provisioning tokens" logs, that's normal. Don't shorten the TTL without checking — it's tied to the installer flow.
13. **Web search is gated per-household and FAIL-CLOSED.** The `web_search.enabled` setting (default **false**) governs the outbound-web server tools `quick_search` (live inline lookups) and `deep_research` (background, push+inbox). The gate is enforced in `conversation_handler.py` at the warmup text-path tool whitelist (`_get_web_search_enabled` → `_safe_tool_names`) and the fast-stream eligibility check, plus an `execute()`-time re-check inside each tool. It is **fail-closed** (any settings error → disabled — deliberately the opposite of the memory gate's fail-open) because it controls outbound internet egress. There is **no keyword pre-exec** — web search flows through a single path: the model self-calling the tool (reliability depends on the tools-first prompt; see gotcha 15). **GOTCHA — double egress:** the gate only governs CC **server** tools. A node's own web-search **client** tool (the legacy `jarvis-cmd-web-search` `search_web` plugin) is merged into the prompt unfiltered in warmup and routed straight to the node — it is NOT governed by `web_search.enabled`. So "toggle off = no egress" only holds if that plugin isn't installed. Don't un-gate `warmup_service.merge_tools` without re-checking this. Households toggle the setting from mobile via `app/api/mobile_household_settings.py` (household-admin auth, not global superuser).
14. **Web search is VOICE-ONLY today.** The mobile/web chat path (`mobile_chat.py`) routes **every** tool call to the node over MQTT (`_route_tool_call_to_node`) and never executes CC **server** tools in-process. So `quick_search`/`deep_research` (and any server tool) don't work on chat — only on the voice/node path, where `tool_execution_engine` runs server tools server-side. Making web search work in browser/mobile chat requires teaching the chat path to execute server tools (distinguish server vs node tools) — a separate piece of work.

---

## API surface — overview

(Each router is a file under `app/api/` or top-level `app/`. Full route inventory in code; this lists the categories.)

### Voice (node auth, `X-API-Key: node_id:node_key`)
- `POST /api/v0/conversation/start` — warm a conversation
- `POST /api/v0/voice/command` — blocking JSON response
- `POST /api/v0/voice/command/stream` — 200 audio OR 202 JSON
- `POST /api/v0/voice/acknowledge` — instant keyword ack (no LLM)
- `POST /api/v0/voice/command/continue` / `/continue/stream` — return tool results

### Async callbacks (token-protected, called by llm-proxy)
- `POST /api/v0/adapters/jobs/callback` — adapter training results
- `POST /api/v0/deep-research/callback` — long-form research results
- `POST /api/v0/memory-extraction/callback` — passive memory extraction results

### Node-facing (node auth)
- `app/admin.py` — `/api/v0/admin/nodes` CRUD (admin token), plus public factory-reset verification
- `app/provisioning.py` — token-based new-node setup
- `app/chat.py` — direct chat (legacy/test)
- `app/date_context.py` — date helpers for nodes
- `app/node_settings.py` — settings push + MQTT
- `app/api/node_commands.py` — execute named commands on node
- `app/api/node_updates.py` — version pinning, mobile-triggered, MQTT-dispatched
- `app/api/wake_response.py` — text generator for wake-word responses
- `app/api/package_install.py` — Pantry package → install via MQTT
- `app/api/test_install.py` — Forge share code → temp install
- `app/api/test_commands.py` — app-to-app smoke testing
- `app/api/smart_home.py` — rooms, devices, config push
- `app/api/cameras.py` — go2rtc proxy
- `app/api/bluetooth.py` — scan, pair, disconnect via MQTT
- `app/api/oauth.py` — OAuth session management (for integrations needing it)

### Mobile-facing (`/api/v0/mobile/*`, JWT auth)
- `mobile_chat.py` — chat surface
- `mobile_audio.py` — audio handling
- `mobile_voice_profiles.py` — voice enrollment
- `node_tools.py` — node-tool inspection
- `routine_builder.py` — routine CRUD
- `traces.py:mobile_router` — request trace viewer
- `mobile_household_settings.py` — household-controllable settings (e.g. the web-search toggle). Authorized by **household role** (`verify_household_role`, admin to write), NOT a global superuser like the shared `/settings/*` router. Allowlist-gated to a fixed set of keys.

### Shared
- `app/api/callbacks.py` — interactive-notification callbacks (mobile tap → executor → result). **Two dispatch planes** since 2026-07: `target_node_id` set → node plane (MQTT → node executes the command's `@callback` method); omitted → **server plane** (CC executes a handler registered in `app/services/server_callback_registry.py` in a background task; body must carry `household_id`; membership enforced the same way). Server tools (phone-call confirm/escalation cards) register `(command_name, callback_name)` handlers at import time. Nodes can never read or complete a server-plane job; result recording (status poll + `new_notification` inbox fan-out) is identical on both planes. `callback_jobs.node_id` is nullable — NULL means server plane.
- `app/api/media.py` — proxy to whisper + tts (node auth)
- `app/api/memories.py` — user memory CRUD (admin or JWT)
- `app/api/transcripts.py` — transcript rating UI (mobile)
- `app/api/adapters.py` — adapter proposal/approval UI (dormant)
- `app/api/traces.py:admin_router` — admin trace viewer

### Training
- `POST /api/v0/tool-router/train` — fastText classifier training (gates the fast voice path)
- `POST /api/v0/adapters/train` — queue LoRA training (dormant)

### Settings
- `/settings/*` — combined-auth reads, superuser writes

---

## Data model (high level)

PostgreSQL, ~20 tables. Key ones (`app/models.py`):

| Table | Purpose |
|---|---|
| `nodes` | Node identity + api_key, room, household, voice_mode, adapter_hash (dormant) |
| `node_tasks` | Update tasks (state machine: pending → dispatched → in_progress → succeeded/failed) |
| `user_memories` | Per-user memories injected into prompt; optional TTL |
| `transcripts` | Voice command transcripts for passive memory extraction (TTL'd) |
| `request_traces` | Latency + trace data per request (TTL'd) |
| `routines` | Mobile-built automations |
| `rooms`, `devices` | Smart-home topology (parent_room_id for hierarchy) |
| `adapter_proposals`, `adapter_history` | Dormant adapter pipeline |
| `provisioning_tokens` | New-node setup tokens (TTL'd) |
| `inbox_*` | Confirmation inbox state (mirrored to notifications service) |
| `settings` | Standard multi-tenant settings table |

All TTL'd tables are cleaned by the background workers listed above.

---

## Config surface

Most behavior is controlled by **settings** (in DB), not env vars. Env vars are limited to bootstrap:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres connection |
| `JARVIS_CONFIG_URL` | Service discovery |
| `JARVIS_APP_ID` / `JARVIS_APP_KEY` | App credential for calling other services |
| `JARVIS_MCP_URL` | MCP server (optional) |
| `JARVIS_LLM_PROXY_API_URL` | Fallback only — prefer config-service discovery |
| `LLM_PROXY_INTERNAL_TOKEN` | Token for queue-enqueue calls |
| `JARVIS_ADAPTER_CALLBACK_TOKEN` | Validates inbound async-job callbacks (adapter / deep-research / memory). **Fail-closed:** if unset, those callbacks return 503 (not open) — set it in any non-trivial deploy. |
| `JARVIS_ALLOW_INSECURE_CALLBACKS` | Explicit opt-out (`1`/`true`) to run async-job callbacks **unauthenticated** when the token is unset — trusted local dev only, logged loudly at startup. Never set in prod. |
| `JARVIS_TEST_MODE` | Enables adapter override on `conversation/start` (eval harness only) |
| `JARVIS_STREAM_TOOL_RESPONSES` | Enables the tool-stream voice path |
| `JARVIS_LOG_CONSOLE_LEVEL` / `JARVIS_LOG_REMOTE_LEVEL` | Logging |
| `ADMIN_API_KEY` | Admin endpoint protection |
| `MQTT_*` | MQTT broker config (host, port, user, pass) |

Key settings (DB-driven, viewable via admin UI):
- `memory.extraction_enabled`, `memory.extraction_interval_seconds`, `memory.transcript_ttl_days`
- `tracing.retention_days`
- `adapter.auto_train_*` (dormant)
- `llm.interface` (which prompt provider to use)
- `web_search.enabled` (per-household, **default false**, fail-closed) — master toggle for the outbound-web tools `quick_search` + `deep_research`. See Invariants #13–14.

---

## Architecture

```
app/
├── main.py                              # FastAPI factory, startup workers, top-level voice routes
├── deps.py                              # verify_api_key, get_model_service
├── db.py / models.py                    # DB engine + ~20 SQLAlchemy models
├── debug_setup.py                       # debugpy hook (DEBUG=1)
├── chat.py / admin.py / provisioning.py # Top-level routers
├── date_context.py / node_settings.py
│
├── api/                                 # 24 routers — see API surface above
│
├── core/                                # Heavy lifting — LLM integration, tools, voice pipeline
│   ├── model_service.py                 # process_voice_command_with_tools, streaming variants
│   ├── model_factory.py
│   ├── llm_proxy_client.py              # httpx client for llm-proxy /api/v1/chat
│   ├── conversation_cache.py            # In-process warm-conversation store
│   ├── conversation_handler.py          # warmup + memory loading
│   ├── streaming_handler.py             # stream_text_as_audio (LLM tokens → TTS pipe)
│   ├── system_prompt_builder.py         # Speaker + memories + agents + room into prompt
│   ├── prompt_providers/                # qwen3, llama, hermes, ... — per-model templates
│   ├── prompt_provider_factory.py
│   ├── tool_router/                     # fastText classifier (fast voice path)
│   ├── tool_routing.py / tool_builder.py / tool_call_parser.py
│   ├── tool_execution_engine.py / tool_executor.py / tool_registry.py
│   ├── tools/                           # remember_tool, forget_tool (IServerTool)
│   ├── clients/                         # tts_client (streaming PCM)
│   ├── mqtt_client.py                   # Server → node async push
│   ├── service_config.py                # jarvis-config-client wrapper
│   ├── date_detector.py / date_replacer.py / date_resolution.py
│   ├── param_refinement.py / param_validation.py
│   ├── utils/
│   │   ├── speaker_resolver.py          # user_id → display name (cached)
│   │   ├── latency_logger.py            # per-request latency trace
│   │   └── rest_client.py               # httpx wrapper (test-mockable)
│   ├── warmup_service.py
│   └── ...
│
├── services/                            # Business logic between API and core
│   ├── memory_service.py / memory_extraction_service.py
│   ├── transcript_service.py
│   ├── settings_service.py / settings_definitions.py
│   ├── node_command_service.py
│   ├── agent_context_service.py
│   ├── deep_research_service.py
│   ├── adapter_*.py                     # Dormant subsystem
│   ├── prompt_provider_installer.py / prompt_variant_builder.py
│   ├── training_data_extractor.py / training_orchestrator.py
│   ├── inbox_notification_service.py    # Push to notifications inbox
│   ├── acknowledgment_service.py        # /voice/acknowledge keyword match
│   └── github_releases.py               # For node-updates flow
│
├── context_providers/
│   └── node_context_provider.py         # Wraps verify_api_key, exposes household + node
│
├── request_models/ + response_models/   # Pydantic
└── alembic/                             # Migrations
```

---

## Testing

- **Unit tests + DB-backed tests.** DB tests need Postgres in Docker — `python run_database_tests.py --type docker`.
- Tests under `tests/` are unit; database tests under `tests/database/` need the docker harness.
- Auth dependencies overridden via `dependency_overrides`.
- LLM proxy calls mocked at `app.core.utils.rest_client.post` boundary — `main.py:31` imports it specifically for this reason.
- **Cross-service integration tests.** `.github/workflows/integration-trigger.yml` fires a `repository_dispatch` to `jarvis-node-setup` on every PR. The receiver workflow there runs the integration suite against the PR's HEAD SHA and posts a `<!-- integration-test-results:v1 -->` comment + a `jarvis-integration` commit status back here. Requires the `INTEGRATION_DISPATCH_TOKEN` secret (fine-grained PAT, `contents:write` on `jarvis-node-setup`). Full design and ops guide: `jarvis-node-setup/docs/integration-tests.md`.

---

## Failure modes

| Failure | Behavior |
|---|---|
| llm-proxy down | All voice routes 5xx |
| auth down | Node validation 5xx → all voice routes fail |
| tts down | Streaming voice paths fail; blocking path still returns 202 JSON |
| whisper down | Media proxy fails; voice path itself unaffected (STT is on the node) |
| Postgres down | All routes 5xx |
| MCP server down | Logged as non-fatal; MCP tools unavailable in LLM context |
| MQTT broker down | Async server→node messages drop; HTTP responses still work |
| notifications down | `inbox_notification_service` fails silently — voice continues |
| Background worker exception | Logged, loop continues; the affected feature is degraded until next tick |
| Conversation cache miss (CC restarted) | `voice/command` returns 400 "Conversation not initialized" — node calls `/conversation/start` and retries |

---

## Out of scope / explicitly not here

- **STT.** Whisper runs in its own service; CC just proxies via `/media/whisper`.
- **TTS synthesis.** Piper runs in `jarvis-tts`; CC pipes tokens to it.
- **LLM inference.** llm-proxy owns the models. CC is the orchestrator.
- **Voice profile enrollment storage.** Whisper owns the model and profile storage; CC's `/media/whisper/voice-profiles/*` is a proxy.
- **Long-term conversation history.** Transcripts have a 7-day TTL by default. If you need durable history, that's a separate system.
- **Adapter rollout.** Dormant — see Invariants #2.

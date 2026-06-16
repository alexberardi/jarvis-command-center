# jarvis-command-center

The "brain" of the Jarvis ecosystem — the central voice-orchestration service. It receives voice commands and chat from Pi Zero nodes and the mobile/web clients, builds prompts, calls the LLM proxy for inference, routes to tools, runs routines and agents, manages long-term memory with embeddings, and coordinates STT/TTS and node management.

This is a large FastAPI service (tens of thousands of lines across `app/`, dozens of routers under `/api/v0`), **not** a minimal app. It requires **PostgreSQL with the `pgvector` extension** (used for memory/embedding storage) — there is no SQLite mode; the database layer rejects any non-PostgreSQL URL.

Runs on **port 7703**.

## What it does

- Voice + chat orchestration for nodes and the mobile/web clients
- Prompt building, model routing, and streaming via the LLM proxy
- Tool registry, tool routing/classification, and tool execution
- Long-term memory + embeddings (pgvector) and memory extraction
- Routines (scheduled/triggered) and agents
- Node provisioning, node settings, node commands, OTA node updates, Bluetooth
- Smart-home, media, cameras, OAuth relay integration
- Training-data extraction and tool-router classifier (optional)
- MQTT messaging to/from nodes

## Requirements

- Python 3.11+
- Docker & Docker Compose (recommended)
- PostgreSQL **with pgvector** (the standalone compose profile uses the `pgvector/pgvector:pg15` image)

## Setup & run

```bash
cp .env.example .env   # then edit values

# Local dev (kills the port, installs sibling jarvis client libs, hot reload):
./run.sh

# Docker dev (hot reload):
./run.sh --docker          # add --build, or --rebuild for a clean build
# equivalently:
docker compose -f docker-compose.dev.yaml up --build
```

The app module is `app.main:app` and the service listens on **7703**. Alembic migrations (`alembic upgrade head`) run automatically on container startup. To run a bundled pgvector Postgres for local testing, use the compose `standalone` profile:

```bash
docker compose -f docker-compose.dev.yaml --profile standalone up
```

To run Uvicorn directly (you must provide a reachable Postgres):

```bash
uvicorn app.main:app --reload --port 7703
```

## Environment

Copy `.env.example` to `.env`. Key variables:

| Variable | Purpose |
|---|---|
| `DB_URL` / `DATABASE_URL` | **Required.** PostgreSQL connection string (must be `postgresql://...`; pgvector required). |
| `MIGRATIONS_DATABASE_URL` | Postgres URL used for Alembic migrations (use `localhost` even when running in Docker). |
| `ADMIN_API_KEY` | Protects admin POST/PATCH/DELETE endpoints. |
| `JARVIS_CONFIG_URL` | jarvis-config-service URL (service discovery). Set `JARVIS_CONFIG_URL_STYLE=dockerized` in Docker. |
| `JARVIS_APP_ID` / `JARVIS_APP_KEY` | App-to-app credentials (key populated via service registration). |
| `JARVIS_RELAY_URL` | Public OAuth relay used to bounce provider auth codes back to the mobile app. |
| `JARVIS_MQTT_BROKER_URL` | MQTT broker for node messaging. |
| `JARVIS_MODEL_INTERFACE` | Model adapter interface (e.g. `JarvisAdapterModel`). |

## Endpoints

- `GET /health` — health check
- `GET /api/v0/ping` — basic liveness (`{"message": "pong"}`)
- Routers are mounted under `/api/v0` (and `/api/v0/admin`, `/api/v0/mobile`) — see `app/main.py` for the full set (admin, nodes, provisioning, chat, routines, agents, memories, smart-home, media, OAuth, node-updates, etc.). Node-facing routes authenticate with `X-API-Key`; admin routes use `ADMIN_API_KEY`.

## Migrations

```bash
alembic upgrade head                                  # apply
alembic revision --autogenerate -m "description"      # create
alembic history                                       # view
```

## Optional: tool-router classifier

A small local FastText tool-router classifier can be trained and enabled:

```bash
export JARVIS_TOOL_CLASSIFIER_EXTRA_TRAINING_PATH=/app/temp/tool_router_extra_training.jsonl
python -m app.core.tool_router.training

export JARVIS_TOOL_CLASSIFIER_ENABLED=true
export JARVIS_TOOL_CLASSIFIER_MODEL_PATH=/app/temp/tool_classifier.bin
```

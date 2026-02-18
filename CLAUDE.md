# jarvis-command-center

Central voice command API. Routes voice from Pi Zero nodes through speech-to-text, LLM processing, and tool execution.

## Quick Reference

```bash
# Run (Docker recommended - includes PostgreSQL)
bash run-docker-dev.sh

# Health check
curl http://localhost:7703/health

# Test (requires PostgreSQL via Docker)
python run_database_tests.py --type docker
```

## Architecture

```
app/
├── main.py              # FastAPI app, startup/shutdown
├── chat.py              # Voice command processing
├── admin.py             # Node CRUD
├── core/
│   ├── model_service.py # LLM integration, tool processing (large file)
│   └── conversation_cache.py  # Conversation state
├── context_providers/
│   └── node_context_provider.py  # Node auth
├── models.py            # SQLAlchemy models
└── request_models/      # Pydantic schemas
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `ADMIN_API_KEY` | Yes | Admin endpoint protection |
| `JARVIS_LLM_PROXY_API_URL` | Yes | LLM proxy service URL |
| `JARVIS_LOG_CONSOLE_LEVEL` | No | Logging level (default: INFO) |

## API Endpoints

**Voice:**
- `POST /api/v0/conversation/start` → Start conversation
- `POST /api/v0/voice/command` → Process voice command
- `POST /api/v0/voice/command/continue` → Continue with tool results

**Admin (requires X-Admin-Api-Key):**
- `GET /api/v0/admin/nodes` → List nodes
- `POST /api/v0/admin/nodes` → Create node
- `DELETE /api/v0/admin/nodes/{id}` → Delete node

**Training:**
- `POST /api/v0/tool-router/train` → Train tool router (fastText)
- `POST /api/v0/adapters/train` → Queue adapter training

**Health:**
- `GET /health` → Health check

## Node Authentication

Nodes authenticate via `X-API-Key` header. Keys are stored in the nodes table.

## Database

PostgreSQL is required. The docker-compose.dev.yaml includes a PostgreSQL container.

```bash
# Run with Docker (recommended)
bash run-docker-dev.sh

# Or run migrations manually
DATABASE_URL=postgresql://user:pass@localhost:5432/db alembic upgrade head

# Run tests with Docker PostgreSQL
python run_database_tests.py --type docker
```

## Dependencies

**Python Libraries:**
- FastAPI, SQLAlchemy, Alembic
- psycopg2 (PostgreSQL driver)
- httpx (service calls)
- fasttext (tool routing)
- jarvis-log-client

**Service Dependencies:**
- ✅ **Required**: `jarvis-llm-proxy-api` (7704) - LLM inference for command parsing
- ✅ **Required**: `jarvis-auth` (7701) - Node authentication validation
- ✅ **Required**: PostgreSQL - Database for nodes, conversations
- ⚠️ **Optional**: `jarvis-logs` (7702) - Centralized logging (degrades to console if unavailable)
- ⚠️ **Optional**: `jarvis-whisper-api` (7706) - Speech-to-text (if used)
- ⚠️ **Optional**: `jarvis-ocr-service` (7031) - OCR (if used)
- ⚠️ **Optional**: `jarvis-config-service` (7700) - Service discovery

**Used By:**
- `jarvis-node-setup` - Pi Zero voice nodes send commands here

**Impact if Down:**
- ❌ Voice commands cannot be processed
- ❌ Nodes cannot communicate with Jarvis
- ❌ No LLM-based intent classification or tool routing

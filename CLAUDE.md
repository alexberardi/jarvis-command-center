# jarvis-command-center

Central voice command API. Routes voice from Pi Zero nodes through speech-to-text, LLM processing, and tool execution.

## Quick Reference

```bash
# Run (Docker recommended)
bash run-docker-dev.sh

# Health check
curl http://localhost:8002/api/v0/health

# Test
python run_database_tests.py --type sqlite
poetry run pytest
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

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_TYPE` | sqlite | sqlite or postgres |
| `DB_URL` | - | Database connection string |
| `ADMIN_API_KEY` | - | Admin endpoint protection |
| `JARVIS_LLM_PROXY_API_URL` | - | LLM proxy service URL |
| `JARVIS_LOG_CONSOLE_LEVEL` | INFO | Logging level |

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
- `GET /api/v0/health` → Health check

## Node Authentication

Nodes authenticate via `X-API-Key` header. Keys are stored in the nodes table.

## Database

```bash
# Setup
python setup_database.py
alembic upgrade head

# Run tests
python run_database_tests.py --type sqlite
python run_database_tests.py --type postgres
```

## Dependencies

- FastAPI, SQLAlchemy, Alembic
- httpx (service calls)
- fasttext (tool routing)
- jarvis-log-client

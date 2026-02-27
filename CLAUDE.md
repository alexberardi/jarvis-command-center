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
├── api/
│   ├── media.py         # TTS/Whisper proxy endpoints
│   ├── memories.py      # User memory CRUD API
│   └── ...
├── core/
│   ├── model_service.py         # LLM integration, tool processing
│   ├── conversation_cache.py    # Conversation state
│   ├── conversation_handler.py  # Warmup, memory loading
│   ├── system_prompt_builder.py # Prompt construction (speaker + memories)
│   ├── utils/
│   │   └── speaker_resolver.py  # user_id → display name (cached)
│   └── tools/
│       ├── remember_tool.py     # IServerTool: save user memories
│       └── forget_tool.py       # IServerTool: remove user memories
├── services/
│   └── memory_service.py        # User memory CRUD + prompt formatting
├── context_providers/
│   └── node_context_provider.py # Node auth
├── models.py            # SQLAlchemy models (Node, UserMemory, etc.)
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

**Memories (requires admin key or JWT):**
- `GET /api/v0/memories?user_id=&household_id=` → List active memories
- `POST /api/v0/memories?user_id=&household_id=` → Create memory (upserts on key)
- `GET /api/v0/memories/{id}` → Get single memory
- `PUT /api/v0/memories/{id}` → Update memory
- `DELETE /api/v0/memories/{id}` → Soft-delete memory

**Media Proxy (node auth):**
- `POST /api/v0/media/whisper/transcribe` → Proxy to whisper
- `POST /api/v0/media/tts/speak` → Proxy to TTS
- `POST /api/v0/media/whisper/voice-profiles/enroll` → Enroll voice profile
- `DELETE /api/v0/media/whisper/voice-profiles/{user_id}` → Delete voice profile
- `GET /api/v0/media/whisper/voice-profiles` → List voice profiles

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

# Voice API

A minimal FastAPI app for handling voice commands from remote nodes.

- Uses SQLite for node context and API key storage
- Validates incoming requests via `X-API-Key` header
- Includes an admin API to manage nodes
- Alembic for migrations
- Admin POST/PATCH/DELETE routes protected by `ADMIN_API_KEY` from `.env`

## Dev Run
```bash
uvicorn app.main:app --reload --port 8001
```

## Docker Dev Run with Hot Reload
```bash
docker compose -f docker-compose.dev.yaml up --build
```

## Docker Run (Production-style)
```bash
docker build -t voice-api .
docker run -p 8001:8001 voice-api
```

## Endpoints
- `GET /ping` → Health check
- `POST /voice` → Voice input (requires `X-API-Key` header)
- `GET /admin/nodes` → List all nodes
- `POST /admin/nodes` → Create a node (requires `ADMIN_API_KEY`)
- `PATCH /admin/nodes/{node_id}` → Update a node (requires `ADMIN_API_KEY`)
- `DELETE /admin/nodes/{node_id}` → Delete a node (requires `ADMIN_API_KEY`)

## Migrations
```bash
alembic init alembic
# Edit alembic.ini & env.py to point to voice_api.db
alembic revision --autogenerate -m "init"
alembic upgrade head
```

## Persistent Database
The SQLite database is stored in `./data/voice_api.db` on the host machine and mounted into the container to ensure it persists across container rebuilds.


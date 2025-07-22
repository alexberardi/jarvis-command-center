# Voice API

A minimal FastAPI app for handling voice commands from remote nodes.

- Supports both SQLite (default) and PostgreSQL databases
- Validates incoming requests via `X-API-Key` header
- Includes an admin API to manage nodes
- Alembic for migrations
- Admin POST/PATCH/DELETE routes protected by `ADMIN_API_KEY` from `.env`

## Database Configuration

The application supports both SQLite and PostgreSQL databases. Configure using environment variables:

### Environment Variables
- `DB_TYPE`: Database type (`sqlite` or `postgres`, default: `sqlite`)
- `DB_URL`: Database connection URL
  - For SQLite: File path or SQLite URL (default: `sqlite:///./data/voice_api.db`)
  - For PostgreSQL: Full PostgreSQL URL (e.g., `postgresql://user:pass@localhost:5432/dbname`)

### Quick Setup
Run the database setup script to configure your database:
```bash
python setup_database.py
```

### Manual Configuration

#### SQLite (Default)
```bash
export DB_TYPE=sqlite
export DB_URL=sqlite:///./data/voice_api.db
```

#### PostgreSQL
```bash
export DB_TYPE=postgres
export DB_URL=postgresql://username:password@localhost:5432/jarvis_command_center
```

## Testing

The project includes comprehensive database tests to ensure both SQLite and PostgreSQL work correctly.

### Running Database Tests

#### SQLite Tests (Default)
```bash
python run_database_tests.py --type sqlite
```

#### PostgreSQL Tests (Requires PostgreSQL Server)
```bash
# Check if PostgreSQL is available
python run_database_tests.py --check-postgres

# Run PostgreSQL tests
python run_database_tests.py --type postgres
```

#### Docker PostgreSQL Tests
```bash
python run_database_tests.py --type docker
```

#### All Tests
```bash
python run_database_tests.py --type all
```

### Test Coverage

The database test suite covers:

- **Configuration Tests**: Environment variable parsing, URL generation
- **Connection Tests**: Database engine creation and connection validation
- **Integration Tests**: ORM operations, CRUD functionality
- **FastAPI Integration**: API endpoints with different database backends
- **Migration Tests**: Alembic integration with both databases
- **Error Handling**: Invalid configurations and connection failures

### Test Files

- `tests/test_database_config.py`: Core database configuration and integration tests
- `tests/test_postgres_integration.py`: PostgreSQL-specific integration tests
- `run_database_tests.py`: Test runner script with different configurations

## Dev Run
```bash
uvicorn app.main:app --reload --port 8001
```

## Docker Dev Run with Hot Reload
```bash
docker compose -f docker-compose.dev.yaml up --build
```

## Docker Run with PostgreSQL
```bash
docker-compose -f docker-compose.postgres.yaml up --build
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
# Run migrations (uses DB_TYPE and DB_URL environment variables)
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "description"

# View migration history
alembic history
```

## Database Storage

### SQLite
The SQLite database is stored in `./data/voice_api.db` on the host machine and mounted into the container to ensure it persists across container rebuilds.

### PostgreSQL
For PostgreSQL, ensure your database server is running and accessible. The application will create tables automatically on first run.


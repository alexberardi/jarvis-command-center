#!/bin/bash
# Development server with hot reload
# Usage: ./run.sh [--docker]

set -e
cd "$(dirname "$0")"

if [[ "$1" == "--docker" ]]; then
    # Docker development mode
    BUILD_FLAGS=""
    if [[ "$2" == "--rebuild" ]]; then
        docker compose --env-file .env -f docker-compose.dev.yaml build --no-cache
        BUILD_FLAGS="--build"
    elif [[ "$2" == "--build" ]]; then
        BUILD_FLAGS="--build"
    fi
    docker compose --env-file .env -f docker-compose.dev.yaml up $BUILD_FLAGS
else
    # Local development mode
    export $(grep -v '^#' .env | xargs)

    # Kill any existing process on the port
    PORT=${PORT:-8002}
    echo "Killing any existing process on port $PORT..."
    lsof -ti:$PORT | xargs kill -9 2>/dev/null || echo "No local process found on port $PORT"

    # Also stop any Docker containers using this port
    DOCKER_CONTAINER=$(docker ps --filter "publish=$PORT" --format "{{.Names}}" 2>/dev/null | head -1)
    if [ -n "$DOCKER_CONTAINER" ]; then
        echo "Stopping Docker container: $DOCKER_CONTAINER"
        docker stop "$DOCKER_CONTAINER" >/dev/null 2>&1 || true
    fi

    # Activate venv
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        source venv/bin/activate
    fi

    # Install jarvis-log-client from local path
    pip install -q -e ../jarvis-log-client 2>/dev/null || echo "Note: jarvis-log-client not found, remote logging disabled"

    echo "Starting server on port $PORT..."
    uvicorn app.main:app --host 0.0.0.0 --port $PORT --reload
fi

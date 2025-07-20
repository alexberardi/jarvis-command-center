#!/bin/bash
export $(grep -v '^#' .env | xargs)

# Kill any existing process on the port
PORT=${PORT:-9998}
echo "Killing any existing process on port $PORT..."
lsof -ti:$PORT | xargs kill -9 2>/dev/null || echo "No process found on port $PORT"

# Start virtual env
source venv/bin/activate

# run app
echo "Starting server on port $PORT..."
uvicorn app.main:app --host 0.0.0.0 --port $PORT --reload
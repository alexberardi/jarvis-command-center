#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
    VENV_PYTHON="${SCRIPT_DIR}/venv/bin/python"
fi

if [ ! -x "$VENV_PYTHON" ]; then
    echo "Error: .venv not found. Create it with:"
    echo "  python3 -m venv .venv && .venv/bin/pip install -e ."
    exit 1
fi

# Load .env for DATABASE_URL / MIGRATIONS_DATABASE_URL
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
fi

# Use MIGRATIONS_DATABASE_URL if set, otherwise DATABASE_URL
if [ -n "${MIGRATIONS_DATABASE_URL:-}" ]; then
    export DATABASE_URL="$MIGRATIONS_DATABASE_URL"
fi

exec "$VENV_PYTHON" -m alembic upgrade head

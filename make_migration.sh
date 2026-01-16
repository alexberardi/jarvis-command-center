#!/usr/bin/env bash
set -euo pipefail

# Create a new Alembic revision (autogenerate from models) using Python helper
# Usage: ./make_migration.sh "add new table"

if [ $# -lt 1 ]; then
  echo "Usage: $0 \"message for migration\""
  exit 1
fi

python scripts/make_migration.py "$*"


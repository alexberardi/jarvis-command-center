#!/usr/bin/env bash
set -euo pipefail

# Apply latest Alembic migrations using Python helper
python scripts/apply_migrations.py


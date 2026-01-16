#!/usr/bin/env python
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/make_migration.py "message"')
        sys.exit(1)

    # Load .env at repo root
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    # Allow override for host-run migrations
    migrations_url = os.getenv("MIGRATIONS_DATABASE_URL")
    if migrations_url:
        os.environ["DATABASE_URL"] = migrations_url

    message = sys.argv[1]
    cmd = [sys.executable, "-m", "alembic", "revision", "--autogenerate", "-m", message]
    result = subprocess.run(cmd, cwd=repo_root)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()


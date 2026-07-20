#!/usr/bin/env python3
"""Database test runner for Jarvis Command Center.

The DB-backed tests need a real PostgreSQL with pgvector (the embeddings
migration self-creates the `vector` extension, so the pgvector image is the
only setup required).

Two ways to get one:

    python run_database_tests.py                   # disposable one in Docker
    python run_database_tests.py --type postgres   # a server you already run

Port note. CI publishes its Postgres on 5433 and `tests/conftest.py` defaults
to that, which is correct in CI. On a dev box running the full jarvis stack,
5433 belongs to *pantry* — the suite then connects successfully to the wrong
server and fails with "password authentication failed for user jarvis_user",
which reads like bad credentials rather than a port collision. The Docker path
below therefore uses 5544 and sets TEST_DATABASE_URL explicitly instead of
relying on the default.
"""

import argparse
import os
import subprocess
import sys
import time

COMPOSE_FILE = "docker-compose.dev.yaml"
TEST_SERVICE = "postgres-test"
TEST_PORT = 5544
DIRECT_PORT = 5433  # matches CI and the conftest default

TEST_DB_URL_TEMPLATE = (
    "postgresql://jarvis_user:jarvis_password@localhost:{port}/jarvis_command_center"
)


def _pytest(extra_env: dict | None = None) -> bool:
    env = {**os.environ, **(extra_env or {})}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"], env=env
    ).returncode == 0


def _compose(*args: str) -> subprocess.CompletedProcess:
    """docker compose (v2) with the test profile enabled."""
    return subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "--profile", "test", *args],
        capture_output=True,
        text=True,
    )


def _wait_until_healthy(timeout_s: int = 90) -> bool:
    """Poll the container's healthcheck rather than sleeping a fixed amount."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        cid = _compose("ps", "-q", TEST_SERVICE).stdout.strip()
        if cid:
            state = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Health.Status}}", cid],
                capture_output=True,
                text=True,
            ).stdout.strip()
            if state == "healthy":
                return True
            if state == "unhealthy":
                return False
        time.sleep(2)
    return False


def run_postgres_tests() -> bool:
    """Run against a PostgreSQL you are already running (CI-style, port 5433)."""
    url = TEST_DB_URL_TEMPLATE.format(port=DIRECT_PORT)
    print(f"🧪 Running database tests against {url}")
    print("   Override with TEST_DATABASE_URL if your server is elsewhere.")
    return _pytest({"TEST_DATABASE_URL": os.getenv("TEST_DATABASE_URL", url)})


def run_docker_postgres_tests() -> bool:
    """Start a disposable Postgres, run the suite, then remove just that one."""
    print(f"🚀 Starting {TEST_SERVICE} on port {TEST_PORT}...")
    up = _compose("up", "-d", TEST_SERVICE)
    if up.returncode != 0:
        print(f"❌ Failed to start {TEST_SERVICE}:\n{up.stderr}")
        return False

    try:
        print("⏳ Waiting for it to report healthy...")
        if not _wait_until_healthy():
            print(f"❌ {TEST_SERVICE} never became healthy")
            return False
        url = TEST_DB_URL_TEMPLATE.format(port=TEST_PORT)
        print(f"🧪 Running database tests against {url}")
        return _pytest({"TEST_DATABASE_URL": url})
    finally:
        # Remove ONLY the test database. `docker compose down` here would take
        # the whole dev stack (jarvis-voice-api, caddy) with it — running the
        # test suite must never stop the app you are developing against.
        print(f"🧹 Removing {TEST_SERVICE}...")
        _compose("rm", "-sf", TEST_SERVICE)


def check_postgres_connection(port: int) -> bool:
    try:
        import psycopg2

        psycopg2.connect(
            host="localhost",
            port=port,
            dbname="jarvis_command_center",
            user="jarvis_user",
            password="jarvis_password",
            connect_timeout=5,
        ).close()
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run database tests for Jarvis Command Center"
    )
    parser.add_argument(
        "--type",
        choices=["postgres", "docker"],
        default="docker",
        help="docker = start a disposable Postgres (default); "
        "postgres = use one you already run",
    )
    parser.add_argument(
        "--check-postgres",
        action="store_true",
        help="Report whether a usable PostgreSQL is reachable",
    )
    args = parser.parse_args()

    if args.check_postgres:
        port = TEST_PORT if args.type == "docker" else DIRECT_PORT
        if check_postgres_connection(port):
            print(f"✅ PostgreSQL reachable on {port}")
        else:
            print(f"❌ No usable PostgreSQL on {port}")
            print(
                "   (a server answering with different credentials counts as "
                "unusable — check nothing else owns that port)"
            )
        return

    ok = (
        run_docker_postgres_tests()
        if args.type == "docker"
        else run_postgres_tests()
    )
    print("\n🎉 All tests passed!" if ok else "\n❌ Some tests failed!")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

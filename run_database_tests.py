#!/usr/bin/env python3
"""
Database Test Runner for Jarvis Command Center

This script provides easy ways to run database tests with PostgreSQL:
- Direct PostgreSQL tests (requires PostgreSQL server)
- Docker PostgreSQL tests (requires Docker Compose)
"""

import os
import sys
import subprocess
import argparse


def run_postgres_tests():
    """Run PostgreSQL database tests (requires PostgreSQL server)"""
    print("🧪 Running PostgreSQL database tests...")
    print("⚠️  This requires a PostgreSQL server running on localhost:5433")
    print("   Database: jarvis_command_center, User: jarvis_user, Password: jarvis_password")

    # Set environment for PostgreSQL tests
    os.environ["TEST_DATABASE_URL"] = "postgresql://jarvis_user:jarvis_password@localhost:5433/jarvis_command_center"

    # Run tests
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short"
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_docker_postgres_tests():
    """Run PostgreSQL tests with Docker Compose"""
    print("🧪 Running PostgreSQL tests with Docker Compose...")

    # Start Docker Compose services
    print("🚀 Starting PostgreSQL Docker container...")
    compose_cmd = [
        "docker-compose", "-f", "docker-compose.dev.yaml", "up", "-d", "postgres"
    ]

    result = subprocess.run(compose_cmd)
    if result.returncode != 0:
        print("❌ Failed to start PostgreSQL container")
        return False

    # Wait for PostgreSQL to be ready
    print("⏳ Waiting for PostgreSQL to be ready...")
    import time
    time.sleep(10)

    try:
        # Set environment for Docker PostgreSQL tests
        os.environ["TEST_DATABASE_URL"] = "postgresql://jarvis_user:jarvis_password@localhost:5433/jarvis_command_center"

        # Run tests
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "--tb=short"
        ]

        result = subprocess.run(cmd)
        return result.returncode == 0

    finally:
        # Clean up Docker services
        print("🧹 Cleaning up Docker services...")
        subprocess.run([
            "docker-compose", "-f", "docker-compose.dev.yaml", "down"
        ])


def check_postgres_connection():
    """Check if PostgreSQL server is available"""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5433,
            database="jarvis_command_center",
            user="jarvis_user",
            password="jarvis_password"
        )
        conn.close()
        return True
    except Exception as e:
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run database tests for Jarvis Command Center")
    parser.add_argument(
        "--type",
        choices=["postgres", "docker"],
        default="docker",
        help="Type of tests to run (default: docker)"
    )
    parser.add_argument(
        "--check-postgres",
        action="store_true",
        help="Check if PostgreSQL server is available"
    )

    args = parser.parse_args()

    if args.check_postgres:
        if check_postgres_connection():
            print("✅ PostgreSQL server is available")
        else:
            print("❌ PostgreSQL server is not available")
        return

    success = False

    if args.type == "postgres":
        success = run_postgres_tests()
    elif args.type == "docker":
        success = run_docker_postgres_tests()

    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

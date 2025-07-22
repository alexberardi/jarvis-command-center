#!/usr/bin/env python3
"""
Database Test Runner for Jarvis Command Center

This script provides easy ways to run different types of database tests:
- SQLite tests (default, fast)
- PostgreSQL tests (requires PostgreSQL server)
- Docker PostgreSQL tests (requires Docker Compose)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_sqlite_tests():
    """Run SQLite database tests"""
    print("🧪 Running SQLite database tests...")
    
    # Set environment for SQLite tests
    os.environ["DB_TYPE"] = "sqlite"
    os.environ["DB_URL"] = "sqlite:///:memory:"
    
    # Run tests
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_database_config.py",
        "-v",
        "--tb=short"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_postgres_tests():
    """Run PostgreSQL database tests (requires PostgreSQL server)"""
    print("🧪 Running PostgreSQL database tests...")
    print("⚠️  This requires a PostgreSQL server running on localhost:5432")
    print("   Database: test, User: test, Password: test")
    
    # Set environment for PostgreSQL tests
    os.environ["DB_TYPE"] = "postgres"
    os.environ["DB_URL"] = "postgresql://test:test@localhost:5432/test"
    os.environ["TEST_POSTGRES"] = "1"
    
    # Run tests
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_postgres_integration.py",
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
        "docker-compose", "-f", "docker-compose.postgres.yaml", "up", "-d", "postgres"
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
        os.environ["DB_TYPE"] = "postgres"
        os.environ["DB_URL"] = "postgresql://jarvis_user:jarvis_password@localhost:5433/jarvis_command_center"
        os.environ["TEST_POSTGRES_DOCKER"] = "1"
        
        # Run tests
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/test_postgres_integration.py::TestPostgreSQLDockerIntegration",
            "-v",
            "--tb=short"
        ]
        
        result = subprocess.run(cmd)
        return result.returncode == 0
        
    finally:
        # Clean up Docker services
        print("🧹 Cleaning up Docker services...")
        subprocess.run([
            "docker-compose", "-f", "docker-compose.postgres.yaml", "down"
        ])


def run_all_tests():
    """Run all database tests"""
    print("🧪 Running all database tests...")
    
    success = True
    
    # Run SQLite tests
    print("\n" + "="*50)
    print("SQLITE TESTS")
    print("="*50)
    if not run_sqlite_tests():
        success = False
    
    # Run PostgreSQL tests if available
    print("\n" + "="*50)
    print("POSTGRESQL TESTS")
    print("="*50)
    if not run_postgres_tests():
        print("⚠️  PostgreSQL tests skipped (server not available)")
    
    return success


def check_postgres_connection():
    """Check if PostgreSQL server is available"""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="test",
            user="test",
            password="test"
        )
        conn.close()
        return True
    except Exception:
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run database tests for Jarvis Command Center")
    parser.add_argument(
        "--type", 
        choices=["sqlite", "postgres", "docker", "all"],
        default="sqlite",
        help="Type of tests to run (default: sqlite)"
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
    
    if args.type == "sqlite":
        success = run_sqlite_tests()
    elif args.type == "postgres":
        success = run_postgres_tests()
    elif args.type == "docker":
        success = run_docker_postgres_tests()
    elif args.type == "all":
        success = run_all_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 
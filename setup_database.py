#!/usr/bin/env python3
"""
Database setup script for Jarvis Command Center

This script helps configure the database settings for the application.
It supports both SQLite (default) and PostgreSQL databases.
"""

import os
import sys
from pathlib import Path

def setup_sqlite():
    """Setup SQLite database configuration"""
    print("🔧 Setting up SQLite database...")
    
    # Create data directory if it doesn't exist
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Set environment variables for SQLite
    os.environ["DB_TYPE"] = "sqlite"
    os.environ["DB_URL"] = "sqlite:///./data/voice_api.db"
    
    print("✅ SQLite configuration:")
    print(f"   DB_TYPE: {os.environ['DB_TYPE']}")
    print(f"   DB_URL: {os.environ['DB_URL']}")
    print(f"   Database file: {data_dir.absolute()}/voice_api.db")

def setup_postgres():
    """Setup PostgreSQL database configuration"""
    print("🔧 Setting up PostgreSQL database...")
    
    # Get PostgreSQL connection details
    host = input("Enter PostgreSQL host (default: localhost): ").strip() or "localhost"
    port = input("Enter PostgreSQL port (default: 5432): ").strip() or "5432"
    database = input("Enter database name (default: jarvis_command_center): ").strip() or "jarvis_command_center"
    username = input("Enter username: ").strip()
    password = input("Enter password: ").strip()
    
    # Build PostgreSQL URL
    postgres_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    # Set environment variables for PostgreSQL
    os.environ["DB_TYPE"] = "postgres"
    os.environ["DB_URL"] = postgres_url
    
    print("✅ PostgreSQL configuration:")
    print(f"   DB_TYPE: {os.environ['DB_TYPE']}")
    print(f"   DB_URL: {os.environ['DB_URL']}")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Database: {database}")
    print(f"   Username: {username}")

def run_migrations():
    """Run database migrations"""
    print("\n🔄 Running database migrations...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], 
                              capture_output=True, text=True, check=True)
        print("✅ Migrations completed successfully!")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Migration failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error during migration: {e}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 Jarvis Command Center - Database Setup")
    print("=" * 50)
    
    # Check if environment variables are already set
    if os.getenv("DB_TYPE") and os.getenv("DB_URL"):
        print("⚠️  Database environment variables are already set:")
        print(f"   DB_TYPE: {os.getenv('DB_TYPE')}")
        print(f"   DB_URL: {os.getenv('DB_URL')}")
        
        choice = input("\nDo you want to reconfigure? (y/N): ").strip().lower()
        if choice != 'y':
            print("Using existing configuration.")
            run_migrations()
            return
    
    # Choose database type
    print("\n📋 Choose your database type:")
    print("1. SQLite (default, file-based, good for self-hosted)")
    print("2. PostgreSQL (recommended for production)")
    
    choice = input("\nEnter your choice (1 or 2, default: 1): ").strip() or "1"
    
    if choice == "1":
        setup_sqlite()
    elif choice == "2":
        setup_postgres()
    else:
        print("❌ Invalid choice. Using SQLite as default.")
        setup_sqlite()
    
    # Run migrations
    if run_migrations():
        print("\n🎉 Database setup completed successfully!")
        print("\n📝 To use these settings in your environment:")
        print(f"   export DB_TYPE={os.environ['DB_TYPE']}")
        print(f"   export DB_URL='{os.environ['DB_URL']}'")
        print("\n   Or add them to your .env file:")
        print(f"   DB_TYPE={os.environ['DB_TYPE']}")
        print(f"   DB_URL={os.environ['DB_URL']}")
    else:
        print("\n❌ Database setup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
# Test Suite Documentation

This directory contains tests for the Jarvis Command Center application with a focus on the tool-based flow and database configuration.

## Test Structure Overview

```
tests/
├── README.md                           # This documentation file
├── conftest.py                         # Pytest configuration and fixtures
├── pytest.ini                         # Pytest settings
├── test_*.py                          # Individual test files
└── __pycache__/                       # Python cache directory
```

## Current Tests

### `test_database_config.py`
Validates database configuration parsing and URL handling.

### `test_postgres_integration.py`
PostgreSQL integration tests (requires a running PostgreSQL instance).

## Test Execution

### Running All Tests
```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run with coverage
python -m pytest --cov=app
```

### Running Specific Test Categories
```bash
# Run database config tests
python -m pytest tests/test_database_config.py
```

### Running Individual Tests
```bash
# Run a specific test
python -m pytest tests/test_postgres_integration.py::TestPostgreSQLIntegration::test_connection
```

## Notes

This test set is intentionally minimal while the tool-based flow stabilizes. Add new tests alongside new features.
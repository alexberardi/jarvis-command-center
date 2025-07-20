#!/bin/bash

echo "Running Jarvis Command Center Test Suite..."
echo "==========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating venv..."
    python3 -m venv venv
    INSTALL_DEPS=true
else
    INSTALL_DEPS=false
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if requirements need to be installed/updated
if [ "$INSTALL_DEPS" = true ] || [ requirements.txt -nt venv/pyvenv.cfg ] || [ ! -f "venv/.requirements_installed" ]; then
    echo "Installing/updating requirements..."
    pip install -r requirements.txt
    # Create a marker file to track when requirements were last installed
    touch venv/.requirements_installed
else
    echo "Requirements up to date, skipping installation..."
fi

# Install pytest if not already installed
if ! command -v pytest &> /dev/null; then
    echo "Installing pytest..."
    pip install pytest
fi

# Run tests with verbose output
echo ""
echo "Running tests..."
echo "=================="
pytest tests/ -v --tb=short

echo ""
echo "Test run completed!"
echo "Deactivating virtual environment..."
deactivate 
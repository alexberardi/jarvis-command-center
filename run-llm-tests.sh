#!/bin/bash

echo "Running Jarvis LLM Integration Tests..."
echo "======================================"
echo ""
echo "⚠️  WARNING: These tests make real API calls to your LLM!"
echo "   - Tests will be slower (several seconds per test)"
echo "   - Ensure your LLM proxy is running and accessible"
echo "   - Set JARVIS_LLM_PROXY_API_URL environment variable"
echo ""

# Check if LLM API URL is configured
if [ -z "$JARVIS_LLM_PROXY_API_URL" ]; then
    echo "❌ JARVIS_LLM_PROXY_API_URL is not set!"
    echo "   Please set this environment variable to your LLM proxy URL"
    echo "   Example: export JARVIS_LLM_PROXY_API_URL=http://10.0.0.69:8000"
    exit 1
fi

echo "✅ LLM API URL configured: $JARVIS_LLM_PROXY_API_URL"
echo ""

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
    touch venv/.requirements_installed
else
    echo "Requirements up to date, skipping installation..."
fi

# Install pytest if not already installed
if ! command -v pytest &> /dev/null; then
    echo "Installing pytest..."
    pip install pytest
fi

# Run LLM integration tests
echo ""
echo "Running LLM integration tests..."
echo "================================="
echo "⏱️  This may take several minutes..."
echo ""

pytest tests/test_llm_integration.py -v -s --tb=short

echo ""
echo "LLM integration tests completed!"
echo "Deactivating virtual environment..."
deactivate 
#!/bin/bash

echo "Running Jarvis LLM Integration Tests..."
echo "======================================"
echo ""
echo "⚠️  WARNING: These tests make real API calls to your LLM!"
echo "   - Tests will be slower (several seconds per test)"
echo "   - Ensure your LLM proxy is running and accessible"
echo "   - Set JARVIS_LLM_PROXY_API_URL, LLM_APP_ID, and LLM_APP_KEY"
echo ""

# Check if LLM proxy authentication is configured
missing_vars=()
if [ -z "$JARVIS_LLM_PROXY_API_URL" ]; then
    missing_vars+=("JARVIS_LLM_PROXY_API_URL")
fi
if [ -z "$LLM_APP_ID" ]; then
    missing_vars+=("LLM_APP_ID")
fi
if [ -z "$LLM_APP_KEY" ]; then
    missing_vars+=("LLM_APP_KEY")
fi

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo "❌ Missing required environment variables: ${missing_vars[*]}"
    echo "   Please set these before running tests. Examples:"
    echo "   export JARVIS_LLM_PROXY_API_URL=http://10.0.0.69:8000"
    echo "   export LLM_APP_ID=<your-app-id>"
    echo "   export LLM_APP_KEY=<your-app-key>"
    exit 1
fi

echo "✅ LLM API URL configured: $JARVIS_LLM_PROXY_API_URL"
echo "✅ LLM app credentials configured"
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
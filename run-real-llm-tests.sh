#!/bin/bash

# Script to run real LLM API tests with metrics collection
# This script runs tests that make actual HTTP calls to the LLM proxy API

set -e

echo "🧪 Real LLM API Tests Runner"
echo "============================"

# Check if we're in the right directory
if [ ! -f "app/main.py" ]; then
    echo "❌ Error: Please run this script from the jarvis-command-center root directory"
    exit 1
fi

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "📄 Loading configuration from .env file"
    export $(grep -v '^#' .env | xargs)
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Virtual environment not detected"
    echo "   Consider activating your virtual environment first"
fi

# Check if required LLM proxy environment variables are set (only required for running tests, not listing)
MISSING_VARS=()
if [ -z "$JARVIS_LLM_PROXY_API_URL" ]; then
    MISSING_VARS+=("JARVIS_LLM_PROXY_API_URL")
fi
if [ -z "$LLM_APP_ID" ]; then
    MISSING_VARS+=("LLM_APP_ID")
fi
if [ -z "$LLM_APP_KEY" ]; then
    MISSING_VARS+=("LLM_APP_KEY")
fi

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo "⚠️  Warning: Missing required environment variables: ${MISSING_VARS[*]}"
    echo "   These are required for running tests, but not for listing them."
    echo ""
    echo "To run tests, set:"
    echo "   export JARVIS_LLM_PROXY_API_URL=http://your-llm-proxy-url:port"
    echo "   export LLM_APP_ID=<your-app-id>"
    echo "   export LLM_APP_KEY=<your-app-key>"
    echo "   or add them to your .env file"
    echo ""
else
    echo "✅ LLM API URL configured: $JARVIS_LLM_PROXY_API_URL"
    echo "✅ LLM app headers configured"
fi
echo ""

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --save-metrics     Save detailed metrics to JSON file"
    echo "  --output-file=FILE Specify output file for metrics (default: real_llm_test_metrics.json)"
    echo "  --list-tests       List all real LLM API tests without running them"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run tests with basic output"
    echo "  $0 --save-metrics            # Run tests and save metrics"
    echo "  $0 --output-file=my_metrics.json  # Save metrics to specific file"
    echo "  $0 --list-tests              # List tests without running"
    echo ""
}

# Parse command line arguments
SAVE_METRICS=""
OUTPUT_FILE=""
LIST_TESTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --save-metrics)
            SAVE_METRICS="--save-metrics"
            shift
            ;;
        --output-file=*)
            OUTPUT_FILE="--output-file=${1#*=}"
            shift
            ;;
        --list-tests)
            LIST_TESTS="--list-tests"
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Build the command
CMD="python run_real_llm_tests.py"

if [ -n "$SAVE_METRICS" ]; then
    CMD="$CMD $SAVE_METRICS"
fi

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD $OUTPUT_FILE"
fi

if [ -n "$LIST_TESTS" ]; then
    CMD="$CMD $LIST_TESTS"
fi

echo "🚀 Running command: $CMD"
echo ""

# Run the command
eval $CMD

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Real LLM API tests completed successfully!"
else
    echo ""
    echo "❌ Real LLM API tests failed!"
    exit 1
fi 
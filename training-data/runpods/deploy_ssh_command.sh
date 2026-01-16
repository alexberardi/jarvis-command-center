#!/bin/bash

# RunPods Deployment Script that accepts full SSH command
# Usage: ./deploy_ssh_command.sh "ssh root@host -p port -i keypath"
# Example: ./deploy_ssh_command.sh "ssh root@194.68.245.27 -p 22129 -i ~/.ssh/id_ed25519"

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_usage() {
    echo "Usage: $0 \"ssh user@host -p port -i keypath\""
    echo ""
    echo "Examples:"
    echo "  $0 \"ssh root@194.68.245.27 -p 22129 -i ~/.ssh/id_ed25519\""
    echo "  $0 \"ssh ubuntu@1.2.3.4 -p 22 -i ~/.ssh/my_key\""
    echo ""
    echo "Or use the individual arguments version:"
    echo "  ./deploy_to_runpods.sh 194.68.245.27 22129 ~/.ssh/id_ed25519"
}

# Check if argument provided
if [ $# -eq 0 ]; then
    print_error "No SSH command provided"
    print_usage
    exit 1
fi

SSH_COMMAND="$1"

# Parse the SSH command
if [[ ! "$SSH_COMMAND" =~ ^ssh ]]; then
    print_error "Command must start with 'ssh'"
    print_usage
    exit 1
fi

print_status "Parsing SSH command: $SSH_COMMAND"

# Extract components using regex and parameter expansion
# Remove 'ssh ' from the beginning
PARAMS="${SSH_COMMAND#ssh }"

# Initialize variables
USER=""
HOST=""
PORT=""
KEY_PATH=""

# Parse parameters
while [[ $PARAMS ]]; do
    case $PARAMS in
        -p\ *)
            # Extract port number
            PARAMS="${PARAMS#-p }"
            PORT="${PARAMS%% *}"
            PARAMS="${PARAMS#$PORT}"
            PARAMS="${PARAMS# }"
            ;;
        -i\ *)
            # Extract key path
            PARAMS="${PARAMS#-i }"
            KEY_PATH="${PARAMS%% *}"
            PARAMS="${PARAMS#$KEY_PATH}"
            PARAMS="${PARAMS# }"
            ;;
        *@*)
            # Extract user@host
            USER_HOST="${PARAMS%% *}"
            USER="${USER_HOST%@*}"
            HOST="${USER_HOST#*@}"
            PARAMS="${PARAMS#$USER_HOST}"
            PARAMS="${PARAMS# }"
            ;;
        *)
            # Skip unknown parameters
            PARAM="${PARAMS%% *}"
            PARAMS="${PARAMS#$PARAM}"
            PARAMS="${PARAMS# }"
            ;;
    esac
done

# Validate extracted parameters
if [ -z "$HOST" ]; then
    print_error "Could not extract host from SSH command"
    print_usage
    exit 1
fi

if [ -z "$PORT" ]; then
    print_error "Could not extract port from SSH command"
    print_usage
    exit 1
fi

if [ -z "$KEY_PATH" ]; then
    print_error "Could not extract key path from SSH command"
    print_usage
    exit 1
fi

# Default user to root if not specified
if [ -z "$USER" ]; then
    USER="root"
fi

print_status "Extracted parameters:"
print_status "  Host: $HOST"
print_status "  Port: $PORT"
print_status "  User: $USER"
print_status "  Key:  $KEY_PATH"

# Call the main deployment script with extracted parameters
print_status "Calling deployment script..."
exec ./deploy_to_runpods.sh "$HOST" "$PORT" "$KEY_PATH" "$USER"

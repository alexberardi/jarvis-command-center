#!/bin/bash

# Robust RunPods Deployment Script with Better Error Handling
# Handles large file uploads and connection issues

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

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_usage() {
    echo "Usage: $0 \"ssh user@host -p port -i keypath\""
    echo ""
    echo "Examples:"
    echo "  $0 \"ssh root@194.68.245.27 -p 22129 -i ~/.ssh/id_ed25519\""
    echo ""
    echo "Features:"
    echo "  - Robust file uploads with retry logic"
    echo "  - Compression for large files"
    echo "  - Skip large files option"
    echo "  - Connection testing"
}

# Parse SSH command (reusing logic from deploy_ssh_command.sh)
if [ $# -eq 0 ]; then
    print_error "No SSH command provided"
    print_usage
    exit 1
fi

SSH_COMMAND="$1"

if [[ ! "$SSH_COMMAND" =~ ^ssh ]]; then
    print_error "Command must start with 'ssh'"
    print_usage
    exit 1
fi

print_status "Parsing SSH command: $SSH_COMMAND"

# Extract components
PARAMS="${SSH_COMMAND#ssh }"
USER=""
HOST=""
PORT=""
KEY_PATH=""

while [[ $PARAMS ]]; do
    case $PARAMS in
        -p\ *)
            PARAMS="${PARAMS#-p }"
            PORT="${PARAMS%% *}"
            PARAMS="${PARAMS#$PORT}"
            PARAMS="${PARAMS# }"
            ;;
        -i\ *)
            PARAMS="${PARAMS#-i }"
            KEY_PATH="${PARAMS%% *}"
            PARAMS="${PARAMS#$KEY_PATH}"
            PARAMS="${PARAMS# }"
            ;;
        *@*)
            USER_HOST="${PARAMS%% *}"
            USER="${USER_HOST%@*}"
            HOST="${USER_HOST#*@}"
            PARAMS="${PARAMS#$USER_HOST}"
            PARAMS="${PARAMS# }"
            ;;
        *)
            PARAM="${PARAMS%% *}"
            PARAMS="${PARAMS#$PARAM}"
            PARAMS="${PARAMS# }"
            ;;
    esac
done

# Validate and set defaults
if [ -z "$HOST" ] || [ -z "$PORT" ] || [ -z "$KEY_PATH" ]; then
    print_error "Could not extract required parameters from SSH command"
    print_usage
    exit 1
fi

if [ -z "$USER" ]; then
    USER="root"
fi

SSH_KEY_PATH="${KEY_PATH/#\~/$HOME}"

print_status "Connection details:"
print_status "  Host: $HOST"
print_status "  Port: $PORT"
print_status "  User: $USER"
print_status "  Key:  $SSH_KEY_PATH"

# Function to upload file with retry logic
upload_file_robust() {
    local local_file="$1"
    local remote_path="$2"
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        print_status "Uploading $(basename "$local_file") (attempt $((retry_count + 1))/$max_retries)..."
        
        if scp -i "$SSH_KEY_PATH" -P "$PORT" -o ConnectTimeout=30 -o ServerAliveInterval=60 "$local_file" "$USER@$HOST:$remote_path"; then
            print_success "Successfully uploaded $(basename "$local_file")"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                print_warning "Upload failed, retrying in 5 seconds..."
                sleep 5
            fi
        fi
    done
    
    print_error "Failed to upload $(basename "$local_file") after $max_retries attempts"
    return 1
}

# Function to upload compressed file
upload_compressed() {
    local local_file="$1"
    local remote_path="$2"
    local compressed_file="${local_file}.gz"
    
    print_status "Compressing $(basename "$local_file")..."
    gzip -c "$local_file" > "$compressed_file"
    
    if upload_file_robust "$compressed_file" "${remote_path}.gz"; then
        print_status "Decompressing on remote server..."
        ssh -i "$SSH_KEY_PATH" -p "$PORT" "$USER@$HOST" "
            cd $(dirname "$remote_path") && 
            gunzip -f $(basename "${remote_path}.gz")
        "
        rm "$compressed_file"
        return 0
    else
        rm "$compressed_file"
        return 1
    fi
}

# Test SSH connection
print_status "Testing SSH connection..."
if ! ssh -i "$SSH_KEY_PATH" -p "$PORT" -o ConnectTimeout=10 -o BatchMode=yes "$USER@$HOST" exit 2>/dev/null; then
    print_error "Cannot connect to RunPods instance"
    exit 1
fi
print_success "SSH connection successful"

# Create remote directory structure
print_status "Creating remote directory structure..."
ssh -i "$SSH_KEY_PATH" -p "$PORT" "$USER@$HOST" "
    mkdir -p /workspace/date-finetuning/data
    mkdir -p /workspace/date-finetuning/scripts
    mkdir -p /workspace/date-finetuning/logs
    mkdir -p /workspace/date-finetuning/checkpoints
"

# Upload scripts (small files, should work fine)
print_status "Uploading scripts..."
upload_file_robust "./setup_runpods.sh" "/workspace/date-finetuning/setup_runpods.sh"
upload_file_robust "./fine_tune_single_shot_v1.py" "/workspace/date-finetuning/fine_tune_single_shot_v1.py"
upload_file_robust "./test_model.py" "/workspace/date-finetuning/test_model.py"
upload_file_robust "./requirements.txt" "/workspace/date-finetuning/requirements.txt"

# Handle training data uploads with size-based strategy
TRAINING_DATA_DIR="../relative-date-training"

if [ -d "$TRAINING_DATA_DIR" ]; then
    print_status "Analyzing training data files..."
    
    # Ask user which files to upload
    echo ""
    print_status "Available training data files:"
    ls -lh "$TRAINING_DATA_DIR"/*.jsonl
    echo ""
    
    print_warning "The large 79MB file may cause connection issues."
    echo "Options:"
    echo "1. Upload smaller files only (3anchors: 715KB, 5anchors: 1.2MB, 50anchors: 8.5MB)"
    echo "2. Upload all files with compression"
    echo "3. Skip data upload (upload manually later)"
    
    read -p "Choose option (1/2/3): " -n 1 -r
    echo
    
    case $REPLY in
        1)
            print_status "Uploading smaller training files..."
            upload_file_robust "$TRAINING_DATA_DIR/date_training_data_3anchors.jsonl" "/workspace/date-finetuning/data/date_training_data_3anchors.jsonl"
            upload_file_robust "$TRAINING_DATA_DIR/date_training_data_5anchors.jsonl" "/workspace/date-finetuning/data/date_training_data_5anchors.jsonl"
            upload_file_robust "$TRAINING_DATA_DIR/date_training_data_50anchors.jsonl" "/workspace/date-finetuning/data/date_training_data_50anchors.jsonl"
            ;;
        2)
            print_status "Uploading all files with compression..."
            for file in "$TRAINING_DATA_DIR"/*.jsonl; do
                filename=$(basename "$file")
                if ! upload_compressed "$file" "/workspace/date-finetuning/data/$filename"; then
                    print_warning "Failed to upload $filename, continuing with others..."
                fi
            done
            ;;
        3)
            print_status "Skipping data upload. You can upload manually later."
            ;;
        *)
            print_warning "Invalid option, uploading smaller files only..."
            upload_file_robust "$TRAINING_DATA_DIR/date_training_data_3anchors.jsonl" "/workspace/date-finetuning/data/date_training_data_3anchors.jsonl"
            upload_file_robust "$TRAINING_DATA_DIR/date_training_data_5anchors.jsonl" "/workspace/date-finetuning/data/date_training_data_5anchors.jsonl"
            upload_file_robust "$TRAINING_DATA_DIR/date_training_data_50anchors.jsonl" "/workspace/date-finetuning/data/date_training_data_50anchors.jsonl"
            ;;
    esac
else
    print_error "Training data directory not found: $TRAINING_DATA_DIR"
fi

# Make scripts executable
print_status "Making scripts executable..."
ssh -i "$SSH_KEY_PATH" -p "$PORT" "$USER@$HOST" "
    chmod +x /workspace/date-finetuning/setup_runpods.sh
    chmod +x /workspace/date-finetuning/fine_tune_single_shot_v1.py
    chmod +x /workspace/date-finetuning/test_model.py
"

print_success "✅ Deployment completed!"

# Show uploaded files
print_status "📋 Uploaded files:"
ssh -i "$SSH_KEY_PATH" -p "$PORT" "$USER@$HOST" "
    echo 'Scripts:'
    ls -la /workspace/date-finetuning/*.{sh,py} 2>/dev/null || echo 'No scripts found'
    echo ''
    echo 'Training Data:'
    ls -lh /workspace/date-finetuning/data/ 2>/dev/null || echo 'No data files found'
"

# Ask about setup and training
echo ""
read -p "Do you want to run the setup script now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Running setup script..."
    ssh -i "$SSH_KEY_PATH" -p "$PORT" "$USER@$HOST" "cd /workspace/date-finetuning && ./setup_runpods.sh"
    print_success "Setup completed!"
    
    echo ""
    read -p "Do you want to start training now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Starting training in tmux session..."
        ssh -i "$SSH_KEY_PATH" -p "$PORT" "$USER@$HOST" "
            cd /workspace/date-finetuning && 
            tmux new-session -d -s training 'source venv/bin/activate && python fine_tune_single_shot_v1.py --push-to-hf --repo-name alexberardi/jarvis-llama-3b --max-examples 15000 --public; exec bash'
        "
        print_success "Training started in tmux session 'training'"
        print_status "To monitor: ssh and run 'tmux attach -t training'"
    fi
fi

print_success "🎉 Robust deployment completed!"
echo ""
print_status "Manual upload command for large file (if needed):"
echo "scp -i $SSH_KEY_PATH -P $PORT $TRAINING_DATA_DIR/date_training_data.jsonl $USER@$HOST:/workspace/date-finetuning/data/"

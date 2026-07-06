#!/bin/bash

# RunPods Deployment Script with SSH Arguments
# Usage: ./deploy_to_runpods.sh <host> <port> <key_path> [user]
# Example: ./deploy_to_runpods.sh 203.0.113.10 22000 ~/.ssh/id_ed25519

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
    echo "Usage: $0 <host> <port> <key_path> [user]"
    echo ""
    echo "Examples:"
    echo "  $0 203.0.113.10 22000 ~/.ssh/id_ed25519"
    echo "  $0 203.0.113.10 22000 ~/.ssh/id_ed25519 root"
    echo ""
    echo "Your SSH command: ssh root@203.0.113.10 -p 22000 -i ~/.ssh/id_ed25519"
    echo "Becomes: $0 203.0.113.10 22000 ~/.ssh/id_ed25519"
}

# Check arguments
if [ $# -lt 3 ]; then
    print_error "Missing required arguments"
    print_usage
    exit 1
fi

# Parse arguments
RUNPODS_HOST="$1"
RUNPODS_PORT="$2"
SSH_KEY_PATH="$3"
RUNPODS_USER="${4:-root}"  # Default to root if not specified

# Expand tilde in SSH key path
SSH_KEY_PATH="${SSH_KEY_PATH/#\~/$HOME}"

print_status "🚀 Starting RunPods deployment..."
print_status "Target: $RUNPODS_USER@$RUNPODS_HOST:$RUNPODS_PORT"
print_status "SSH Key: $SSH_KEY_PATH"

# Check if SSH key exists
if [ ! -f "$SSH_KEY_PATH" ]; then
    print_error "SSH key not found at: $SSH_KEY_PATH"
    exit 1
fi

# Test SSH connection
print_status "Testing SSH connection..."
if ssh -i "$SSH_KEY_PATH" -p "$RUNPODS_PORT" -o ConnectTimeout=10 -o BatchMode=yes "$RUNPODS_USER@$RUNPODS_HOST" exit 2>/dev/null; then
    print_success "SSH connection successful"
else
    print_error "Cannot connect to RunPods instance"
    print_warning "Check your host, port, and SSH key configuration"
    exit 1
fi

# Create remote directory structure
print_status "Creating remote directory structure..."
ssh -i "$SSH_KEY_PATH" -p "$RUNPODS_PORT" "$RUNPODS_USER@$RUNPODS_HOST" "
    mkdir -p /workspace/date-finetuning/data
    mkdir -p /workspace/date-finetuning/scripts
    mkdir -p /workspace/date-finetuning/logs
    mkdir -p /workspace/date-finetuning/checkpoints
"

# Upload setup script
print_status "Uploading setup script..."
scp -i "$SSH_KEY_PATH" -P "$RUNPODS_PORT" "./setup_runpods.sh" "$RUNPODS_USER@$RUNPODS_HOST:/workspace/date-finetuning/setup_runpods.sh"

# Upload fine-tuning script
print_status "Uploading fine-tuning script..."
scp -i "$SSH_KEY_PATH" -P "$RUNPODS_PORT" "./fine_tune_model.py" "$RUNPODS_USER@$RUNPODS_HOST:/workspace/date-finetuning/fine_tune_model.py"

# Upload test script
print_status "Uploading test script..."
scp -i "$SSH_KEY_PATH" -P "$RUNPODS_PORT" "./test_model.py" "$RUNPODS_USER@$RUNPODS_HOST:/workspace/date-finetuning/test_model.py"

# Upload requirements file
print_status "Uploading requirements..."
scp -i "$SSH_KEY_PATH" -P "$RUNPODS_PORT" "./requirements.txt" "$RUNPODS_USER@$RUNPODS_HOST:/workspace/date-finetuning/requirements.txt"

# Upload training data
print_status "Uploading training data..."
TRAINING_DATA_DIR="../relative-date-training"

if [ -d "$TRAINING_DATA_DIR" ]; then
    # Upload all JSONL files
    print_status "Uploading JSONL training files..."
    scp -i "$SSH_KEY_PATH" -P "$RUNPODS_PORT" "$TRAINING_DATA_DIR"/*.jsonl "$RUNPODS_USER@$RUNPODS_HOST:/workspace/date-finetuning/data/"
    
    # Also upload the schema and generation script for reference
    if [ -f "$TRAINING_DATA_DIR/date_output_schema.py" ]; then
        scp -i "$SSH_KEY_PATH" -P "$RUNPODS_PORT" "$TRAINING_DATA_DIR/date_output_schema.py" "$RUNPODS_USER@$RUNPODS_HOST:/workspace/date-finetuning/data/"
    fi
    
    print_success "Training data uploaded"
else
    print_error "Training data directory not found: $TRAINING_DATA_DIR"
    exit 1
fi

# Make scripts executable
print_status "Making scripts executable..."
ssh -i "$SSH_KEY_PATH" -p "$RUNPODS_PORT" "$RUNPODS_USER@$RUNPODS_HOST" "
    chmod +x /workspace/date-finetuning/setup_runpods.sh
    chmod +x /workspace/date-finetuning/fine_tune_model.py
    chmod +x /workspace/date-finetuning/test_model.py
"

print_success "✅ All files uploaded successfully!"
print_status "📁 Files uploaded to: /workspace/date-finetuning/"

# Show what was uploaded
print_status "📋 Uploaded files:"
ssh -i "$SSH_KEY_PATH" -p "$RUNPODS_PORT" "$RUNPODS_USER@$RUNPODS_HOST" "
    echo 'Scripts:'
    ls -la /workspace/date-finetuning/*.{sh,py} 2>/dev/null || echo 'No scripts found'
    echo ''
    echo 'Training Data:'
    ls -la /workspace/date-finetuning/data/ 2>/dev/null || echo 'No data files found'
    echo ''
    echo 'Total training examples:'
    wc -l /workspace/date-finetuning/data/*.jsonl 2>/dev/null || echo 'Could not count lines'
"

# Ask user if they want to run setup automatically
echo ""
read -p "Do you want to run the setup script now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Running setup script on RunPods..."
    ssh -i "$SSH_KEY_PATH" -p "$RUNPODS_PORT" "$RUNPODS_USER@$RUNPODS_HOST" "cd /workspace/date-finetuning && ./setup_runpods.sh"
    print_success "Setup completed!"
    
    echo ""
    read -p "Do you want to start training now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Starting training process..."
        print_warning "This will take a while. You can disconnect and the training will continue."
        print_status "Training will run in tmux session 'training' so you can reconnect later."
        
        # Run training in tmux session
        ssh -i "$SSH_KEY_PATH" -p "$RUNPODS_PORT" "$RUNPODS_USER@$RUNPODS_HOST" "
            cd /workspace/date-finetuning && 
            tmux new-session -d -s training 'source venv/bin/activate && python fine_tune_model.py; exec bash' &&
            echo 'Training started in tmux session: training' &&
            echo 'To monitor: tmux attach -t training' &&
            echo 'To detach: Ctrl+B then D'
        "
        
        print_success "Training started in background!"
        print_status "To monitor training, SSH in and run: tmux attach -t training"
    else
        print_status "Training not started. To start manually, run:"
        echo "ssh -i $SSH_KEY_PATH -p $RUNPODS_PORT $RUNPODS_USER@$RUNPODS_HOST"
        echo "cd /workspace/date-finetuning"
        echo "source venv/bin/activate"
        echo "python fine_tune_model.py"
    fi
else
    print_status "Setup not run. To run manually, execute:"
    echo "ssh -i $SSH_KEY_PATH -p $RUNPODS_PORT $RUNPODS_USER@$RUNPODS_HOST"
    echo "cd /workspace/date-finetuning"
    echo "./setup_runpods.sh"
fi

print_success "🎉 RunPods deployment process completed!"

# Show connection info
echo ""
print_status "📋 Connection Information:"
echo "SSH Command: ssh -i $SSH_KEY_PATH -p $RUNPODS_PORT $RUNPODS_USER@$RUNPODS_HOST"
echo "Work Directory: /workspace/date-finetuning"
echo "Activate Environment: source /workspace/date-finetuning/venv/bin/activate"
echo "Start Training: python fine_tune_model.py"
echo "Monitor Logs: tail -f /workspace/date-finetuning/logs/training.log"
echo "Attach to Training: tmux attach -t training"
echo ""
print_status "🔥 Happy fine-tuning!"

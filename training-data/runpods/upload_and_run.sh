#!/bin/bash

# RunPods Upload and Execute Script
# This script uploads training data and scripts to RunPods and executes the training

set -e

# Configuration
RUNPODS_IP=""  # Set this to your RunPods instance IP
RUNPODS_USER="root"  # Default RunPods user
SSH_KEY_PATH="$HOME/.ssh/id_rsa"  # Path to your SSH private key

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

# Check if RunPods IP is set
if [ -z "$RUNPODS_IP" ]; then
    print_error "Please set RUNPODS_IP variable in this script"
    print_warning "Example: RUNPODS_IP=\"123.456.789.012\""
    exit 1
fi

# Check if SSH key exists
if [ ! -f "$SSH_KEY_PATH" ]; then
    print_error "SSH key not found at: $SSH_KEY_PATH"
    print_warning "Make sure you have your SSH key configured for RunPods"
    exit 1
fi

print_status "🚀 Starting RunPods deployment process..."
print_status "Target: $RUNPODS_USER@$RUNPODS_IP"

# Test SSH connection
print_status "Testing SSH connection..."
if ssh -i "$SSH_KEY_PATH" -o ConnectTimeout=10 -o BatchMode=yes "$RUNPODS_USER@$RUNPODS_IP" exit 2>/dev/null; then
    print_success "SSH connection successful"
else
    print_error "Cannot connect to RunPods instance"
    print_warning "Check your IP address and SSH key configuration"
    exit 1
fi

# Create remote directory structure
print_status "Creating remote directory structure..."
ssh -i "$SSH_KEY_PATH" "$RUNPODS_USER@$RUNPODS_IP" "
    mkdir -p /workspace/date-finetuning/data
    mkdir -p /workspace/date-finetuning/scripts
    mkdir -p /workspace/date-finetuning/logs
    mkdir -p /workspace/date-finetuning/checkpoints
"

# Upload setup script
print_status "Uploading setup script..."
scp -i "$SSH_KEY_PATH" "./setup_runpods.sh" "$RUNPODS_USER@$RUNPODS_IP:/workspace/date-finetuning/setup_runpods.sh"

# Upload fine-tuning script
print_status "Uploading fine-tuning script..."
scp -i "$SSH_KEY_PATH" "./fine_tune_model.py" "$RUNPODS_USER@$RUNPODS_IP:/workspace/date-finetuning/fine_tune_model.py"

# Upload requirements file
print_status "Uploading requirements..."
scp -i "$SSH_KEY_PATH" "./requirements.txt" "$RUNPODS_USER@$RUNPODS_IP:/workspace/date-finetuning/requirements.txt"

# Upload training data
print_status "Uploading training data..."
TRAINING_DATA_DIR="../relative-date-training"

if [ -d "$TRAINING_DATA_DIR" ]; then
    # Upload all JSONL files
    scp -i "$SSH_KEY_PATH" "$TRAINING_DATA_DIR"/*.jsonl "$RUNPODS_USER@$RUNPODS_IP:/workspace/date-finetuning/data/"
    print_success "Training data uploaded"
else
    print_error "Training data directory not found: $TRAINING_DATA_DIR"
    exit 1
fi

# Make scripts executable
print_status "Making scripts executable..."
ssh -i "$SSH_KEY_PATH" "$RUNPODS_USER@$RUNPODS_IP" "
    chmod +x /workspace/date-finetuning/setup_runpods.sh
    chmod +x /workspace/date-finetuning/fine_tune_model.py
"

print_success "✅ All files uploaded successfully!"
print_status "📁 Files uploaded to: /workspace/date-finetuning/"

# Ask user if they want to run setup automatically
echo ""
read -p "Do you want to run the setup script now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Running setup script on RunPods..."
    ssh -i "$SSH_KEY_PATH" "$RUNPODS_USER@$RUNPODS_IP" "cd /workspace/date-finetuning && ./setup_runpods.sh"
    print_success "Setup completed!"
    
    echo ""
    read -p "Do you want to start training now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Starting training process..."
        print_warning "This will take a while. You can disconnect and the training will continue."
        ssh -i "$SSH_KEY_PATH" "$RUNPODS_USER@$RUNPODS_IP" "cd /workspace/date-finetuning && source venv/bin/activate && python fine_tune_model.py"
    else
        print_status "Training not started. To start manually, run:"
        echo "ssh -i $SSH_KEY_PATH $RUNPODS_USER@$RUNPODS_IP"
        echo "cd /workspace/date-finetuning"
        echo "source venv/bin/activate"
        echo "python fine_tune_model.py"
    fi
else
    print_status "Setup not run. To run manually, execute:"
    echo "ssh -i $SSH_KEY_PATH $RUNPODS_USER@$RUNPODS_IP"
    echo "cd /workspace/date-finetuning"
    echo "./setup_runpods.sh"
fi

print_success "🎉 RunPods deployment process completed!"

# Show connection info
echo ""
print_status "📋 Connection Information:"
echo "SSH Command: ssh -i $SSH_KEY_PATH $RUNPODS_USER@$RUNPODS_IP"
echo "Work Directory: /workspace/date-finetuning"
echo "Activate Environment: source /workspace/date-finetuning/venv/bin/activate"
echo "Start Training: python fine_tune_model.py"
echo "Monitor Logs: tail -f /workspace/date-finetuning/logs/training.log"

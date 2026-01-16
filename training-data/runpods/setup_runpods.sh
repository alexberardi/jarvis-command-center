#!/bin/bash

# RunPods Fine-tuning Setup Script
# This script sets up the environment for fine-tuning on RunPods

set -e  # Exit on any error

echo "🚀 Setting up RunPods environment for date parsing fine-tuning..."

# Update system and install dependencies
echo "📦 Installing system dependencies..."
apt-get update -y
apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    htop \
    tmux \
    vim

# Create working directory
WORK_DIR="/workspace/date-finetuning"
echo "📁 Creating work directory: $WORK_DIR"
mkdir -p $WORK_DIR
cd $WORK_DIR

# Create Python virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📚 Installing Python packages..."
pip install --upgrade pip
pip install \
    torch \
    transformers \
    datasets \
    accelerate \
    wandb \
    numpy \
    pandas \
    tqdm \
    scikit-learn \
    jsonlines

# Install additional ML tools
pip install \
    peft \
    bitsandbytes \
    scipy

echo "✅ Environment setup complete!"
echo "📍 Working directory: $WORK_DIR"
echo "🔧 Virtual environment activated at: $WORK_DIR/venv"

# Create directories for training
mkdir -p data
mkdir -p models
mkdir -p logs
mkdir -p checkpoints

echo "🎯 Ready for training data upload!"
echo ""
echo "Next steps:"
echo "1. Upload your training data to: $WORK_DIR/data/"
echo "2. Run the fine-tuning script: python fine_tune_model.py"
echo ""
echo "Current directory structure:"
ls -la

# Keep environment activated for subsequent commands
echo "source $WORK_DIR/venv/bin/activate" >> ~/.bashrc

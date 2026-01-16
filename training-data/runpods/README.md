# RunPods Fine-tuning Setup for Date Parsing

This directory contains a complete, repeatable workflow for fine-tuning a language model on RunPods using the generated date parsing training data.

## 🚀 Quick Start

### 1. Start a RunPods Instance
1. Go to [RunPods.io](https://runpods.io)
2. Create a new pod with:
   - **Template**: PyTorch 2.0+ (recommended)
   - **GPU**: Any CUDA-capable GPU (RTX 4090, A100, etc.)
   - **Storage**: At least 20GB
   - **SSH**: Enable SSH access

### 2. Get Your RunPods Connection Info
After your pod starts, note:
- **IP Address**: Your pod's public IP
- **SSH Port**: Usually 22
- **SSH Key**: Make sure your SSH key is configured

### 3. Configure and Run the Upload Script
```bash
# Edit the upload script to set your RunPods IP
nano upload_and_run.sh

# Set this line with your RunPods IP:
RUNPODS_IP="your.runpods.ip.here"

# Make the script executable and run it
chmod +x upload_and_run.sh
./upload_and_run.sh
```

The script will:
- ✅ Test SSH connection
- ✅ Upload all scripts and training data
- ✅ Optionally run setup and start training

### 4. Manual Steps (if needed)

If you prefer to run steps manually:

```bash
# SSH into your RunPods instance
ssh root@your.runpods.ip

# Run the setup script
cd /workspace/date-finetuning
./setup_runpods.sh

# Start training
source venv/bin/activate
python fine_tune_model.py
```

## 📁 File Structure

```
runpods/
├── setup_runpods.sh           # Environment setup script
├── fine_tune_model.py         # Main training script
├── test_model.py              # Model testing script
├── upload_and_run.sh          # Automated deployment script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Configuration Options

### Training Configuration
Edit `fine_tune_model.py` to adjust:

```python
@dataclass
class TrainingConfig:
    model_name: str = "microsoft/DialoGPT-small"  # Base model
    max_length: int = 512                         # Token limit
    batch_size: int = 8                          # Batch size
    learning_rate: float = 5e-5                 # Learning rate
    num_epochs: int = 3                          # Training epochs
    # ... more options
```

### Model Selection
Choose your base model in `TrainingConfig.model_name`:
- `microsoft/DialoGPT-small` - Fast, lightweight (117M params)
- `microsoft/DialoGPT-medium` - Balanced (345M params)
- `gpt2` - Classic GPT-2 (124M params)
- `distilgpt2` - Smaller, faster (82M params)

## 📊 Training Data

The system automatically uses your JSONL training data files:
- `date_training_data_3anchors.jsonl` (3,625 examples)
- `date_training_data_5anchors.jsonl` (6,042 examples)  
- `date_training_data_50anchors.jsonl` (60,417 examples)

The largest file is automatically selected for training.

### Data Format
```json
{
  "input": "Current Date: 2025-05-19 22:24:00\\nTimezone: UTC\\nExpression: 'today'",
  "output": "{\\\"result\\\": {\\\"date\\\": \\\"2025-05-19T00:00:00+00:00\\\"}, \\\"result_type\\\": \\\"single\\\"}"
}
```

## 🎯 Training Process

### What Happens During Training:
1. **Environment Setup**: Installs PyTorch, Transformers, and dependencies
2. **Data Loading**: Processes JSONL files into training format
3. **Model Loading**: Downloads and prepares the base model
4. **Training**: Fine-tunes the model on date parsing examples
5. **Saving**: Saves the trained model and tokenizer

### Training Outputs:
- **Model**: Saved to `./checkpoints/date-parsing-model/`
- **Logs**: Training logs in `./logs/training.log`
- **Checkpoints**: Intermediate saves every 500 steps

### Monitoring Training:
```bash
# Watch training progress
tail -f /workspace/date-finetuning/logs/training.log

# Check GPU usage
nvidia-smi

# Monitor system resources
htop
```

## 🧪 Testing Your Model

After training completes, test the model:

```bash
# Test with predefined examples
python test_model.py

# Interactive testing
python test_model.py --interactive

# Test specific model checkpoint
python test_model.py --model-path ./checkpoints/date-parsing-model
```

## 📈 Performance Expectations

### Training Time (estimates):
- **3K examples + RTX 4090**: ~15-30 minutes
- **60K examples + RTX 4090**: ~2-4 hours
- **60K examples + A100**: ~1-2 hours

### Model Size:
- **DialoGPT-small**: ~470MB fine-tuned
- **DialoGPT-medium**: ~1.4GB fine-tuned

## 🔍 Troubleshooting

### Common Issues:

**SSH Connection Failed:**
```bash
# Check your SSH key
ls -la ~/.ssh/
# Verify RunPods IP and SSH configuration
```

**Out of Memory:**
- Reduce `batch_size` in `TrainingConfig`
- Enable `fp16=True` for mixed precision
- Use a smaller base model

**Training Stuck:**
- Check GPU availability: `nvidia-smi`
- Monitor logs: `tail -f logs/training.log`
- Verify data format in uploaded JSONL files

**Model Not Loading:**
- Check model path: `ls -la checkpoints/`
- Verify training completed successfully
- Look for error messages in logs

### Getting Help:
```bash
# Check system resources
htop
free -h
df -h

# Check GPU
nvidia-smi

# Check Python environment
source venv/bin/activate
pip list | grep torch
```

## 🎉 Next Steps

After successful training:

1. **Download Your Model**: 
   ```bash
   scp -r root@your.runpods.ip:/workspace/date-finetuning/checkpoints ./local-checkpoints
   ```

2. **Test Locally**: Use the trained model in your applications

3. **Deploy**: Integrate the model into your Jarvis system

4. **Iterate**: Adjust training parameters and retrain if needed

## 💰 Cost Optimization

### RunPods Cost Tips:
- **Use Spot Instances**: 50-80% cheaper
- **Stop When Done**: Don't forget to terminate your pod
- **Right-size GPU**: Don't over-provision
- **Batch Training**: Train multiple models in one session

### Estimated Costs:
- **RTX 4090 Spot**: ~$0.30/hour
- **A100 Spot**: ~$1.20/hour
- **Training 60K examples**: ~$2-8 total cost

## 🔄 Continuous Improvement

### For Better Results:
1. **More Data**: Generate additional training examples
2. **Data Quality**: Review and clean training data
3. **Hyperparameter Tuning**: Experiment with learning rates, batch sizes
4. **Model Architecture**: Try different base models
5. **Evaluation**: Create comprehensive test sets

---

**Happy Fine-tuning! 🎯**

For questions or issues, check the logs first, then review this README.

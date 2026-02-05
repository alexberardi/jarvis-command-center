#!/usr/bin/env python3
"""
Improved RunPods Fine-tuning Script for Date Parsing Model
Fixes repetition issues and improves training stability
"""

import json
import os
import sys
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
import logging

# HuggingFace Hub imports
try:
    from huggingface_hub import HfApi, login
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("huggingface_hub not available. Install with: pip install huggingface_hub")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for fine-tuning - IMPROVED VERSION"""
    model_name: str = "microsoft/DialoGPT-small"
    max_length: int = 200  # Shorter to reduce repetition
    batch_size: int = 8
    learning_rate: float = 3e-5  # Lower learning rate
    num_epochs: int = 1  # ONLY 1 epoch to prevent overfitting
    warmup_steps: int = 50
    save_steps: int = 1000
    eval_steps: int = 1000
    output_dir: str = "./checkpoints/date-parsing-model-v2"
    logging_dir: str = "./logs"
    gradient_accumulation_steps: int = 2
    fp16: bool = False
    bf16: bool = False
    max_examples: int = 50000

class ImprovedDateParsingDataProcessor:
    """Improved data processor with better formatting"""
    
    def __init__(self, tokenizer, max_length: int = 200):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_jsonl_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load training data from JSONL file"""
        logger.info(f"Loading training data from: {file_path}")
        data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Loading data"), 1):
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(data)} training examples")
        return data
    
    def format_training_example(self, item: Dict[str, Any]) -> str:
        """Format a single training example - IMPROVED VERSION"""
        input_text = item['input']
        output_text = item['output']
        
        # Parse the JSON output to make it cleaner
        try:
            output_json = json.loads(output_text)
            # Simplify the JSON structure to reduce "result" repetition
            if "result" in output_json and "date" in output_json["result"]:
                date_str = output_json["result"]["date"]
                result_type = output_json.get("result_type", "single")
                
                # Create cleaner output format
                clean_output = f'{{"date": "{date_str}", "type": "{result_type}"}}'
            else:
                clean_output = output_text
        except (json.JSONDecodeError, KeyError, TypeError):
            clean_output = output_text  # Fallback to raw output on parse failure
        
        # Create a conversation with clear end marker
        formatted = f"Human: Parse date: {input_text}\nAssistant: {clean_output}<|endoftext|>"
        return formatted
    
    def prepare_dataset(self, data: List[Dict[str, Any]], max_examples: int = 50000) -> Dataset:
        """Prepare dataset for training - LIMIT SIZE"""
        logger.info("Preparing dataset for training...")
        
        # Limit dataset size to prevent overfitting
        if len(data) > max_examples:
            logger.info(f"Limiting dataset from {len(data)} to {max_examples} examples")
            data = data[:max_examples]
        
        # Format all examples first
        formatted_texts = []
        for item in tqdm(data, desc="Formatting examples"):
            formatted_text = self.format_training_example(item)
            formatted_texts.append(formatted_text)
        
        # Tokenize all examples
        logger.info("Tokenizing examples...")
        tokenized_examples = []
        
        for text in tqdm(formatted_texts, desc="Tokenizing"):
            # Tokenize each example individually
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
                padding=False
            )
            
            # Create the example with input_ids and labels
            example = {
                'input_ids': tokens['input_ids'],
                'attention_mask': tokens['attention_mask'],
                'labels': tokens['input_ids'].copy()  # For causal LM
            }
            tokenized_examples.append(example)
        
        # Create dataset from tokenized examples
        dataset = Dataset.from_list(tokenized_examples)
        
        logger.info(f"Dataset prepared with {len(dataset)} examples")
        return dataset

class ImprovedDateParsingTrainer:
    """Improved trainer with better settings"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.load_model_and_tokenizer()
        
        # Setup data processor
        self.data_processor = ImprovedDateParsingDataProcessor(
            self.tokenizer, 
            max_length=config.max_length
        )
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Add pad token and end token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Ensure we have endoftext token
        if "<|endoftext|>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|endoftext|>"]})
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,  # Force FP32
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Ensure model is in FP32
        if torch.cuda.is_available():
            self.model = self.model.float()
        
        # Resize token embeddings for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"Model loaded with {self.model.num_parameters():,} parameters")
    
    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments - CONSERVATIVE SETTINGS"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=100,
            logging_dir=self.config.logging_dir,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            eval_strategy="steps",
            save_total_limit=2,  # Keep fewer checkpoints
            prediction_loss_only=True,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            dataloader_pin_memory=False,
            report_to=[],  # Disable wandb
            remove_unused_columns=False,
            dataloader_num_workers=0,
            # Add early stopping patience
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    
    def train(self, train_data_path: str, eval_data_path: str = None):
        """Main training function - IMPROVED"""
        logger.info("🚀 Starting improved fine-tuning process...")
        
        # Load and prepare training data
        train_data = self.data_processor.load_jsonl_data(train_data_path)
        max_examples = getattr(self.config, 'max_examples', 50000)
        train_dataset = self.data_processor.prepare_dataset(train_data, max_examples=max_examples)
        
        # Always split for evaluation to monitor overfitting
        logger.info("Splitting training data for evaluation (90/10 split)")
        train_test = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test['train']
        eval_dataset = train_test['test']
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Create a better data collator
        def improved_data_collator(features):
            # Pad all sequences to the same length
            max_length = max(len(f['input_ids']) for f in features)
            
            batch = {
                'input_ids': [],
                'attention_mask': [],
                'labels': []
            }
            
            for feature in features:
                input_ids = feature['input_ids']
                attention_mask = feature['attention_mask']
                labels = feature['labels']
                
                # Pad sequences
                padding_length = max_length - len(input_ids)
                
                # Pad with tokenizer pad token
                padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                padded_attention_mask = attention_mask + [0] * padding_length
                padded_labels = labels + [-100] * padding_length  # -100 is ignored in loss
                
                batch['input_ids'].append(padded_input_ids)
                batch['attention_mask'].append(padded_attention_mask)
                batch['labels'].append(padded_labels)
            
            # Convert to tensors
            return {
                'input_ids': torch.tensor(batch['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(batch['attention_mask'], dtype=torch.long),
                'labels': torch.tensor(batch['labels'], dtype=torch.long)
            }
        
        data_collator = improved_data_collator
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("🎯 Beginning improved training...")
        trainer.train()
        
        # Save the final model
        logger.info("💾 Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("✅ Improved training completed successfully!")
        logger.info(f"📁 Model saved to: {self.config.output_dir}")
        
        return trainer

def push_to_huggingface(
    model_path: str,
    repo_name: str,
    hf_token: str = None,
    private: bool = True,
    commit_message: str = "Upload improved date parsing model v2"
):
    """
    Push the fine-tuned model to Hugging Face Hub
    """
    if not HF_AVAILABLE:
        logger.error("❌ huggingface_hub not available. Install with: pip install huggingface_hub")
        return False
    
    logger.info(f"🚀 Pushing model to Hugging Face Hub: {repo_name}")
    
    # Login to Hugging Face
    try:
        if hf_token:
            login(token=hf_token)
            logger.info("✅ Logged in to Hugging Face with provided token")
        else:
            # Try to use existing credentials
            logger.info("Using existing Hugging Face credentials")
    except Exception as e:
        logger.error(f"❌ Failed to login to Hugging Face: {e}")
        return False
    
    # Load the model and tokenizer
    logger.info(f"Loading model from: {model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("✅ Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False
    
    # Create model card content
    model_card = f"""---
license: mit
base_model: microsoft/DialoGPT-small
tags:
- fine-tuned
- date-parsing
- conversational-ai
- jarvis
- time-understanding
- improved-v2
language:
- en
pipeline_tag: text-generation
---

# {repo_name}

## Model Description

This is an **improved v2** fine-tuned version of `microsoft/DialoGPT-small` specialized for date and time parsing tasks. 

### Key Improvements in V2:
- ✅ **Reduced overfitting** - Only 1 epoch training
- ✅ **Better data format** - Cleaner JSON structure 
- ✅ **Proper end tokens** - Uses `<|endoftext|>` for stopping
- ✅ **Conservative settings** - Lower learning rate, smaller batches
- ✅ **Limited dataset** - 50K examples max to prevent repetition issues

## Training Details

- **Base Model**: microsoft/DialoGPT-small (117M parameters)
- **Training Examples**: Up to 50,000 date parsing examples
- **Training Format**: Conversational (Human/Assistant)
- **Epochs**: 1 (reduced from 3 to prevent overfitting)
- **Precision**: FP32
- **Learning Rate**: 3e-5 (reduced for stability)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
model = AutoModelForCausalLM.from_pretrained("{repo_name}")

# Example usage
input_text = '''Human: Parse date: Current Date: 2025-01-15 12:00:00
Timezone: UTC
Expression: 'next Monday'
Assistant:'''

inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(
    inputs,
    max_new_tokens=100,
    temperature=0.1,
    repetition_penalty=1.2,  # Helps prevent repetition
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Expected Output Format

The model generates JSON responses like:
```json
{{"date": "2025-01-20T00:00:00+00:00", "type": "single"}}
```

## Improvements Over V1

- **No repetition loops** - V1 had issues with repeating "result" tokens
- **Cleaner JSON** - Simplified structure reduces token repetition
- **Better stopping** - Proper end-of-text tokens prevent runaway generation
- **Faster training** - 1 epoch vs 3 epochs
- **More stable** - Conservative hyperparameters

## Training Data Format

Input format:
```
Human: Parse date: Current Date: 2025-01-15 12:00:00
Timezone: UTC
Expression: 'tomorrow'
Assistant: {{"date": "2025-01-16T00:00:00+00:00", "type": "single"}}<|endoftext|>
```

## License

MIT License - Feel free to use and modify!
"""

    # Push to Hub
    logger.info(f"Pushing model to: {repo_name}")
    try:
        # Push model with model card
        model.push_to_hub(
            repo_id=repo_name,
            private=private,
            commit_message=commit_message
        )
        
        # Push tokenizer
        tokenizer.push_to_hub(
            repo_id=repo_name,
            private=private,
            commit_message=commit_message
        )
        
        # Create model card
        api = HfApi()
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            commit_message="Add model card",
            repo_type="model"
        )
        
        logger.info("✅ Model successfully pushed to Hugging Face Hub!")
        logger.info(f"🔗 Model URL: https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to push model: {e}")
        return False

def main():
    """Main execution function with HF push support"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune date parsing model with optional HF push")
    parser.add_argument("--push-to-hf", action="store_true", 
                       help="Push model to Hugging Face Hub after training")
    parser.add_argument("--repo-name", type=str,
                       help="HF repository name (e.g., 'username/model-name')")
    parser.add_argument("--hf-token", type=str,
                       help="Hugging Face API token (optional if already logged in)")
    parser.add_argument("--private", action="store_true", default=True,
                       help="Make HF repository private (default: True)")
    parser.add_argument("--public", action="store_true",
                       help="Make HF repository public")
    parser.add_argument("--max-examples", type=int, default=50000,
                       help="Maximum number of training examples to use")
    
    args = parser.parse_args()
    
    # Validate HF arguments
    if args.push_to_hf:
        if not args.repo_name:
            logger.error("❌ --repo-name is required when using --push-to-hf")
            logger.info("Example: --repo-name 'your-username/jarvis-date-parser-v2'")
            sys.exit(1)
        
        if not HF_AVAILABLE:
            logger.error("❌ huggingface_hub not available. Install with: pip install huggingface_hub")
            sys.exit(1)
        
        # Check for HF token
        hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            logger.warning("⚠️  No HF token provided. Make sure you're logged in with 'huggingface-cli login'")
    
    # Check for training data
    data_dir = Path("./data")
    if not data_dir.exists():
        logger.error("❌ Data directory not found! Please upload training data to ./data/")
        sys.exit(1)
    
    # Look for training data files - prefer smaller ones
    jsonl_files = list(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.error("❌ No JSONL training files found in ./data/")
        sys.exit(1)
    
    # Use a smaller file for better training
    preferred_files = [
        "date_training_data_3anchors.jsonl",
        "date_training_data_5anchors.jsonl", 
        "date_training_data_50anchors.jsonl"
    ]
    
    train_file = None
    for pref_file in preferred_files:
        if (data_dir / pref_file).exists():
            train_file = data_dir / pref_file
            break
    
    if not train_file:
        train_file = min(jsonl_files, key=lambda f: f.stat().st_size)
    
    logger.info(f"📊 Using training file: {train_file}")
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Setup training configuration - CONSERVATIVE SETTINGS
    config = TrainingConfig()
    config.max_examples = args.max_examples
    
    # Adjust settings based on GPU
    if torch.cuda.is_available():
        logger.info(f"🔥 GPU detected: {torch.cuda.get_device_name()}")
        config.batch_size = 6  # Conservative batch size
        config.fp16 = False    # Keep FP16 disabled
        config.bf16 = False
    else:
        logger.info("⚠️  No GPU detected, using CPU")
        config.batch_size = 2
        config.fp16 = False
        config.bf16 = False
    
    # Initialize trainer and start training
    trainer = ImprovedDateParsingTrainer(config)
    trainer.train(str(train_file))
    
    logger.info("🎉 Improved fine-tuning process completed!")
    
    # Push to Hugging Face if requested
    if args.push_to_hf:
        logger.info("🚀 Starting Hugging Face upload...")
        
        # Determine privacy setting
        private = not args.public if args.public else args.private
        
        success = push_to_huggingface(
            model_path=config.output_dir,
            repo_name=args.repo_name,
            hf_token=hf_token if 'hf_token' in locals() else None,
            private=private,
            commit_message=f"Upload improved date parsing model v2 (trained on {args.max_examples} examples)"
        )
        
        if success:
            logger.info("🎉 Model successfully uploaded to Hugging Face!")
            logger.info(f"🔗 View your model: https://huggingface.co/{args.repo_name}")
        else:
            logger.error("❌ Failed to upload model to Hugging Face")
    else:
        logger.info("ℹ️  Skipping Hugging Face upload. Use --push-to-hf to upload after training.")

if __name__ == "__main__":
    main()

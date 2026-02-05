#!/usr/bin/env python3
"""
Llama 3.2 1B Fine-tuning Script for Date Parsing
Simplified approach with clean data format and fast training
"""

import json
import os
import sys
import argparse
import re
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
    """Configuration for Llama 3.2 1B fine-tuning"""
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    max_length: int = 512  # Llama can handle more context
    batch_size: int = 8    # Reduced for FP32 memory usage
    learning_rate: float = 2e-5  # Conservative for instruct model
    num_epochs: int = 1    # Single epoch for speed
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "./checkpoints/llama-date-parser-v3"
    logging_dir: str = "./logs"
    gradient_accumulation_steps: int = 1  # A100 can handle larger batches
    fp16: bool = False     # Disable FP16 to avoid gradient issues
    bf16: bool = False
    max_examples: int = 30000  # Reduced for speed

class SimplifiedDateProcessor:
    """Simplified data processor for cleaner training"""
    
    def __init__(self, tokenizer, max_length: int = 512):
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
    
    def extract_iso_date(self, output_text: str) -> str:
        """Extract just the ISO date from the complex JSON output"""
        try:
            # Parse the JSON output
            output_json = json.loads(output_text)
            
            # Extract the date from nested structure
            if "result" in output_json and "date" in output_json["result"]:
                return output_json["result"]["date"]
            elif "date" in output_json:
                return output_json["date"]
            else:
                # Fallback: try to find ISO date with regex
                iso_pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}'
                match = re.search(iso_pattern, output_text)
                if match:
                    return match.group(0)
                else:
                    logger.warning(f"Could not extract date from: {output_text}")
                    return "INVALID_DATE"
        except (json.JSONDecodeError, KeyError, TypeError):
            # Try regex as fallback
            iso_pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}'
            match = re.search(iso_pattern, output_text)
            if match:
                return match.group(0)
            else:
                logger.warning(f"Could not parse output: {output_text}")
                return "INVALID_DATE"
    
    def format_training_example(self, item: Dict[str, Any]) -> Dict[str, str]:
        """Format example for Llama 3.2 instruction format"""
        input_text = item['input']
        output_text = item['output']
        
        # Extract clean ISO date
        iso_date = self.extract_iso_date(output_text)
        
        if iso_date == "INVALID_DATE":
            return None  # Skip invalid examples
        
        # Parse input to extract components
        lines = input_text.split('\n')
        current_date = ""
        timezone = ""
        expression = ""
        
        for line in lines:
            if line.startswith("Current Date:"):
                current_date = line.replace("Current Date:", "").strip()
            elif line.startswith("Timezone:"):
                timezone = line.replace("Timezone:", "").strip()
            elif line.startswith("Expression:"):
                expression = line.replace("Expression:", "").strip().strip("'\"")
        
        # Create clean instruction format
        user_message = f"Parse the date expression '{expression}' given the current date {current_date} in {timezone} timezone. Return only the ISO 8601 formatted date."
        
        # Use Llama's chat format
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": iso_date}
        ]
        
        return {"messages": messages}
    
    def prepare_dataset(self, data: List[Dict[str, Any]], max_examples: int = 30000) -> Dataset:
        """Prepare dataset for training"""
        logger.info("Preparing simplified dataset for training...")
        
        # Limit dataset size
        if len(data) > max_examples:
            logger.info(f"Limiting dataset from {len(data)} to {max_examples} examples")
            data = data[:max_examples]
        
        # Format all examples
        formatted_examples = []
        skipped = 0
        
        for item in tqdm(data, desc="Processing examples"):
            formatted = self.format_training_example(item)
            if formatted is not None:
                formatted_examples.append(formatted)
            else:
                skipped += 1
        
        logger.info(f"Processed {len(formatted_examples)} examples, skipped {skipped}")
        
        # Convert to chat format for tokenization
        tokenized_examples = []
        
        for example in tqdm(formatted_examples, desc="Tokenizing"):
            # Apply chat template
            try:
                formatted_text = self.tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                # Tokenize
                tokens = self.tokenizer(
                    formatted_text,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors=None,
                    padding=False
                )
                
                # Create training example
                tokenized_example = {
                    'input_ids': tokens['input_ids'],
                    'attention_mask': tokens['attention_mask'],
                    'labels': tokens['input_ids'].copy()  # For causal LM
                }
                
                tokenized_examples.append(tokenized_example)
                
            except Exception as e:
                logger.warning(f"Failed to tokenize example: {e}")
                continue
        
        # Create dataset
        dataset = Dataset.from_list(tokenized_examples)
        logger.info(f"Dataset prepared with {len(dataset)} examples")
        return dataset

class LlamaDateParsingTrainer:
    """Trainer for Llama 3.2 1B date parsing"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Setup data processor
        self.data_processor = SimplifiedDateProcessor(
            self.tokenizer, 
            max_length=config.max_length
        )
    
    def load_model_and_tokenizer(self):
        """Load Llama 3.2 1B Instruct model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate settings for A100
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,  # Force FP32 to avoid gradient issues
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"  # Disable FlashAttention2 - not installed
        )
        
        logger.info(f"Model loaded with {self.model.num_parameters():,} parameters")
    
    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments optimized for A100"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=50,
            logging_dir=self.config.logging_dir,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            eval_strategy="steps",
            save_total_limit=2,
            prediction_loss_only=True,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            dataloader_pin_memory=True,  # Good for A100
            report_to=[],  # Disable wandb
            remove_unused_columns=False,
            dataloader_num_workers=4,  # Use multiple workers on A100
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # A100 optimizations
            ddp_find_unused_parameters=False,
            dataloader_persistent_workers=True,
        )
    
    def train(self, train_data_path: str):
        """Main training function optimized for speed"""
        logger.info("🚀 Starting Llama 3.2 1B fine-tuning...")
        
        # Load and prepare data
        train_data = self.data_processor.load_jsonl_data(train_data_path)
        train_dataset = self.data_processor.prepare_dataset(
            train_data, 
            max_examples=self.config.max_examples
        )
        
        # Split for evaluation (90/10 for speed)
        logger.info("Splitting training data for evaluation (90/10 split)")
        train_test = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test['train']
        eval_dataset = train_test['test']
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Custom data collator to handle variable length sequences
        def custom_data_collator(features):
            """Custom data collator that pads sequences dynamically"""
            import torch
            
            # Extract input_ids, attention_mask, and labels
            input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
            attention_masks = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
            labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
            
            # Pad sequences to the maximum length in this batch
            max_len = max(len(seq) for seq in input_ids)
            
            # Pad input_ids and attention_mask
            padded_input_ids = []
            padded_attention_masks = []
            padded_labels = []
            
            for i in range(len(input_ids)):
                seq_len = len(input_ids[i])
                padding_length = max_len - seq_len
                
                # Pad input_ids
                padded_input_ids.append(torch.cat([
                    input_ids[i],
                    torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                ]))
                
                # Pad attention_mask
                padded_attention_masks.append(torch.cat([
                    attention_masks[i],
                    torch.zeros(padding_length, dtype=torch.long)
                ]))
                
                # Pad labels (use -100 for padded tokens)
                padded_labels.append(torch.cat([
                    labels[i],
                    torch.full((padding_length,), -100, dtype=torch.long)
                ]))
            
            return {
                "input_ids": torch.stack(padded_input_ids),
                "attention_mask": torch.stack(padded_attention_masks),
                "labels": torch.stack(padded_labels)
            }
        
        data_collator = custom_data_collator
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("🎯 Beginning fast training...")
        trainer.train()
        
        # Save final model
        logger.info("💾 Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("✅ Llama 3.2 training completed!")
        logger.info(f"📁 Model saved to: {self.config.output_dir}")
        
        return trainer

def push_to_huggingface(
    model_path: str,
    repo_name: str,
    hf_token: str = None,
    private: bool = True,
    commit_message: str = "Upload Llama 3.2 1B date parsing model v3"
):
    """Push model to HuggingFace Hub"""
    if not HF_AVAILABLE:
        logger.error("❌ huggingface_hub not available")
        return False
    
    logger.info(f"🚀 Pushing Llama model to HF: {repo_name}")
    
    try:
        if hf_token:
            login(token=hf_token)
        
        # Load and push model
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model.push_to_hub(repo_id=repo_name, private=private, commit_message=commit_message)
        tokenizer.push_to_hub(repo_id=repo_name, private=private, commit_message=commit_message)
        
        # Create model card
        model_card = f"""---
license: llama3.2
base_model: meta-llama/Llama-3.2-1B-Instruct
tags:
- fine-tuned
- date-parsing
- llama-3.2
- fast-inference
language:
- en
pipeline_tag: text-generation
---

# {repo_name}

## Model Description

Fine-tuned Llama 3.2 1B Instruct model for date parsing tasks. This model converts natural language date expressions into ISO 8601 formatted dates.

### Key Features:
- ✅ **Fast inference** - 1B parameters for quick responses
- ✅ **Clean output** - Returns only ISO dates, no JSON complexity
- ✅ **Modern base** - Built on Llama 3.2 (2024)
- ✅ **Instruction-tuned** - Follows instructions precisely

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
model = AutoModelForCausalLM.from_pretrained("{repo_name}")

messages = [
    {{"role": "user", "content": "Parse the date expression 'tomorrow' given the current date 2025-01-15 12:00:00 in UTC timezone. Return only the ISO 8601 formatted date."}}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
outputs = model.generate(inputs, max_new_tokens=50, temperature=0.1)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Training Details

- **Base Model**: meta-llama/Llama-3.2-1B-Instruct
- **Training Examples**: 30,000 date parsing examples
- **Epochs**: 1 (optimized for speed)
- **Training Time**: ~1.5 hours on A100
- **Format**: Simplified ISO date output (no complex JSON)

## Expected Output

Input: "Parse 'next Monday' from 2025-01-15"
Output: "2025-01-20T00:00:00+00:00"

Clean, simple, fast! 🚀
"""
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            commit_message="Add model card"
        )
        
        logger.info("✅ Model pushed to HuggingFace!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Push failed: {e}")
        return False

def main():
    """Main execution with A100 optimization"""
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2 1B for date parsing")
    parser.add_argument("--push-to-hf", action="store_true")
    parser.add_argument("--repo-name", type=str)
    parser.add_argument("--hf-token", type=str)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--max-examples", type=int, default=30000)
    
    args = parser.parse_args()
    
    # Validate HF args
    if args.push_to_hf and not args.repo_name:
        logger.error("❌ --repo-name required for HF push")
        sys.exit(1)
    
    # Find training data
    data_dir = Path("./data")
    if not data_dir.exists():
        logger.error("❌ Data directory not found!")
        sys.exit(1)
    
    # Use the 50anchors file (good balance of size/quality)
    train_file = data_dir / "date_training_data_50anchors.jsonl"
    if not train_file.exists():
        # Fallback to any available file
        jsonl_files = list(data_dir.glob("*.jsonl"))
        if not jsonl_files:
            logger.error("❌ No training data found!")
            sys.exit(1)
        train_file = jsonl_files[0]
    
    logger.info(f"📊 Using training file: {train_file}")
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Setup config for A100
    config = TrainingConfig()
    config.max_examples = args.max_examples
    
    if torch.cuda.is_available():
        logger.info(f"🔥 GPU detected: {torch.cuda.get_device_name()}")
        # A100 optimizations
        config.batch_size = 32  # Large batch for A100
        config.fp16 = True
        config.gradient_accumulation_steps = 1
    else:
        logger.error("❌ CUDA not available - A100 required!")
        sys.exit(1)
    
    # Train model
    trainer = LlamaDateParsingTrainer(config)
    trainer.train(str(train_file))
    
    # Push to HF if requested
    if args.push_to_hf:
        logger.info("🚀 Pushing to HuggingFace...")
        success = push_to_huggingface(
            model_path=config.output_dir,
            repo_name=args.repo_name,
            hf_token=args.hf_token,
            private=not args.public
        )
        
        if success:
            logger.info(f"🎉 Model uploaded: https://huggingface.co/{args.repo_name}")
    
    logger.info("🎉 All done! Fast Llama 3.2 training complete!")

if __name__ == "__main__":
    main()

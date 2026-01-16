#!/usr/bin/env python3
"""
RunPods Fine-tuning Script for Date Parsing Model
Processes JSONL training data and fine-tunes a language model for date parsing tasks.
"""

import json
import os
import sys
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
    """Configuration for fine-tuning"""
    model_name: str = "microsoft/DialoGPT-small"  # Lightweight model for date parsing
    max_length: int = 256  # Reduced from 512 to avoid length issues
    batch_size: int = 4   # Reduced from 8 to be more conservative
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "./checkpoints/date-parsing-model"
    logging_dir: str = "./logs"
    gradient_accumulation_steps: int = 4  # Increased to maintain effective batch size
    fp16: bool = False  # Disable FP16 to avoid gradient scaling issues
    bf16: bool = False  # Also disable BF16 for now

class DateParsingDataProcessor:
    """Processes JSONL training data for date parsing fine-tuning"""
    
    def __init__(self, tokenizer, max_length: int = 256):
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
        """Format a single training example for fine-tuning"""
        input_text = item['input']
        output_text = item['output']
        
        # Create a conversation-style format
        formatted = f"Human: Parse this date expression:\n{input_text}\n\nAssistant: {output_text}"
        return formatted
    
    def prepare_dataset(self, data: List[Dict[str, Any]]) -> Dataset:
        """Prepare dataset for training"""
        logger.info("Preparing dataset for training...")
        
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
                'labels': tokens['input_ids'].copy()  # For causal LM, labels = input_ids
            }
            tokenized_examples.append(example)
        
        # Create dataset from tokenized examples
        dataset = Dataset.from_list(tokenized_examples)
        
        logger.info(f"Dataset prepared with {len(dataset)} examples")
        return dataset

class DateParsingTrainer:
    """Main trainer class for date parsing model"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.load_model_and_tokenizer()
        
        # Setup data processor
        self.data_processor = DateParsingDataProcessor(
            self.tokenizer, 
            max_length=config.max_length
        )
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,  # Force FP32 to avoid gradient scaling issues
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Ensure model is in FP32
        if torch.cuda.is_available():
            self.model = self.model.float()  # Convert to FP32 explicitly
        
        # Resize token embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"Model loaded with {self.model.num_parameters():,} parameters")
    
    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments"""
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
            save_total_limit=3,
            prediction_loss_only=True,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            dataloader_pin_memory=False,  # Helps with memory issues
            report_to=[],  # Disable wandb completely
            remove_unused_columns=False,  # Keep all columns for debugging
            dataloader_num_workers=0,  # Disable multiprocessing to avoid issues
        )
    
    def train(self, train_data_path: str, eval_data_path: str = None):
        """Main training function"""
        logger.info("🚀 Starting fine-tuning process...")
        
        # Load and prepare training data
        train_data = self.data_processor.load_jsonl_data(train_data_path)
        train_dataset = self.data_processor.prepare_dataset(train_data)
        
        # Load evaluation data if provided
        eval_dataset = None
        if eval_data_path and os.path.exists(eval_data_path):
            eval_data = self.data_processor.load_jsonl_data(eval_data_path)
            eval_dataset = self.data_processor.prepare_dataset(eval_data)
        else:
            # Split training data for evaluation
            logger.info("Splitting training data for evaluation (80/20 split)")
            train_test = train_dataset.train_test_split(test_size=0.2, seed=42)
            train_dataset = train_test['train']
            eval_dataset = train_test['test']
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Create a custom data collator that properly handles padding
        def custom_data_collator(features):
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
                
                # Pad input_ids and attention_mask with pad_token_id
                padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                padded_attention_mask = attention_mask + [0] * padding_length
                
                # Pad labels with -100 (ignore index)
                padded_labels = labels + [-100] * padding_length
                
                batch['input_ids'].append(padded_input_ids)
                batch['attention_mask'].append(padded_attention_mask)
                batch['labels'].append(padded_labels)
            
            # Convert to tensors
            import torch
            return {
                'input_ids': torch.tensor(batch['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(batch['attention_mask'], dtype=torch.long),
                'labels': torch.tensor(batch['labels'], dtype=torch.long)
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
        logger.info("🎯 Beginning training...")
        trainer.train()
        
        # Save the final model
        logger.info("💾 Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"📁 Model saved to: {self.config.output_dir}")
        
        return trainer

def main():
    """Main execution function"""
    # Check for training data
    data_dir = Path("./data")
    if not data_dir.exists():
        logger.error("❌ Data directory not found! Please upload training data to ./data/")
        sys.exit(1)
    
    # Look for training data files
    jsonl_files = list(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.error("❌ No JSONL training files found in ./data/")
        sys.exit(1)
    
    # Use the largest training file (assuming it's the main one)
    train_file = max(jsonl_files, key=lambda f: f.stat().st_size)
    logger.info(f"📊 Using training file: {train_file}")
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Setup training configuration
    config = TrainingConfig()
    
    # Check if we have GPU
    if torch.cuda.is_available():
        logger.info(f"🔥 GPU detected: {torch.cuda.get_device_name()}")
        logger.info(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        config.batch_size = 8  # Moderate batch size for GPU with FP32
        config.fp16 = False  # Keep FP16 disabled to avoid gradient issues
        config.bf16 = False  # Keep BF16 disabled too
    else:
        logger.info("⚠️  No GPU detected, using CPU (training will be slower)")
        config.batch_size = 2  # Smaller batch size for CPU
        config.fp16 = False
        config.bf16 = False
    
    # Initialize trainer and start training
    trainer = DateParsingTrainer(config)
    trainer.train(str(train_file))
    
    logger.info("🎉 Fine-tuning process completed!")

if __name__ == "__main__":
    main()

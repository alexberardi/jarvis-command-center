#!/usr/bin/env python3
"""
Push fine-tuned model to Hugging Face Hub
"""

import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi, login
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def push_model_to_hf(
    model_path: str,
    repo_name: str,
    hf_token: str = None,
    private: bool = True,
    commit_message: str = "Upload fine-tuned date parsing model"
):
    """
    Push the fine-tuned model to Hugging Face Hub
    
    Args:
        model_path: Local path to the fine-tuned model
        repo_name: Name for the HF repository (e.g., "username/model-name")
        hf_token: Hugging Face token (optional if already logged in)
        private: Whether to make the repo private
        commit_message: Commit message for the upload
    """
    
    # Login to Hugging Face
    if hf_token:
        login(token=hf_token)
        logger.info("Logged in to Hugging Face with provided token")
    else:
        logger.info("Using existing Hugging Face credentials")
    
    # Load the model and tokenizer
    logger.info(f"Loading model from: {model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("✅ Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False
    
    # Push to Hub
    logger.info(f"Pushing model to: {repo_name}")
    try:
        # Push model
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
        
        logger.info("✅ Model successfully pushed to Hugging Face Hub!")
        logger.info(f"🔗 Model URL: https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to push model: {e}")
        return False

def create_model_card(model_path: str, repo_name: str):
    """Create a model card for the uploaded model"""
    
    model_card_content = f"""---
license: mit
base_model: microsoft/DialoGPT-small
tags:
- fine-tuned
- date-parsing
- conversational-ai
- jarvis
- time-understanding
language:
- en
pipeline_tag: text-generation
---

# {repo_name}

## Model Description

This is a fine-tuned version of `microsoft/DialoGPT-small` specialized for date and time parsing tasks. The model has been trained on a large dataset of date expressions and their corresponding structured outputs.

## Training Data

- **Base Model**: microsoft/DialoGPT-small (117M parameters)
- **Training Examples**: ~392,000 date parsing examples
- **Training Format**: Conversational (Human/Assistant)
- **Epochs**: 3
- **Precision**: FP32

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
model = AutoModelForCausalLM.from_pretrained("{repo_name}")

# Example usage
input_text = '''Human: Parse this date expression:
Current Date: 2025-01-15 12:00:00
Timezone: UTC
Expression: 'next Monday'

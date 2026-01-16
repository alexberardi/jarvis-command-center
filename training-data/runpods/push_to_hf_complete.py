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

def main():
    parser = argparse.ArgumentParser(description="Push fine-tuned model to Hugging Face Hub")
    parser.add_argument("--model-path", default="./checkpoints/date-parsing-model",
                       help="Path to the fine-tuned model")
    parser.add_argument("--repo-name", required=True,
                       help="Repository name on HF Hub (e.g., 'username/model-name')")
    parser.add_argument("--token", 
                       help="Hugging Face token (optional if already logged in)")
    parser.add_argument("--public", action="store_true",
                       help="Make the repository public (default: private)")
    parser.add_argument("--message", default="Upload fine-tuned date parsing model",
                       help="Commit message")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"❌ Model path not found: {args.model_path}")
        return
    
    # Check for required files
    required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(args.model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"⚠️  Some files might be missing: {missing_files}")
        logger.info("Continuing anyway - some files might have different names...")
    
    # Show what we're about to upload
    logger.info("📋 Upload Summary:")
    logger.info(f"  Local Path: {args.model_path}")
    logger.info(f"  Repository: {args.repo_name}")
    logger.info(f"  Private: {not args.public}")
    logger.info(f"  Message: {args.message}")
    
    # List files to upload
    logger.info("📁 Files to upload:")
    for root, dirs, files in os.walk(args.model_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, args.model_path)
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            logger.info(f"  {rel_path} ({size:.1f}MB)")
    
    # Confirm upload
    confirm = input("\n🚀 Proceed with upload? (y/n): ")
    if confirm.lower() != 'y':
        logger.info("❌ Upload cancelled")
        return
    
    # Perform upload
    success = push_model_to_hf(
        model_path=args.model_path,
        repo_name=args.repo_name,
        hf_token=args.token,
        private=not args.public,
        commit_message=args.message
    )
    
    if success:
        logger.info("🎉 Upload completed successfully!")
        logger.info(f"🔗 View your model: https://huggingface.co/{args.repo_name}")
    else:
        logger.error("❌ Upload failed")

if __name__ == "__main__":
    main()

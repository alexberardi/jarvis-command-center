#!/usr/bin/env python3
"""
Test script for the fine-tuned date parsing model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DateParsingTester:
    def __init__(self, model_path: str):
        """Initialize the tester with a trained model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        logger.info(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def parse_date(self, current_date: str, timezone: str, expression: str) -> str:
        """Parse a date expression using the fine-tuned model"""
        # Format input like training data
        input_text = f"Current Date: {current_date}\nTimezone: {timezone}\nExpression: '{expression}'"
        prompt = f"Human: Parse this date expression:\n{input_text}\n\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=150,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        assistant_start = response.find("Assistant:") + len("Assistant:")
        result = response[assistant_start:].strip()
        
        return result
    
    def test_examples(self, test_cases: list):
        """Test multiple examples"""
        print("\n🧪 Testing Date Parsing Model")
        print("=" * 50)
        
        for i, test_case in enumerate(test_cases, 1):
            current_date = test_case.get("current_date", "2025-01-15 12:00:00")
            timezone = test_case.get("timezone", "UTC")
            expression = test_case["expression"]
            expected = test_case.get("expected", "N/A")
            
            print(f"\nTest {i}: '{expression}'")
            print(f"Current Date: {current_date}")
            print(f"Timezone: {timezone}")
            
            try:
                result = self.parse_date(current_date, timezone, expression)
                print(f"Result: {result}")
                
                if expected != "N/A":
                    print(f"Expected: {expected}")
                    
            except Exception as e:
                print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned date parsing model")
    parser.add_argument("--model-path", default="./checkpoints/date-parsing-model", 
                       help="Path to the fine-tuned model")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = DateParsingTester(args.model_path)
    
    if args.interactive:
        # Interactive mode
        print("\n🎯 Interactive Date Parsing Test")
        print("Type 'quit' to exit")
        
        while True:
            try:
                expression = input("\nEnter date expression: ").strip()
                if expression.lower() in ['quit', 'exit', 'q']:
                    break
                
                current_date = input("Current date (YYYY-MM-DD HH:MM:SS) [2025-01-15 12:00:00]: ").strip()
                if not current_date:
                    current_date = "2025-01-15 12:00:00"
                
                timezone = input("Timezone [UTC]: ").strip()
                if not timezone:
                    timezone = "UTC"
                
                result = tester.parse_date(current_date, timezone, expression)
                print(f"\n✅ Result: {result}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    else:
        # Predefined test cases
        test_cases = [
            {
                "expression": "today",
                "current_date": "2025-01-15 12:00:00",
                "timezone": "UTC",
                "expected": '{"result": {"date": "2025-01-15T00:00:00+00:00"}, "result_type": "single"}'
            },
            {
                "expression": "tomorrow",
                "current_date": "2025-01-15 12:00:00",
                "timezone": "UTC",
                "expected": '{"result": {"date": "2025-01-16T00:00:00+00:00"}, "result_type": "single"}'
            },
            {
                "expression": "next Monday",
                "current_date": "2025-01-15 12:00:00",
                "timezone": "UTC"
            },
            {
                "expression": "in 3 days",
                "current_date": "2025-01-15 12:00:00",
                "timezone": "UTC"
            },
            {
                "expression": "last Friday",
                "current_date": "2025-01-15 12:00:00",
                "timezone": "UTC"
            }
        ]
        
        tester.test_examples(test_cases)

if __name__ == "__main__":
    main()

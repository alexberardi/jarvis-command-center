#!/usr/bin/env python3
"""
ng Single-shot command processing fine-tuning script for Llama 3.2 3B.
Combines command inference and parameter extraction into a single model call.
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for single-shot Llama 3.2 3B fine-tuning"""
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_length: int = 512
    batch_size: int = 4    # Conservative for 3B model
    learning_rate: float = 2e-5
    num_epochs: int = 2
    warmup_steps: int = 200
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "./checkpoints/jarvis-llama-3b-single-shot"
    logging_dir: str = "./logs"
    gradient_accumulation_steps: int = 4
    fp16: bool = False
    bf16: bool = False
    max_examples: int = 20000  # Total examples


class SingleShotDataGenerator:
    """Generate training data for single-shot command processing"""
    
    def __init__(self):
        self.rooms = ["living room", "bedroom", "kitchen", "office", "bathroom", "dining room", "garage"]
        self.users = ["default", "alex", "sarah", "mike", "emma"]
        self.voice_modes = ["brief", "detailed", "casual"]
        self.node_ids = ["node-123", "node-456", "node-789", "jarvis-main", "assistant-1"]
        
        # Real commands from the system
        self.real_commands = [
            {
                "name": "calculator_command",
                "description": "Perform basic arithmetic operations (add, subtract, multiply, divide) on two numbers",
                "parameters": [
                    {"name": "expression", "type": "string", "required": True, "description": "Mathematical expression to calculate"}
                ],
                "keywords": ["calculate", "math", "add", "subtract", "multiply", "divide", "plus", "minus"]
            },
            {
                "name": "general_knowledge_command", 
                "description": "Answers general knowledge questions using AI, covering topics like history, geography, science, and more. If the user's question is something you can accurately answer, select this command.",
                "parameters": [
                    {"name": "question", "type": "string", "required": True, "description": "The question to answer"}
                ],
                "keywords": ["what", "who", "when", "where", "why", "how", "question", "tell me", "explain"]
            },
            {
                "name": "open_weather_command",
                "description": "Gets the current weather or forecast for a city",
                "parameters": [
                    {"name": "location", "type": "string", "required": False, "description": "City or location name"},
                    {"name": "type", "type": "string", "required": False, "enum_values": ["current", "forecast"], "description": "Weather type"}
                ],
                "keywords": ["weather", "forecast", "temperature", "rain", "sunny", "cloudy", "hot", "cold"]
            },
            {
                "name": "read_calendar_command",
                "description": "Reads calendar events from your configured calendar service (iCloud, Google, etc.)",
                "parameters": [
                    {"name": "timeframe", "type": "string", "required": False, "description": "Time period to check (today, tomorrow, this week, etc.)"}
                ],
                "keywords": ["calendar", "events", "meetings", "appointments", "schedule", "what's on"]
            },
            {
                "name": "sports_schedule_command",
                "description": "Get sports schedules, find when teams play next, check upcoming games, and get game times for NFL, NBA, MLB, NHL, and college sports.",
                "parameters": [
                    {"name": "team_name", "type": "string", "required": True, "description": "Team name as spoken by user"},
                    {"name": "sport", "type": "string", "required": False, "enum_values": ["NFL", "NBA", "MLB", "NHL", "college"], "description": "Sport type"}
                ],
                "keywords": ["schedule", "when", "game", "play", "next game", "upcoming", "sports"]
            },
            {
                "name": "sports_score_command",
                "description": "Get sports scores and results for NFL, NBA, MLB, NHL, and college sports.",
                "parameters": [
                    {"name": "team_name", "type": "string", "required": True, "description": "The complete team name as spoken by the user"}
                ],
                "keywords": ["score", "result", "win", "lose", "game", "final", "how did", "did they win"]
            },
            {
                "name": "tell_a_story",
                "description": "Writes and reads a story to the user in chunks. Use 'continue' to hear more, 'end story' to finish.",
                "parameters": [
                    {"name": "genre", "type": "string", "required": False, "description": "Story genre or theme"},
                    {"name": "action", "type": "string", "required": False, "enum_values": ["start", "continue", "end"], "description": "Story action"}
                ],
                "keywords": ["story", "tale", "tell me", "once upon", "continue", "end story"]
            },
            {
                "name": "tell_a_joke",
                "description": "Tells a family-friendly joke, optionally on a specific topic",
                "parameters": [
                    {"name": "topic", "type": "string", "required": False, "description": "Joke topic or theme"}
                ],
                "keywords": ["joke", "funny", "humor", "laugh", "tell me a joke", "make me laugh"]
            }
        ]
        
        # Date mapping for relative dates
        self.date_mappings = {
            "today": "2025-09-08T04:00:00Z",
            "tomorrow": "2025-09-09T04:00:00Z", 
            "yesterday": "2025-09-07T04:00:00Z",
            "this week": ["2025-09-07T04:00:00Z", "2025-09-08T04:00:00Z", "2025-09-09T04:00:00Z", "2025-09-10T04:00:00Z", "2025-09-11T04:00:00Z", "2025-09-12T04:00:00Z", "2025-09-13T04:00:00Z"],
            "next week": ["2025-09-14T04:00:00Z", "2025-09-15T04:00:00Z", "2025-09-16T04:00:00Z", "2025-09-17T04:00:00Z", "2025-09-18T04:00:00Z", "2025-09-19T04:00:00Z", "2025-09-20T04:00:00Z"],
            "last week": ["2025-08-31T04:00:00Z", "2025-09-01T04:00:00Z", "2025-09-02T04:00:00Z", "2025-09-03T04:00:00Z", "2025-09-04T04:00:00Z", "2025-09-05T04:00:00Z", "2025-09-06T04:00:00Z"]
        }
        
        # Cities for weather examples
        self.cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose", "Boston", "Seattle", "Miami", "Denver", "Atlanta"]
        
        # Sports teams
        self.teams = {
            "NFL": ["Patriots", "Cowboys", "Packers", "Steelers", "49ers", "Giants", "Eagles", "Chiefs", "Broncos", "Raiders"],
            "NBA": ["Lakers", "Celtics", "Warriors", "Heat", "Bulls", "Knicks", "Spurs", "Nets", "Clippers", "Rockets"],
            "MLB": ["Yankees", "Red Sox", "Dodgers", "Giants", "Cubs", "Cardinals", "Astros", "Phillies", "Braves", "Mets"]
        }
    
    def generate_node_context(self) -> str:
        """Generate varied node context"""
        room = random.choice(self.rooms)
        user = random.choice(self.users)
        voice_mode = random.choice(self.voice_modes)
        node_id = random.choice(self.node_ids)
        
        context = f"room: {room}\nnode_id: {node_id}\nuser: {user}\nvoice_mode: {voice_mode}"
        
        # Sometimes add optional fields
        if random.random() < 0.3:
            context += f"\ndevice_id: device-{random.randint(100, 999)}"
        if random.random() < 0.2:
            context += f"\ntimezone: America/New_York"
        if random.random() < 0.1:
            context += f"\nlanguage: en-US"
            
        return context
    
    def generate_date_mapping_context(self) -> str:
        """Generate the date mapping context that appears in system prompts"""
        return '''RELATIVE TO ABSOLUTE DATE MAPPING:
{
  "now": "2025-09-08T13:31:36.641176-04:00",
  "today": "2025-09-08T04:00:00Z",
  "tomorrow": "2025-09-09T04:00:00Z",
  "yesterday": "2025-09-07T04:00:00Z",
  "this week": [
    "2025-09-07T04:00:00Z",
    "2025-09-08T04:00:00Z",
    "2025-09-09T04:00:00Z",
    "2025-09-10T04:00:00Z",
    "2025-09-11T04:00:00Z",
    "2025-09-12T04:00:00Z",
    "2025-09-13T04:00:00Z"
  ],
  "next week": [
    "2025-09-14T04:00:00Z",
    "2025-09-15T04:00:00Z",
    "2025-09-16T04:00:00Z",
    "2025-09-17T04:00:00Z",
    "2025-09-18T04:00:00Z",
    "2025-09-19T04:00:00Z",
    "2025-09-20T04:00:00Z"
  ],
  "last week": [
    "2025-08-31T04:00:00Z",
    "2025-09-01T04:00:00Z",
    "2025-09-02T04:00:00Z",
    "2025-09-03T04:00:00Z",
    "2025-09-04T04:00:00Z",
    "2025-09-05T04:00:00Z",
    "2025-09-06T04:00:00Z"
  ]
}'''
    
    def generate_single_shot_example(self) -> Dict[str, str]:
        """Generate a single-shot command processing example"""
        command = random.choice(self.real_commands)
        node_context = self.generate_node_context()
        date_context = self.generate_date_mapping_context()
        
        # Generate appropriate voice command and parameters
        if command["name"] == "calculator_command":
            expressions = ["2 + 2", "15 * 7", "100 / 4", "25 - 8", "3.14 * 2", "sqrt(16)"]
            expr = random.choice(expressions)
            voice_command = random.choice([
                f"Calculate {expr}",
                f"What's {expr}?",
                f"Can you compute {expr}",
                f"Math: {expr}"
            ])
            parameters = {"expression": expr}
            
        elif command["name"] == "general_knowledge_command":
            questions = [
                "What is the capital of France?",
                "Who invented the telephone?", 
                "When did World War 2 end?",
                "How many planets are in our solar system?",
                "What is photosynthesis?"
            ]
            question = random.choice(questions)
            voice_command = question
            parameters = {"question": question}
            
        elif command["name"] == "open_weather_command":
            city = random.choice(self.cities)
            weather_types = ["current", "forecast"]
            weather_type = random.choice(weather_types) if random.random() < 0.7 else None
            
            if weather_type:
                voice_command = random.choice([
                    f"What's the {weather_type} weather in {city}?",
                    f"Get me the {weather_type} for {city}",
                    f"Show {weather_type} weather for {city}"
                ])
                parameters = {"location": city, "type": weather_type}
            else:
                voice_command = random.choice([
                    f"What's the weather in {city}?",
                    f"How's the weather in {city}?",
                    f"Weather for {city}"
                ])
                parameters = {"location": city}
                
        elif command["name"] == "read_calendar_command":
            timeframes = ["today", "tomorrow", "this week", "next week"]
            timeframe = random.choice(timeframes)
            
            voice_command = random.choice([
                f"What's on my calendar {timeframe}?",
                f"Show me my schedule for {timeframe}",
                f"Any meetings {timeframe}?",
                f"Calendar for {timeframe}"
            ])
            
            # Convert relative date to absolute using mapping
            if timeframe in self.date_mappings:
                absolute_date = self.date_mappings[timeframe]
                parameters = {"timeframe": absolute_date}
            else:
                parameters = {"timeframe": timeframe}
                
        elif command["name"] == "sports_score_command":
            sport = random.choice(list(self.teams.keys()))
            team = random.choice(self.teams[sport])
            
            voice_command = random.choice([
                f"Did the {team} win yesterday?",
                f"What was the {team} score?",
                f"How did the {team} do?",
                f"{team} game result"
            ])
            parameters = {"team_name": team}
            
        elif command["name"] == "sports_schedule_command":
            sport = random.choice(list(self.teams.keys()))
            team = random.choice(self.teams[sport])
            
            voice_command = random.choice([
                f"When do the {team} play next?",
                f"What's the {team} schedule?",
                f"Next {team} game?",
                f"{team} upcoming games"
            ])
            parameters = {"team_name": team, "sport": sport}
            
        elif command["name"] == "tell_a_story":
            genres = ["adventure", "mystery", "fantasy", "sci-fi", "comedy"]
            genre = random.choice(genres) if random.random() < 0.6 else None
            
            if genre:
                voice_command = random.choice([
                    f"Tell me a {genre} story",
                    f"I want to hear a {genre} story",
                    f"Story time - something {genre}"
                ])
                parameters = {"genre": genre, "action": "start"}
            else:
                voice_command = random.choice([
                    "Tell me a story",
                    "Story time",
                    "I want to hear a story"
                ])
                parameters = {"action": "start"}
                
        elif command["name"] == "tell_a_joke":
            topics = ["animals", "food", "technology", "work", "family"]
            topic = random.choice(topics) if random.random() < 0.5 else None
            
            if topic:
                voice_command = random.choice([
                    f"Tell me a joke about {topic}",
                    f"I want a {topic} joke",
                    f"Joke about {topic}"
                ])
                parameters = {"topic": topic}
            else:
                voice_command = random.choice([
                    "Tell me a joke",
                    "I want to hear a joke",
                    "Make me laugh"
                ])
                parameters = {}
        
        # Create the expected output
        output = {
            "s": True,
            "n": command["name"],
            "p": parameters,
            "e": None
        }
        
        # Create system prompt similar to the real system
        system_prompt = f'''You are an inference engine for a voice assistant. Ultimately, your job is to match a command name and parameters from a voice command to a set of available commands.

Current Date & Time: Monday, September 08 2025 at 01:31 PM
Your Local Timezone: America/New_York (UTC-0400)
Current date (ISO): 2025-09-08

{date_context}

Node Context: {node_context}

You are a voice assistant command classifier. Identify which command the user wants.

COMMANDS & EXAMPLES:

- calculator_command - Perform basic arithmetic operations (add, subtract, multiply, divide) on two numbers
- general_knowledge_command - Answers general knowledge questions using AI, covering topics like history, geography, science, and more. If the user's question is something you can accurately answer, select this command.
- open_weather_command - Gets the current weather or forecast for a city
- read_calendar_command - Reads calendar events from your configured calendar service (iCloud, Google, etc.)
- sports_schedule_command - Get sports schedules, find when teams play next, check upcoming games, and get game times for NFL, NBA, MLB, NHL, and college sports.
- sports_score_command - Get sports scores and results for NFL, NBA, MLB, NHL, and college sports.
- tell_a_story - Writes and reads a story to the user in chunks. Use 'continue' to hear more, 'end story' to finish.
- tell_a_joke - Tells a family-friendly joke, optionally on a specific topic

Always respond with JSON: {{"s":true,"n":"command_name","p":{{...}},"e":null}}'''
        
        # Format as chat messages for Llama
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": voice_command}
        ]
        
        return {
            "messages": messages,
            "expected_output": json.dumps(output, separators=(',', ':'))
        }
    
    def generate_error_example(self) -> Dict[str, str]:
        """Generate error handling examples"""
        node_context = self.generate_node_context()
        date_context = self.generate_date_mapping_context()
        
        error_type = random.choice(["no_command_match", "missing_parameters"])
        
        if error_type == "no_command_match":
            # Commands that don't match any available command
            voice_commands = [
                "Turn on the lights in the living room",
                "Set the temperature to 72 degrees", 
                "Play some music",
                "Lock the front door",
                "Start the vacuum cleaner"
            ]
            voice_command = random.choice(voice_commands)
            
            output = {
                "s": False,
                "n": None,
                "p": {},
                "e": "No matching command found"
            }
        else:  # missing_parameters
            # Commands that are missing required parameters
            voice_commands = [
                "Calculate",  # missing expression
                "What's the weather?",  # missing location
                "When do they play?"  # missing team
            ]
            voice_command = random.choice(voice_commands)
            
            output = {
                "s": False, 
                "n": None,
                "p": {},
                "e": "Missing required parameters"
            }
        
        system_prompt = f'''You are an inference engine for a voice assistant. Ultimately, your job is to match a command name and parameters from a voice command to a set of available commands.

Current Date & Time: Monday, September 08 2025 at 01:31 PM
Your Local Timezone: America/New_York (UTC-0400)
Current date (ISO): 2025-09-08

{date_context}

Node Context: {node_context}

You are a voice assistant command classifier. Identify which command the user wants.

COMMANDS & EXAMPLES:

- calculator_command - Perform basic arithmetic operations (add, subtract, multiply, divide) on two numbers
- general_knowledge_command - Answers general knowledge questions using AI, covering topics like history, geography, science, and more. If the user's question is something you can accurately answer, select this command.
- open_weather_command - Gets the current weather or forecast for a city
- read_calendar_command - Reads calendar events from your configured calendar service (iCloud, Google, etc.)
- sports_schedule_command - Get sports schedules, find when teams play next, check upcoming games, and get game times for NFL, NBA, MLB, NHL, and college sports.
- sports_score_command - Get sports scores and results for NFL, NBA, MLB, NHL, and college sports.
- tell_a_story - Writes and reads a story to the user in chunks. Use 'continue' to hear more, 'end story' to finish.
- tell_a_joke - Tells a family-friendly joke, optionally on a specific topic

Always respond with JSON: {{"s":true,"n":"command_name","p":{{...}},"e":null}}'''
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": voice_command}
        ]
        
        return {
            "messages": messages,
            "expected_output": json.dumps(output, separators=(',', ':'))
        }
    
    def generate_training_data(self, num_examples: int) -> List[Dict[str, Any]]:
        """Generate training dataset"""
        training_data = []
        
        # 90% successful command processing examples
        success_examples = int(num_examples * 0.9)
        logger.info(f"Generating {success_examples} successful command examples...")
        
        for i in range(success_examples):
            example = self.generate_single_shot_example()
            training_data.append(example)
            if (i + 1) % 1000 == 0:
                logger.info(f"Generated {i + 1}/{success_examples} examples")
        
        # 10% error handling examples
        error_examples = num_examples - success_examples
        logger.info(f"Generating {error_examples} error handling examples...")
        
        for i in range(error_examples):
            example = self.generate_error_example()
            training_data.append(example)
        
        # Shuffle
        random.shuffle(training_data)
        
        logger.info(f"✅ Generated {len(training_data)} total training examples")
        return training_data


class SingleShotLlamaTrainer:
    """Trainer for single-shot Llama 3.2 3B fine-tuning"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Setup data generator
        self.data_generator = SingleShotDataGenerator()
    
    def load_model_and_tokenizer(self):
        """Load the Llama model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Check if HF token is needed
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        except Exception as e:
            if "gated repo" in str(e).lower():
                logger.warning("Model requires authentication. Please provide HF token.")
                hf_token = os.getenv("HF_TOKEN")
                if not hf_token:
                    hf_token = input("Enter your Hugging Face token: ").strip()
                    os.environ["HF_TOKEN"] = hf_token
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    token=hf_token
                )
            else:
                raise e
        
        # Load model with FP32 for stability
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            token=os.getenv("HF_TOKEN")
        )
        
        # Ensure model is in FP32
        self.model = self.model.float()
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info(f"✅ Model and tokenizer loaded successfully")
    
    def prepare_dataset(self, examples: List[Dict[str, Any]]) -> Dataset:
        """Prepare dataset for training"""
        logger.info("Preparing dataset for training...")
        
        # Convert to Llama chat format and tokenize
        formatted_examples = []
        
        for example in examples:
            # Apply chat template
            chat_text = self.tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Add the expected output
            full_text = chat_text + example["expected_output"]
            
            formatted_examples.append({"text": full_text})
        
        # Create dataset
        dataset = Dataset.from_list(formatted_examples)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_tensors=None
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        logger.info(f"✅ Dataset prepared with {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def custom_data_collator(self, features):
        """Custom data collator that handles variable sequence lengths"""
        # Find the maximum length in this batch
        max_length = max(len(f["input_ids"]) for f in features)
        
        # Pad all sequences to max length
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = [1] * len(input_ids)
            
            # Pad to max_length
            padding_length = max_length - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            
            # Labels are the same as input_ids, but padded positions get -100
            labels = input_ids.copy()
            for i in range(len(labels) - padding_length, len(labels)):
                labels[i] = -100
            
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)
        
        # Convert to tensors
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long)
        }
    
    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_dir=self.config.logging_dir,
            logging_steps=50,
            save_steps=self.config.save_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to=[]
        )
    
    def train(self) -> None:
        """Execute the training process"""
        logger.info("🚀 Starting single-shot training process...")
        
        # Generate training data
        logger.info("📊 Generating training data...")
        training_examples = self.data_generator.generate_training_data(self.config.max_examples)
        
        # Prepare dataset
        dataset = self.prepare_dataset(training_examples)
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = dataset.train_test_split(
            train_size=train_size, 
            test_size=eval_size,
            seed=42
        ).values()
        
        logger.info(f"📊 Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")
        
        # Setup training
        training_args = self.setup_training_args()
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.custom_data_collator
        )
        
        # Train
        logger.info("🎯 Beginning training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("✅ Single-shot Llama 3.2 training completed!")
        logger.info(f"📁 Model saved to: {self.config.output_dir}")


def push_to_huggingface(model_path: str, repo_name: str, hf_token: str, private: bool = False):
    """Push trained model to Hugging Face Hub"""
    try:
        from huggingface_hub import HfApi, create_repo
        
        logger.info(f"🚀 Pushing model to Hugging Face: {repo_name}")
        
        # Login
        api = HfApi(token=hf_token)
        
        # Create repo
        try:
            create_repo(repo_name, token=hf_token, private=private)
            logger.info(f"✅ Created repository: {repo_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Repository {repo_name} already exists")
            else:
                raise e
        
        # Upload model files
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            token=hf_token
        )
        
        logger.info(f"✅ Model successfully pushed to: https://huggingface.co/{repo_name}")
        
    except ImportError:
        logger.error("huggingface_hub not available. Install with: pip install huggingface_hub")
    except Exception as e:
        logger.error(f"Failed to push to Hugging Face: {e}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Single-shot Llama 3.2 3B fine-tuning")
    parser.add_argument("--push-to-hf", action="store_true", help="Push model to Hugging Face Hub")
    parser.add_argument("--repo-name", type=str, default="alexberardi/jarvis-llama-3b", help="HF repository name")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token")
    parser.add_argument("--private", action="store_true", help="Make HF repo private")
    parser.add_argument("--public", action="store_true", help="Make HF repo public")
    parser.add_argument("--max-examples", type=int, default=20000, help="Maximum training examples")
    
    args = parser.parse_args()
    
    # Setup config
    config = TrainingConfig()
    config.max_examples = args.max_examples
    
    # Adjust batch size based on available GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU memory: {gpu_memory:.1f} GB")
        
        # 3B model needs much smaller batch sizes
        if gpu_memory > 20:  # A100 or similar
            config.batch_size = 2  # Reduced from 8
            config.gradient_accumulation_steps = 8  # Increased from 2
        else:  # A40 or similar
            config.batch_size = 1
            config.gradient_accumulation_steps = 16
    else:
        config.batch_size = 1
        config.gradient_accumulation_steps = 16
    
    logger.info(f"Using batch_size={config.batch_size}, grad_accum={config.gradient_accumulation_steps}")
    
    # Train model
    trainer = SingleShotLlamaTrainer(config)
    trainer.train()
    
    # Push to HF if requested
    if args.push_to_hf:
        hf_token = args.hf_token or os.getenv("HF_TOKEN")
        if not hf_token:
            hf_token = input("Enter your Hugging Face token: ").strip()
        
        private = args.private and not args.public
        
        push_to_huggingface(
            model_path=config.output_dir,
            repo_name=args.repo_name,
            hf_token=hf_token,
            private=private
        )


if __name__ == "__main__":
    main()

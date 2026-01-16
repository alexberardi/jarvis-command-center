#!/usr/bin/env python3
"""
Multi-Task Llama 3.2 1B Fine-tuning Script
Trains on: Command Inference + Parameter Extraction + Date Parsing
"""

import json
import os
import sys
import argparse
import re
import random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
import pytz

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
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
        logging.FileHandler('logs/multitask_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for multi-task Llama 3.2 1B fine-tuning"""
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_length: int = 512  # Longer context for full prompts
    batch_size: int = 8    # Conservative for FP32
    learning_rate: float = 2e-5
    num_epochs: int = 2    # More epochs for multi-task learning
    warmup_steps: int = 200
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "./checkpoints/jarvis-llama-3b-v1"
    logging_dir: str = "./logs"
    gradient_accumulation_steps: int = 2
    fp16: bool = False     # Disable FP16 for stability
    bf16: bool = False
    max_examples_per_task: int = 15000  # 15K per task = 45K total

class MultiTaskDataGenerator:
    """Generate synthetic training data for all three tasks"""
    
    def __init__(self):
        self.rooms = ["living room", "bedroom", "kitchen", "office", "bathroom", "dining room", "garage"]
        self.users = ["default", "alex", "sarah", "mike", "emma"]
        self.voice_modes = ["brief", "detailed", "casual"]
        self.node_ids = ["node-123", "node-456", "node-789", "jarvis-main", "assistant-1"]
        
        # Real commands from your actual system (based on the example you provided)
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
        
        # Sample commands with realistic parameters for variety
        self.sample_commands = [
            {
                "name": "light_control",
                "description": "Control smart lights (on/off, dimming, color changes)",
                "parameters": [
                    {"name": "room", "type": "string", "required": True, "description": "Room where lights are located"},
                    {"name": "action", "type": "string", "required": True, "enum_values": ["on", "off", "dim", "brighten"], "description": "Action to perform"},
                    {"name": "brightness", "type": "number", "required": False, "description": "Brightness level 0-100"},
                    {"name": "color", "type": "string", "required": False, "description": "Color name or hex code"}
                ],
                "keywords": ["light", "lights", "lamp", "illuminate", "dim", "bright"]
            },
            {
                "name": "thermostat_adjust",
                "description": "Adjust temperature settings",
                "parameters": [
                    {"name": "temperature", "type": "number", "required": True, "description": "Target temperature in Fahrenheit"},
                    {"name": "mode", "type": "string", "required": False, "enum_values": ["heat", "cool", "auto", "off"], "description": "HVAC mode"}
                ],
                "keywords": ["temperature", "thermostat", "heat", "cool", "warm", "cold"]
            },
            {
                "name": "music_player",
                "description": "Control music playback",
                "parameters": [
                    {"name": "action", "type": "string", "required": True, "enum_values": ["play", "pause", "stop", "next", "previous"], "description": "Playback action"},
                    {"name": "query", "type": "string", "required": False, "description": "Song, artist, or playlist name"},
                    {"name": "volume", "type": "number", "required": False, "description": "Volume level 0-100"}
                ],
                "keywords": ["music", "play", "song", "artist", "playlist", "volume"]
            },
            {
                "name": "timer_manager",
                "description": "Set, check, or cancel timers",
                "parameters": [
                    {"name": "action", "type": "string", "required": True, "enum_values": ["set", "check", "cancel", "list"], "description": "Timer action"},
                    {"name": "duration", "type": "number", "required": False, "description": "Timer duration in minutes"},
                    {"name": "label", "type": "string", "required": False, "description": "Timer label or description"}
                ],
                "keywords": ["timer", "alarm", "countdown", "remind", "minutes", "hours"]
            },
            {
                "name": "weather_info",
                "description": "Get weather information and forecasts",
                "parameters": [
                    {"name": "location", "type": "string", "required": False, "description": "City or location name"},
                    {"name": "type", "type": "string", "required": False, "enum_values": ["current", "forecast", "hourly"], "description": "Weather information type"}
                ],
                "keywords": ["weather", "forecast", "temperature", "rain", "sunny", "cloudy"]
            },
            {
                "name": "news_reader",
                "description": "Read news headlines and articles",
                "parameters": [
                    {"name": "category", "type": "string", "required": False, "enum_values": ["general", "business", "technology", "sports", "health"], "description": "News category"},
                    {"name": "source", "type": "string", "required": False, "description": "Preferred news source"}
                ],
                "keywords": ["news", "headlines", "article", "update", "current events"]
            },
            {
                "name": "reminder_system",
                "description": "Create and manage reminders",
                "parameters": [
                    {"name": "action", "type": "string", "required": True, "enum_values": ["create", "list", "delete", "complete"], "description": "Reminder action"},
                    {"name": "message", "type": "string", "required": False, "description": "Reminder message or content"},
                    {"name": "priority", "type": "string", "required": False, "enum_values": ["low", "medium", "high", "urgent"], "description": "Reminder priority level"}
                ],
                "keywords": ["remind", "reminder", "note", "remember", "task", "todo"]
            },
            {
                "name": "smart_lock",
                "description": "Control smart door locks",
                "parameters": [
                    {"name": "door", "type": "string", "required": True, "description": "Door identifier (front, back, garage, etc.)"},
                    {"name": "action", "type": "string", "required": True, "enum_values": ["lock", "unlock", "status"], "description": "Lock action to perform"}
                ],
                "keywords": ["lock", "unlock", "door", "secure", "access", "entry"]
            },
            {
                "name": "email_assistant",
                "description": "Send emails and check inbox",
                "parameters": [
                    {"name": "action", "type": "string", "required": True, "enum_values": ["send", "check", "read", "delete"], "description": "Email action"},
                    {"name": "recipient", "type": "string", "required": False, "description": "Email recipient"},
                    {"name": "subject", "type": "string", "required": False, "description": "Email subject line"},
                    {"name": "message", "type": "string", "required": False, "description": "Email message content"}
                ],
                "keywords": ["email", "send", "mail", "inbox", "message", "recipient"]
            },
            {
                "name": "calendar_manager",
                "description": "Manage calendar events and appointments",
                "parameters": [
                    {"name": "action", "type": "string", "required": True, "enum_values": ["create", "read", "update", "delete"], "description": "Calendar action"},
                    {"name": "title", "type": "string", "required": False, "description": "Event title or description"},
                    {"name": "location", "type": "string", "required": False, "description": "Event location"}
                ],
                "keywords": ["calendar", "event", "meeting", "appointment", "schedule"]
            }
        ]

    def generate_node_context(self) -> Dict[str, str]:
        """Generate realistic node context with room-specific and user-specific variations"""
        room = random.choice(self.rooms)
        user = random.choice(self.users)
        
        # Add some context-specific variations
        context = {
            "room": room,
            "node_id": random.choice(self.node_ids),
            "user": user,
            "voice_mode": random.choice(self.voice_modes)
        }
        
        # Add occasional additional context fields
        if random.random() < 0.3:  # 30% chance
            context["device_id"] = f"device-{random.randint(100, 999)}"
        
        if random.random() < 0.2:  # 20% chance  
            context["timezone"] = random.choice(["America/New_York", "America/Los_Angeles", "Europe/London", "UTC"])
            
        if random.random() < 0.1:  # 10% chance
            context["language"] = random.choice(["en", "en-US", "en-GB"])
            
        return context

    def generate_date_context(self, base_date: datetime = None) -> Dict[str, Any]:
        """Generate realistic date context mapping"""
        if base_date is None:
            base_date = datetime.now(pytz.UTC)
        
        # Create date mappings similar to your real system
        date_context = {
            "now": base_date.isoformat(),
            "today": base_date.replace(hour=4, minute=0, second=0, microsecond=0).isoformat().replace('+00:00', 'Z'),
            "tomorrow": (base_date + timedelta(days=1)).replace(hour=4, minute=0, second=0, microsecond=0).isoformat().replace('+00:00', 'Z'),
            "yesterday": (base_date - timedelta(days=1)).replace(hour=4, minute=0, second=0, microsecond=0).isoformat().replace('+00:00', 'Z')
        }
        
        # Add week mappings
        start_of_week = base_date - timedelta(days=base_date.weekday())
        this_week = []
        next_week = []
        last_week = []
        
        for i in range(7):
            this_week.append((start_of_week + timedelta(days=i)).replace(hour=4, minute=0, second=0, microsecond=0).isoformat().replace('+00:00', 'Z'))
            next_week.append((start_of_week + timedelta(days=i+7)).replace(hour=4, minute=0, second=0, microsecond=0).isoformat().replace('+00:00', 'Z'))
            last_week.append((start_of_week + timedelta(days=i-7)).replace(hour=4, minute=0, second=0, microsecond=0).isoformat().replace('+00:00', 'Z'))
        
        date_context.update({
            "this week": this_week,
            "next week": next_week,
            "last week": last_week
        })
        
        return date_context

    def _generate_fictional_commands(self, count: int) -> List[Dict[str, Any]]:
        """Generate fictional commands to prevent overfitting to specific command sets"""
        fictional_bases = [
            ("security_camera", "View and control security cameras"),
            ("garage_door", "Control garage door opening and closing"),
            ("sprinkler_system", "Manage lawn sprinkler schedules"),
            ("pool_heater", "Control swimming pool temperature"),
            ("window_blinds", "Adjust window blinds and curtains"),
            ("coffee_maker", "Program and start coffee brewing"),
            ("dishwasher", "Control dishwasher cycles"),
            ("washing_machine", "Start and monitor laundry cycles"),
            ("air_purifier", "Control air purification settings"),
            ("robot_vacuum", "Schedule and control robotic vacuum"),
            ("smart_doorbell", "Manage doorbell and visitor notifications"),
            ("backup_system", "Control data backup operations"),
            ("network_scanner", "Scan and monitor network devices"),
            ("media_server", "Manage local media streaming"),
            ("plant_monitor", "Monitor plant health and watering"),
        ]
        
        selected = random.sample(fictional_bases, min(count, len(fictional_bases)))
        fictional_commands = []
        
        for name, description in selected:
            # Create realistic but fictional command structure
            fictional_commands.append({
                "name": name,
                "description": description,
                "parameters": [
                    {"name": "action", "type": "string", "required": True, "description": "Action to perform"}
                ],
                "keywords": [name.replace("_", " "), "control", "manage"]
            })
        
        return fictional_commands

    def generate_error_example(self) -> Dict[str, Any]:
        """Generate realistic error/failure cases for robust training"""
        error_types = [
            "no_command_match",
            "missing_parameters", 
            "ambiguous_command",
            "invalid_parameters"
        ]
        
        error_type = random.choice(error_types)
        
        if error_type == "no_command_match":
            # User asks for something not in available commands
            voice_commands = [
                "Turn on the dishwasher", "Start the coffee maker", "Check my stocks",
                "Send an email", "Play Netflix", "Order pizza", "Call mom",
                "Set the security system", "Water the plants"
            ]
            
            # Use commands that don't include the requested functionality
            available_commands = random.sample(self.real_commands[:4], k=3)  # Just a few basic commands
            voice_command = random.choice(voice_commands)
            
            node_context = self.generate_node_context()
            date_context = self.generate_date_context()
            system_prompt = self._build_command_inference_system_prompt(node_context, date_context, available_commands)
            
            expected_output = {
                "s": False,
                "n": None,
                "e": {"type": "no_command_match", "message": "No matching command found"}
            }
            
            return {
                "task": "command_inference_error",
                "input": f"System: {system_prompt}\nUser: {voice_command}",
                "output": json.dumps(expected_output, separators=(',', ':'))
            }
            
        elif error_type == "missing_parameters":
            # User doesn't provide required parameters
            command = random.choice([cmd for cmd in self.real_commands if any(p.get('required', True) for p in cmd['parameters'])])
            
            # Generate command request without required parameters
            voice_commands = {
                "sports_score_command": ["Who won?", "What's the score?", "Did they win?"],
                "open_weather_command": ["What's the weather?", "Is it raining?"],
                "calculator_command": ["Do some math", "Calculate something"]
            }
            
            voice_command = random.choice(voice_commands.get(command['name'], ["Execute command"]))
            
            system_prompt = self._build_parameter_extraction_prompt(command, voice_command)
            
            missing_params = [p['name'] for p in command['parameters'] if p.get('required', True)]
            
            expected_output = {
                "s": False,
                "p": {},
                "e": {
                    "type": "missing_parameters",
                    "missing_parameters": missing_params,
                    "message": f"Missing required parameters: {', '.join(missing_params)}"
                }
            }
            
            return {
                "task": "parameter_extraction_error",
                "input": f"System: {system_prompt}\nUser: {voice_command}",
                "output": json.dumps(expected_output, separators=(',', ':'))
            }
        
        # Default fallback
        return self.generate_command_inference_example()

    def generate_command_inference_example(self) -> Dict[str, Any]:
        """Generate a command inference training example"""
        # Combine real commands with sample commands for variety
        all_available_commands = self.real_commands + self.sample_commands
        
        # Vary the number of available commands more dramatically (2-10)
        num_commands = random.randint(2, min(10, len(all_available_commands)))
        available_commands = random.sample(all_available_commands, k=num_commands)
        
        # Sometimes add completely fictional commands to prevent overfitting
        if random.random() < 0.3:  # 30% chance to add fictional commands
            fictional_commands = self._generate_fictional_commands(random.randint(1, 3))
            available_commands.extend(fictional_commands)
            random.shuffle(available_commands)
        
        # Select the target command (must be from real/sample commands, not fictional)
        real_commands = [cmd for cmd in available_commands if cmd in all_available_commands]
        selected_command = random.choice(real_commands)
        
        # Generate context
        node_context = self.generate_node_context()
        date_context = self.generate_date_context()
        
        # Create voice commands that would trigger this command
        voice_commands = self._generate_voice_commands_for_command(selected_command)
        voice_command = random.choice(voice_commands)
        
        # Build system prompt
        system_prompt = self._build_command_inference_system_prompt(node_context, date_context, available_commands)
        
        # Expected output
        expected_output = {
            "s": True,
            "n": selected_command["name"],
            "e": None
        }
        
        return {
            "task": "command_inference",
            "input": f"System: {system_prompt}\nUser: {voice_command}",
            "output": json.dumps(expected_output, separators=(',', ':'))
        }

    def generate_parameter_extraction_example(self) -> Dict[str, Any]:
        """Generate a parameter extraction training example"""
        command = random.choice(self.sample_commands)
        
        # Generate realistic voice command and parameters
        voice_command, expected_params = self._generate_voice_command_with_params(command)
        
        # Build parameter extraction prompt
        system_prompt = self._build_parameter_extraction_prompt(command, voice_command)
        
        # Expected output
        expected_output = {
            "s": True,
            "p": expected_params,
            "e": None
        }
        
        return {
            "task": "parameter_extraction",
            "input": f"System: {system_prompt}\nUser: {voice_command}",
            "output": json.dumps(expected_output, separators=(',', ':'))
        }

    def generate_date_extraction_example(self) -> Dict[str, Any]:
        """Generate a date extraction training example"""
        # Use existing date training data format but adapt for multi-task
        date_expressions = [
            "today", "tomorrow", "yesterday", "next week", "last week", 
            "this week", "next Monday", "last Friday", "this weekend"
        ]
        
        date_expr = random.choice(date_expressions)
        date_context = self.generate_date_context()
        
        # Create voice command with date reference
        voice_commands = [
            f"What meetings do I have {date_expr}?",
            f"Set a reminder for {date_expr}",
            f"Check my calendar {date_expr}",
            f"What's the weather {date_expr}?",
            f"Schedule something {date_expr}"
        ]
        
        voice_command = random.choice(voice_commands)
        
        # Build date extraction prompt
        system_prompt = self._build_date_extraction_prompt(date_context, voice_command)
        
        # Get expected date value from context
        expected_dates = date_context.get(date_expr, [date_context["today"]])
        if not isinstance(expected_dates, list):
            expected_dates = [expected_dates]
        
        expected_output = {
            "s": True,
            "p": {"datetimes": expected_dates},
            "e": None
        }
        
        return {
            "task": "date_extraction", 
            "input": f"System: {system_prompt}\nUser: {voice_command}",
            "output": json.dumps(expected_output, separators=(',', ':'))
        }

    def _generate_voice_commands_for_command(self, command: Dict[str, Any]) -> List[str]:
        """Generate realistic voice commands for a given command"""
        name = command["name"]
        keywords = command["keywords"]
        
        # Generate variations based on command type
        if name == "light_control":
            return [
                "Turn on the lights", "Turn off the bedroom lights", "Dim the living room lights",
                "Brighten the kitchen", "Set lights to blue", "Make it darker in here"
            ]
        elif name == "thermostat_adjust":
            return [
                "Set temperature to 72", "Make it warmer", "Turn up the heat",
                "Cool it down to 68", "Adjust thermostat to 75 degrees"
            ]
        elif name == "music_player":
            return [
                "Play some music", "Play Taylor Swift", "Pause the music",
                "Skip this song", "Turn up the volume", "Play my workout playlist"
            ]
        elif name == "timer_manager":
            return [
                "Set a timer for 10 minutes", "Set a cooking timer", "How much time is left?",
                "Cancel the timer", "Set a 30 minute timer for laundry"
            ]
        elif name == "weather_info":
            return [
                "What's the weather?", "Weather in New York", "Will it rain today?",
                "What's the forecast?", "How hot is it outside?"
            ]
        elif name == "calendar_manager":
            return [
                "What meetings do I have today?", "Schedule a meeting", "Check my calendar",
                "Add appointment for tomorrow", "What's on my schedule?"
            ]
        else:
            # Generate generic commands using keywords
            return [f"{random.choice(keywords)} {random.choice(['please', 'now', 'for me', ''])}" for _ in range(3)]

    def _generate_voice_command_with_params(self, command: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate voice command with realistic parameter extraction"""
        name = command["name"]
        params = {}
        
        if name == "light_control":
            voice_command = "Turn on the living room lights"
            params = {"room": "living room", "action": "on"}
        elif name == "thermostat_adjust":
            voice_command = "Set temperature to 72 degrees"
            params = {"temperature": 72, "mode": "auto"}
        elif name == "music_player":
            voice_command = "Play some jazz music"
            params = {"action": "play", "query": "jazz music"}
        elif name == "timer_manager":
            voice_command = "Set a 15 minute timer for cooking"
            params = {"action": "set", "duration": 15, "label": "cooking"}
        else:
            # Generate basic params
            voice_command = f"Execute {name}"
            params = {}
        
        return voice_command, params

    def _build_command_inference_system_prompt(self, node_context: Dict[str, str], date_context: Dict[str, Any], available_commands: List[Dict[str, Any]]) -> str:
        """Build system prompt for command inference"""
        # Format commands list
        commands_str = '\n'.join([
            f"- {cmd['name']} - {cmd['description']}"
            for cmd in available_commands
        ])
        
        # Build context strings
        node_context_str = '\n'.join([f"{k}: {v}" for k, v in node_context.items()])
        date_context_str = json.dumps(date_context, indent=2)
        
        return f"""Conversation ID: conv-{random.randint(1000, 9999)}

You are an inference engine for a voice assistant. Ultimately, your job is to match a command name and parameters from a voice command to a set of available commands. This prompt establishes the base context and rules.

Current Date & Time: {datetime.now().strftime('%A, %B %d %Y at %I:%M %p')}
Your Local Timezone: America/New_York (UTC-0400)
Current date (ISO): {datetime.now().strftime('%Y-%m-%d')}

RELATIVE TO ABSOLUTE DATE MAPPING:
{date_context_str}

Node Context: {node_context_str}

You are a voice assistant command classifier. Identify which command the user wants.

COMMANDS & EXAMPLES:

{commands_str}

Always respond with JSON: {{"s":true,"n":"command_name","e":null}}"""

    def _build_parameter_extraction_prompt(self, command: Dict[str, Any], voice_command: str) -> str:
        """Build system prompt for parameter extraction"""
        # Format parameters
        params_str = []
        for param in command["parameters"]:
            param_line = f"- {param['name']} ({param['type']})"
            if param['required']:
                param_line += " [REQUIRED]"
            param_line += f": {param['description']}"
            if param.get('enum_values'):
                param_line += f" [Allowed values: {', '.join(param['enum_values'])}]"
            params_str.append(param_line)
        
        return f"""COMMAND DETAILS:

Command: {command['name']}
Description: {command['description']}

PARAMETERS:
{chr(10).join(params_str)}

🚨 CRITICAL: Return ONLY valid JSON. NO explanations, descriptions, or additional text.

Voice Command: {voice_command}

NOTE: Date parameters (datetime, date, time, timedelta) will be extracted separately. Focus on extracting non-date parameters that you can confidently identify from the voice command."""

    def _build_date_extraction_prompt(self, date_context: Dict[str, Any], voice_command: str) -> str:
        """Build system prompt for date extraction"""
        date_context_str = json.dumps(date_context, indent=2)
        
        return f"""🚨 CRITICAL: Return ONLY valid JSON. NO explanations, descriptions, or additional text.

Extract date parameters from the voice command and return ONLY valid JSON.

REQUIRED OUTPUT FORMAT:
{{"s":true,"p":{{...}},"e":null}}

DATE PARAMETERS TO EXTRACT:
- datetimes (array[datetime]): Array of ISO datetime strings to check for events.

CRITICAL DATE EXTRACTION INSTRUCTIONS:
- Look at the voice command and identify any relative date words
- Go to the RELATIVE TO ABSOLUTE DATE MAPPING section below
- Find the EXACT key that matches the relative date phrase
- Copy the COMPLETE value (array or single) from that mapping entry
- Put that value in the "datetimes" parameter

RELATIVE TO ABSOLUTE DATE MAPPING:
{date_context_str}

Voice Command: {voice_command}

Remember: ONLY JSON, no explanations. Look up the EXACT mapping value."""

class MultiTaskTrainer:
    """Trainer for multi-task Llama model"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Setup data generator
        self.data_generator = MultiTaskDataGenerator()
    
    def load_model_and_tokenizer(self):
        """Load Llama 3.2 1B Instruct model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Check if we need HF token for gated models
        try:
            # Try to load tokenizer first to test access
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
        except Exception as e:
            if "gated repo" in str(e) or "authentication" in str(e).lower():
                logger.error("❌ Model access requires authentication!")
                logger.error("Please ensure you have:")
                logger.error("1. Accepted the Llama 3.2 license at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
                logger.error("2. Run: huggingface-cli login")
                logger.error("3. Or set HF_TOKEN environment variable")
                
                # Try to prompt for token if not available
                try:
                    import os
                    if not os.getenv('HF_TOKEN'):
                        token = input("Please enter your HuggingFace token: ").strip()
                        if token:
                            os.environ['HF_TOKEN'] = token
                            from huggingface_hub import login
                            login(token=token)
                            logger.info("✅ HuggingFace authentication successful")
                            # Retry loading
                            self.tokenizer = AutoTokenizer.from_pretrained(
                                self.config.model_name,
                                trust_remote_code=True
                            )
                        else:
                            raise Exception("HuggingFace token required for this model")
                except KeyboardInterrupt:
                    raise Exception("Training cancelled by user")
            else:
                raise e
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with FP32 for stability
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,  # Force FP32
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"  # Disable FlashAttention2
        )
        
        logger.info(f"Model loaded with {self.model.num_parameters():,} parameters")
    
    def load_existing_date_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load existing date parsing training data"""
        logger.info(f"📚 Loading existing date training data from: {file_path}")
        
        date_examples = []
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        
                        # Convert existing format to multi-task format
                        # Your existing data has: {"input": "...", "output": "..."}
                        # Convert to our task format
                        date_example = {
                            "task": "date_extraction",
                            "input": f"System: Extract date from voice command\nUser: {data['input']}",
                            "output": data['output']
                        }
                        date_examples.append(date_example)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
                        continue
                        
        except FileNotFoundError:
            logger.warning(f"Date training file not found: {file_path}")
            return []
        
        logger.info(f"✅ Loaded {len(date_examples)} existing date examples")
        return date_examples

    def generate_training_data(self) -> List[Dict[str, Any]]:
        """Generate comprehensive multi-task training data"""
        logger.info("🎯 Generating multi-task training data...")
        
        all_examples = []
        
        # 1. Load existing date training data (high quality, proven)
        existing_date_file = "data/date_training_data_50anchors.jsonl"
        existing_date_examples = self.load_existing_date_data(existing_date_file)
        
        # Use existing date data if available, otherwise generate synthetic
        if existing_date_examples:
            logger.info(f"📚 Using {len(existing_date_examples)} existing date examples")
            # Limit to max_examples_per_task if we have too many
            if len(existing_date_examples) > self.config.max_examples_per_task:
                existing_date_examples = random.sample(existing_date_examples, self.config.max_examples_per_task)
                logger.info(f"📊 Sampled down to {len(existing_date_examples)} date examples")
            all_examples.extend(existing_date_examples)
        else:
            logger.info("📝 Generating synthetic date examples...")
            for _ in tqdm(range(self.config.max_examples_per_task), desc="Generating date_extraction"):
                try:
                    example = self.data_generator.generate_date_extraction_example()
                    all_examples.append(example)
                except Exception as e:
                    logger.warning(f"Failed to generate date_extraction example: {e}")
                    continue
        
        # 2. Generate command inference and parameter extraction examples
        for task_name, generator_func in [
            ("command_inference", self.data_generator.generate_command_inference_example),
            ("parameter_extraction", self.data_generator.generate_parameter_extraction_example)
        ]:
            logger.info(f"📝 Generating {self.config.max_examples_per_task} examples for {task_name}...")
            
            for _ in tqdm(range(self.config.max_examples_per_task), desc=f"Generating {task_name}"):
                try:
                    example = generator_func()
                    all_examples.append(example)
                except Exception as e:
                    logger.warning(f"Failed to generate {task_name} example: {e}")
                    continue
        
        # 3. Generate error/failure examples (10% of total)
        error_examples_count = int(len(all_examples) * 0.1)
        logger.info(f"📝 Generating {error_examples_count} error handling examples...")
        
        for _ in tqdm(range(error_examples_count), desc="Generating error_cases"):
            try:
                example = self.data_generator.generate_error_example()
                all_examples.append(example)
            except Exception as e:
                logger.warning(f"Failed to generate error example: {e}")
                continue
        
        # Shuffle all examples
        random.shuffle(all_examples)
        logger.info(f"✅ Generated {len(all_examples)} total training examples")
        
        return all_examples
    
    def prepare_dataset(self, examples: List[Dict[str, Any]]) -> Dataset:
        """Prepare dataset for training"""
        logger.info("Preparing multi-task dataset for training...")
        
        # Format examples for chat template
        formatted_examples = []
        skipped = 0
        
        for example in tqdm(examples, desc="Formatting examples"):
            try:
                # Create messages format
                messages = [
                    {"role": "user", "content": example["input"]},
                    {"role": "assistant", "content": example["output"]}
                ]
                
                # Apply chat template
                formatted_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                # Tokenize
                tokens = self.tokenizer(
                    formatted_text,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors=None,
                    padding=False
                )
                
                # Create training example
                tokenized_example = {
                    'input_ids': tokens['input_ids'],
                    'attention_mask': tokens['attention_mask'],
                    'labels': tokens['input_ids'].copy(),  # For causal LM
                    'task': example['task']  # Keep task info for analysis
                }
                
                formatted_examples.append(tokenized_example)
                
            except Exception as e:
                logger.warning(f"Failed to format example: {e}")
                skipped += 1
                continue
        
        logger.info(f"Formatted {len(formatted_examples)} examples, skipped {skipped}")
        
        # Create dataset
        dataset = Dataset.from_list(formatted_examples)
        return dataset
    
    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments for multi-task learning"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_dir=self.config.logging_dir,
            logging_steps=50,
            save_steps=self.config.save_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            dataloader_num_workers=2,
            report_to=[],  # Disable wandb
        )
    
    def train(self) -> Trainer:
        """Train the multi-task model"""
        logger.info("🚀 Starting multi-task Llama 3.2 1B fine-tuning...")
        
        # Generate training data
        examples = self.generate_training_data()
        
        # Prepare dataset
        dataset = self.prepare_dataset(examples)
        
        # Split for evaluation (90/10)
        logger.info("Splitting training data for evaluation (90/10 split)")
        train_test = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test['train']
        eval_dataset = train_test['test']
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Custom data collator for variable length sequences
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
        logger.info("🎯 Beginning multi-task training...")
        trainer.train()
        
        # Save final model
        logger.info("💾 Saving final multi-task model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("✅ Multi-task Llama 3.2 training completed!")
        logger.info(f"📁 Model saved to: {self.config.output_dir}")
        
        return trainer

def push_to_huggingface(
    model_path: str,
    repo_name: str,
    hf_token: str = None,
    private: bool = True,
    commit_message: str = "Upload Multi-Task Llama 3.2 1B Jarvis model v1"
):
    """Push model to HuggingFace Hub"""
    if not HF_AVAILABLE:
        logger.error("❌ huggingface_hub not available")
        return False
    
    logger.info(f"🚀 Pushing multi-task model to HF: {repo_name}")
    
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
- multi-task
- command-inference
- parameter-extraction
- date-parsing
- voice-assistant
- jarvis
language:
- en
pipeline_tag: text-generation
---

# {repo_name}

## Model Description

Multi-task fine-tuned Llama 3.2 1B Instruct model for voice assistant tasks. This model handles three distinct tasks:

1. **Command Inference**: Identifies which command the user wants to execute
2. **Parameter Extraction**: Extracts non-date parameters from voice commands
3. **Date Parsing**: Converts relative date expressions to ISO datetime strings

### Key Features:
- ✅ **Multi-task capability** - Single model handles all voice assistant inference tasks
- ✅ **Optimized for edge devices** - 1B parameters for Raspberry Pi deployment
- ✅ **JSON-structured output** - Consistent response format across all tasks
- ✅ **Context-aware** - Understands room, user, and temporal context

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
model = AutoModelForCausalLM.from_pretrained("{repo_name}")

# Command Inference
messages = [
    {{"role": "user", "content": "System: [command inference prompt]\\nUser: Turn on the lights"}}
]

# Parameter Extraction  
messages = [
    {{"role": "user", "content": "System: [parameter extraction prompt]\\nUser: Set temperature to 72"}}
]

# Date Parsing
messages = [
    {{"role": "user", "content": "System: [date parsing prompt]\\nUser: What meetings do I have next week?"}}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=100,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
```

## Training Data

The model was trained on ~45,000 synthetic examples:
- 15,000 command inference examples
- 15,000 parameter extraction examples  
- 15,000 date parsing examples

## Performance

Optimized for:
- **Low latency** on edge devices
- **High accuracy** across all three tasks
- **Consistent JSON output** format
- **Multi-domain command support**

## Deployment

Designed for deployment on:
- Raspberry Pi 4/5
- Edge computing devices
- Local voice assistants
- Privacy-focused applications
"""
        
        # Create README
        with open(f"{model_path}/README.md", "w") as f:
            f.write(model_card)
        
        # Push README
        api = HfApi()
        api.upload_file(
            path_or_fileobj=f"{model_path}/README.md",
            path_in_repo="README.md",
            repo_id=repo_name,
            commit_message="Add model card"
        )
        
        logger.info("✅ Model pushed to HuggingFace!")
        logger.info(f"🎉 Model uploaded: https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to push to HuggingFace: {e}")
        return False

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Multi-task Llama 3.2 1B fine-tuning")
    parser.add_argument("--push-to-hf", action="store_true", help="Push model to HuggingFace Hub")
    parser.add_argument("--repo-name", default="alexberardi/jarvis-multitask-llama-1b", help="HuggingFace repo name")
    parser.add_argument("--hf-token", help="HuggingFace token")
    parser.add_argument("--private", action="store_true", help="Make HF repo private")
    parser.add_argument("--public", action="store_true", help="Make HF repo public")
    parser.add_argument("--max-examples", type=int, default=15000, help="Max examples per task")
    
    args = parser.parse_args()
    
    # Setup configuration
    config = TrainingConfig()
    config.max_examples_per_task = args.max_examples
    
    # Adjust batch size based on available hardware
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"🔥 GPU detected: {gpu_name}")
        if "A100" in gpu_name:
            config.batch_size = 12  # A100 can handle larger batches
        elif "A40" in gpu_name:
            config.batch_size = 8
        else:
            config.batch_size = 6
    else:
        logger.info("🖥️  Using CPU")
        config.batch_size = 2
    
    logger.info(f"📊 Using batch size: {config.batch_size}")
    logger.info(f"📊 Max examples per task: {config.max_examples_per_task}")
    logger.info(f"📊 Total examples: {config.max_examples_per_task * 3}")
    
    # Initialize trainer
    trainer_instance = MultiTaskTrainer(config)
    
    # Train model
    trainer = trainer_instance.train()
    
    # Push to HuggingFace if requested
    if args.push_to_hf:
        private = args.private and not args.public  # Default to private unless --public specified
        push_to_huggingface(
            model_path=config.output_dir,
            repo_name=args.repo_name,
            hf_token=args.hf_token,
            private=private
        )
    
    logger.info("🎉 All done! Multi-task Llama 3.2 training complete!")

if __name__ == "__main__":
    main()

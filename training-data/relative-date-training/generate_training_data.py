"""
Comprehensive training data generator for relative date parsing.

Generates thousands of training examples with diverse anchor dates to teach
the model how to parse relative date expressions into UTC timestamps.
"""

import json
import random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pytz
import sys
import os

# Add parent directory to path to import from main app
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from date_output_schema import create_single_date_result, create_date_range_result
from app.core.general_context import generate_date_context_object


class DateTrainingDataGenerator:
    """Generates comprehensive training data for relative date parsing."""
    
    def __init__(self, start_year: int = 2020, end_year: int = 2030):
        self.start_year = start_year
        self.end_year = end_year
        self.timezones = [
            'UTC',
            'America/New_York',
            'America/Los_Angeles', 
            'America/Chicago',
            'Europe/London',
            'Europe/Paris',
            'Asia/Tokyo',
            'Australia/Sydney'
        ]
        
        # Comprehensive relative date expressions
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                 'July', 'August', 'September', 'October', 'November', 'December']
        
        self.relative_expressions = {
            # Single date expressions
            'single': [
                # Basic relative dates
                'today', 'tomorrow', 'yesterday',
                'day after tomorrow', 'day before yesterday',
                
                # All weekdays with next/last/this
                *[f'next {day}' for day in weekdays],
                *[f'last {day}' for day in weekdays], 
                *[f'this {day}' for day in weekdays],
                
                # Month expressions - skipping for now (edge cases)
                # *[f'next {month}' for month in months],
                # *[f'last {month}' for month in months],
                
                # Relative day offsets
                'in 1 day', 'in 2 days', 'in 3 days', 'in 4 days', 'in 5 days',
                'in a week', 'in 2 weeks', 'in 3 weeks', 'in a month',
                '1 day ago', '2 days ago', '3 days ago', '4 days ago', '5 days ago',
                'a week ago', '2 weeks ago', '3 weeks ago', 'a month ago',
            ],
            # Range expressions  
            'range': [
                'this weekend', 'next weekend', 'last weekend',
                'this week', 'next week', 'last week',
                'this month', 'next month', 'last month',
                'this year', 'next year', 'last year',
                
                # Extended ranges
                'the next 3 days', 'the next week', 'the next 2 weeks',
                'the last 3 days', 'the last week', 'the last 2 weeks',
            ]
        }
        
        # Note: Time expressions are now pre-calculated in the date context object
        # We'll get them from there instead of hardcoding here

    def generate_anchor_dates(self, count: int = 5000) -> List[datetime]:
        """Generate diverse anchor dates across the specified year range."""
        anchor_dates = []
        
        start_date = datetime(self.start_year, 1, 1)
        end_date = datetime(self.end_year, 12, 31)
        
        # Generate random dates across the range
        total_days = (end_date - start_date).days
        
        for _ in range(count):
            random_days = random.randint(0, total_days)
            anchor_date = start_date + timedelta(days=random_days)
            
            # Add some random time variation (not just midnight)
            random_hour = random.randint(0, 23)
            random_minute = random.randint(0, 59)
            
            anchor_date = anchor_date.replace(
                hour=random_hour, 
                minute=random_minute, 
                second=0, 
                microsecond=0
            )
            
            anchor_dates.append(anchor_date)
        
        return anchor_dates

    def parse_single_expression(self, expression: str, anchor_date: datetime, tz: pytz.BaseTzInfo) -> str:
        """Parse a single date expression relative to anchor date."""
        # Localize anchor date to timezone
        local_anchor = tz.localize(anchor_date.replace(tzinfo=None))
        
        if expression == 'today':
            result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0)
        elif expression == 'tomorrow':
            result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        elif expression == 'yesterday':
            result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        elif expression == 'day after tomorrow':
            result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=2)
        elif expression == 'day before yesterday':
            result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=2)
        elif expression.startswith('next '):
            weekday = expression.split()[1].lower()
            result = self._get_next_weekday(local_anchor, weekday)
        elif expression.startswith('last '):
            weekday = expression.split()[1].lower()
            result = self._get_last_weekday(local_anchor, weekday)
        elif expression.startswith('this '):
            weekday = expression.split()[1].lower()
            result = self._get_this_weekday(local_anchor, weekday)
        elif 'day ago' in expression or 'days ago' in expression:
            # Parse "X days ago"
            parts = expression.split()
            if parts[0].isdigit():
                days = int(parts[0])
                result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)
            elif parts[0] == 'a':
                result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            else:
                result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0)
        elif 'in' in expression and ('day' in expression or 'days' in expression):
            # Parse "in X days"
            parts = expression.split()
            if len(parts) >= 2 and parts[1].isdigit():
                days = int(parts[1])
                result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days)
            elif 'in a day' in expression:
                result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0)
        elif 'week ago' in expression or 'weeks ago' in expression:
            # Parse "X weeks ago"
            if 'a week ago' in expression:
                result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(weeks=1)
            elif expression.startswith('2 weeks ago'):
                result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(weeks=2)
            elif expression.startswith('3 weeks ago'):
                result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(weeks=3)
            else:
                result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0)
        elif 'in' in expression and ('week' in expression or 'weeks' in expression):
            # Parse "in X weeks"
            if 'in a week' in expression:
                result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(weeks=1)
            elif 'in 2 weeks' in expression:
                result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(weeks=2)
            elif 'in 3 weeks' in expression:
                result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(weeks=3)
            else:
                result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0)
        # Time expressions are now handled via pre-calculated date context
        # This will be processed in the generate_training_examples method
        else:
            # Default to start of day for unrecognized expressions
            result = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Convert to UTC
        return result.astimezone(timezone.utc).isoformat()

    def parse_range_expression(self, expression: str, anchor_date: datetime, tz: pytz.BaseTzInfo) -> Tuple[str, str]:
        """Parse a date range expression relative to anchor date."""
        # Localize anchor date to timezone
        local_anchor = tz.localize(anchor_date.replace(tzinfo=None))
        
        if expression == 'this weekend':
            # Saturday and Sunday of current week
            days_until_saturday = (5 - local_anchor.weekday()) % 7
            saturday = local_anchor + timedelta(days=days_until_saturday)
            sunday = saturday + timedelta(days=1)
            start = saturday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = sunday.replace(hour=23, minute=59, second=59, microsecond=0)
            
        elif expression == 'next weekend':
            # Saturday and Sunday of next week
            days_until_saturday = (5 - local_anchor.weekday()) % 7 + 7
            saturday = local_anchor + timedelta(days=days_until_saturday)
            sunday = saturday + timedelta(days=1)
            start = saturday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = sunday.replace(hour=23, minute=59, second=59, microsecond=0)
            
        elif expression == 'last weekend':
            # Saturday and Sunday of last week
            days_until_saturday = (5 - local_anchor.weekday()) % 7 - 7
            saturday = local_anchor + timedelta(days=days_until_saturday)
            sunday = saturday + timedelta(days=1)
            start = saturday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = sunday.replace(hour=23, minute=59, second=59, microsecond=0)
            
        elif expression == 'this week':
            # Monday to Sunday of current week
            days_since_monday = local_anchor.weekday()
            monday = local_anchor - timedelta(days=days_since_monday)
            sunday = monday + timedelta(days=6)
            start = monday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = sunday.replace(hour=23, minute=59, second=59, microsecond=0)
            
        elif expression == 'next week':
            # Monday to Sunday of next week
            days_since_monday = local_anchor.weekday()
            next_monday = local_anchor - timedelta(days=days_since_monday) + timedelta(days=7)
            next_sunday = next_monday + timedelta(days=6)
            start = next_monday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = next_sunday.replace(hour=23, minute=59, second=59, microsecond=0)
            
        elif expression == 'last week':
            # Monday to Sunday of last week
            days_since_monday = local_anchor.weekday()
            last_monday = local_anchor - timedelta(days=days_since_monday) - timedelta(days=7)
            last_sunday = last_monday + timedelta(days=6)
            start = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = last_sunday.replace(hour=23, minute=59, second=59, microsecond=0)
            
        elif expression == 'this month':
            # First and last day of current month
            start = local_anchor.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if local_anchor.month == 12:
                next_month = local_anchor.replace(year=local_anchor.year + 1, month=1, day=1)
            else:
                next_month = local_anchor.replace(month=local_anchor.month + 1, day=1)
            end = (next_month - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
            
        elif expression == 'next month':
            # First and last day of next month
            if local_anchor.month == 12:
                next_month_start = local_anchor.replace(year=local_anchor.year + 1, month=1, day=1)
                next_month_end = local_anchor.replace(year=local_anchor.year + 1, month=2, day=1) - timedelta(days=1)
            else:
                next_month_start = local_anchor.replace(month=local_anchor.month + 1, day=1)
                if local_anchor.month == 11:
                    next_month_end = local_anchor.replace(year=local_anchor.year + 1, month=1, day=1) - timedelta(days=1)
                else:
                    next_month_end = local_anchor.replace(month=local_anchor.month + 2, day=1) - timedelta(days=1)
            
            start = next_month_start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = next_month_end.replace(hour=23, minute=59, second=59, microsecond=0)
            
        elif expression == 'last month':
            # First and last day of last month
            if local_anchor.month == 1:
                last_month_start = local_anchor.replace(year=local_anchor.year - 1, month=12, day=1)
                last_month_end = local_anchor.replace(year=local_anchor.year - 1, month=12, day=31)
            else:
                last_month_start = local_anchor.replace(month=local_anchor.month - 1, day=1)
                last_month_end = local_anchor.replace(day=1) - timedelta(days=1)
            
            start = last_month_start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = last_month_end.replace(hour=23, minute=59, second=59, microsecond=0)
            
        elif expression == 'this year':
            # First and last day of current year
            start = local_anchor.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end = local_anchor.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=0)
            
        elif expression == 'next year':
            # First and last day of next year
            next_year = local_anchor.year + 1
            start = local_anchor.replace(year=next_year, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end = local_anchor.replace(year=next_year, month=12, day=31, hour=23, minute=59, second=59, microsecond=0)
            
        elif expression == 'last year':
            # First and last day of last year
            last_year = local_anchor.year - 1
            start = local_anchor.replace(year=last_year, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end = local_anchor.replace(year=last_year, month=12, day=31, hour=23, minute=59, second=59, microsecond=0)
            
        else:
            # Default to single day
            start = local_anchor.replace(hour=0, minute=0, second=0, microsecond=0)
            end = local_anchor.replace(hour=23, minute=59, second=59, microsecond=0)
        
        # Convert to UTC
        start_utc = start.astimezone(timezone.utc).isoformat()
        end_utc = end.astimezone(timezone.utc).isoformat()
        
        return start_utc, end_utc

    def _get_next_weekday(self, anchor_date: datetime, weekday: str) -> datetime:
        """Get the next occurrence of a weekday."""
        weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        target_weekday = weekdays.index(weekday)
        
        days_ahead = target_weekday - anchor_date.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
            
        result = anchor_date + timedelta(days=days_ahead)
        return result.replace(hour=0, minute=0, second=0, microsecond=0)

    def _get_last_weekday(self, anchor_date: datetime, weekday: str) -> datetime:
        """Get the last occurrence of a weekday."""
        weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        target_weekday = weekdays.index(weekday)
        
        days_back = anchor_date.weekday() - target_weekday
        if days_back <= 0:  # Target day hasn't happened this week
            days_back += 7
            
        result = anchor_date - timedelta(days=days_back)
        return result.replace(hour=0, minute=0, second=0, microsecond=0)

    def _get_this_weekday(self, anchor_date: datetime, weekday: str) -> datetime:
        """Get this week's occurrence of a weekday."""
        weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        target_weekday = weekdays.index(weekday)
        
        days_diff = target_weekday - anchor_date.weekday()
        result = anchor_date + timedelta(days=days_diff)
        return result.replace(hour=0, minute=0, second=0, microsecond=0)

    def generate_training_examples(self, anchor_dates: List[datetime]) -> List[Dict[str, Any]]:
        """Generate comprehensive training examples for all anchor dates."""
        training_examples = []
        
        for anchor_date in anchor_dates:
            # Generate examples for each timezone
            for tz_name in self.timezones:
                tz = pytz.timezone(tz_name)
                
                # Generate single date examples
                for expression in self.relative_expressions['single']:
                    try:
                        result_date = self.parse_single_expression(expression, anchor_date, tz)
                        
                        # Create training example
                        example = {
                            "input": f"Current Date: {anchor_date.strftime('%Y-%m-%d %H:%M:%S')}\nTimezone: {tz_name}\nExpression: '{expression}'",
                            "output": json.dumps(create_single_date_result(result_date).model_dump())
                        }
                        training_examples.append(example)
                    except Exception as e:
                        print(f"Error processing single expression '{expression}' for {anchor_date} in {tz_name}: {e}")
                
                # Generate range examples
                for expression in self.relative_expressions['range']:
                    try:
                        start_date, end_date = self.parse_range_expression(expression, anchor_date, tz)
                        
                        # Create training example
                        example = {
                            "input": f"Current Date: {anchor_date.strftime('%Y-%m-%d %H:%M:%S')}\nTimezone: {tz_name}\nExpression: '{expression}'",
                            "output": json.dumps(create_date_range_result(start_date, end_date).model_dump())
                        }
                        training_examples.append(example)
                    except Exception as e:
                        print(f"Error processing range expression '{expression}' for {anchor_date} in {tz_name}: {e}")
                
                # Generate time-specific examples using pre-calculated date context
                try:
                    # Generate date context for this anchor date and timezone
                    # We need to create a timezone-aware datetime for the context generation
                    if tz_name == 'UTC':
                        context_anchor = anchor_date.replace(tzinfo=pytz.UTC)
                    else:
                        context_anchor = tz.localize(anchor_date.replace(tzinfo=None))
                    
                    # Generate date context (this will include time_expressions)
                    date_context = generate_date_context_object(tz_name)
                    time_expressions = date_context.get('time_expressions', {})
                    
                    # Generate examples for each time expression
                    for expression, utc_timestamp in time_expressions.items():
                        example = {
                            "input": f"Current Date: {anchor_date.strftime('%Y-%m-%d %H:%M:%S')}\nTimezone: {tz_name}\nExpression: '{expression}'",
                            "output": json.dumps(create_single_date_result(utc_timestamp).model_dump())
                        }
                        training_examples.append(example)
                        
                except Exception as e:
                    print(f"Error processing time expressions for {anchor_date} in {tz_name}: {e}")
        
        return training_examples

    def generate_and_save(self, output_file: str = "date_training_data.jsonl", num_anchors: int = 5000):
        """Generate training data and save to file."""
        print(f"Generating {num_anchors} anchor dates...")
        anchor_dates = self.generate_anchor_dates(num_anchors)
        
        print(f"Generating training examples for {len(anchor_dates)} anchor dates across {len(self.timezones)} timezones...")
        training_examples = self.generate_training_examples(anchor_dates)
        
        print(f"Generated {len(training_examples)} training examples")
        
        # Save to JSONL format
        output_path = Path(__file__).parent / output_file
        with open(output_path, 'w') as f:
            for example in training_examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"Training data saved to {output_path}")
        
        # Generate some stats
        self._print_stats(training_examples)
        
        return training_examples

    def _print_stats(self, training_examples: List[Dict[str, Any]]):
        """Print statistics about generated training data."""
        print("\n" + "="*50)
        print("TRAINING DATA STATISTICS")
        print("="*50)
        print(f"Total examples: {len(training_examples):,}")
        
        # Count by expression type
        single_count = sum(1 for ex in training_examples if '"result_type": "single"' in ex['output'])
        range_count = sum(1 for ex in training_examples if '"result_type": "range"' in ex['output'])
        
        print(f"Single date examples: {single_count:,}")
        print(f"Date range examples: {range_count:,}")
        
        # Estimate token count (rough)
        total_chars = sum(len(ex['input']) + len(ex['output']) for ex in training_examples)
        estimated_tokens = total_chars // 4  # Rough estimate: 4 chars per token
        
        print(f"Estimated total tokens: {estimated_tokens:,}")
        print(f"Average tokens per example: {estimated_tokens // len(training_examples)}")
        
        print("="*50)


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments for dataset size
    num_anchors = 100  # Default to small test set
    if len(sys.argv) > 1:
        try:
            num_anchors = int(sys.argv[1])
        except ValueError:
            print("Usage: python generate_training_data.py [num_anchors]")
            print("Example: python generate_training_data.py 10000")
            sys.exit(1)
    
    print(f"Generating training data with {num_anchors} anchor dates...")
    
    # Generate training data
    generator = DateTrainingDataGenerator()
    training_data = generator.generate_and_save(
        num_anchors=num_anchors,
        output_file=f"date_training_data_{num_anchors}anchors.jsonl"
    )
    
    # Show a few examples for inspection
    print(f"\nSample training examples (first 5 of {len(training_data)}):")
    for i, example in enumerate(training_data[:5]):
        print(f"\n--- Example {i+1} ---")
        print(f"Input: {example['input']}")
        print(f"Output: {example['output']}")
        
    print(f"\n✅ Training data generation complete!")
    print(f"📁 File: date_training_data_{num_anchors}anchors.jsonl")
    print(f"📊 Total examples: {len(training_data):,}")
    
    # Count expression types
    single_count = sum(1 for ex in training_data if '"result_type": "single"' in ex['output'])
    range_count = sum(1 for ex in training_data if '"result_type": "range"' in ex['output'])
    print(f"📅 Single dates: {single_count:,}")
    print(f"📅 Date ranges: {range_count:,}")
    
    # Estimate tokens
    total_chars = sum(len(ex['input']) + len(ex['output']) for ex in training_data)
    estimated_tokens = total_chars // 4
    print(f"🔢 Estimated tokens: {estimated_tokens:,}")
    print(f"🔢 Avg tokens/example: {estimated_tokens // len(training_data)}")

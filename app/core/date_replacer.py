#!/usr/bin/env python3
"""
Date replacement module for voice commands.
Converts relative date references to absolute dates using the date context.
"""

import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class DateReplacer:
    """Handles replacement of relative date references with absolute dates in voice commands."""
    
    def __init__(self, date_context: Dict):
        """
        Initialize with date context.
        
        Args:
            date_context: The date context object from generate_date_context_object
        """
        self.date_context = date_context
        
    def replace_dates_in_command(self, voice_command: str) -> Tuple[str, str]:
        """
        Replace relative date references with absolute dates.
        
        Args:
            voice_command: The original voice command
            
        Returns:
            Tuple of (cleaned_command, description) where:
            - cleaned_command: Command with dates replaced
            - description: Human-readable description of what was replaced
        """
        original_command = voice_command
        detected_dates = []
        
        # Detect weekend references
        weekend_dates = self._detect_weekend_references(voice_command)
        detected_dates.extend(weekend_dates)
        
        # Detect week references
        week_dates = self._detect_week_references(voice_command)
        detected_dates.extend(week_dates)
        
        # Detect specific weekday references
        weekday_dates = self._detect_weekday_references(voice_command)
        detected_dates.extend(weekday_dates)
        
        # Detect relative date references
        relative_dates = self._detect_relative_dates(voice_command)
        detected_dates.extend(relative_dates)
        
        # Build the final command with DETECTED DATES section
        if detected_dates:
            detected_dates_section = "\n\nDETECTED DATES:\n" + "\n".join(detected_dates)
            final_command = voice_command + detected_dates_section
            description = f"Date detection: {len(detected_dates)} dates found"
        else:
            final_command = voice_command
            description = "No dates detected"
            
        return final_command, description
    
    def _detect_weekend_references(self, command: str) -> List[str]:
        """Detect weekend references like 'this weekend', 'next weekend', 'last weekend'."""
        detected_dates = []
        
        # Weekend patterns
        weekend_patterns = {
            r'\bthis weekend\b': ('This weekend', self.date_context['weekend']['this_weekend']),
            r'\bnext weekend\b': ('Next weekend', self.date_context['weekend']['next_weekend']),
            r'\blast weekend\b': ('Last weekend', self.date_context['weekend']['last_weekend'])
        }
        
        for pattern, (label, weekend_data) in weekend_patterns.items():
            if re.search(pattern, command, re.IGNORECASE):
                # Format as array of dates with timestamps
                dates_with_timestamps = [f"'{day['date']} {day['utc_start_of_day']}'" for day in weekend_data]
                detected_dates.append(f"{label}: [{', '.join(dates_with_timestamps)}]")
        
        return detected_dates
    
    def _detect_week_references(self, command: str) -> List[str]:
        """Detect week references like 'this week', 'next week', 'last week'."""
        detected_dates = []
        
        # Week patterns
        week_patterns = {
            r'\bthis week\b': ('This week', self.date_context['weeks']['this_week']),
            r'\bnext week\b': ('Next week', self.date_context['weeks']['next_week']),
            r'\blast week\b': ('Last week', self.date_context['weeks']['last_week'])
        }
        
        for pattern, (label, week_data) in week_patterns.items():
            if re.search(pattern, command, re.IGNORECASE):
                # Format as array of dates with timestamps
                dates_with_timestamps = [f"'{day['date']} {day['utc_start_of_day']}'" for day in week_data]
                detected_dates.append(f"{label}: [{', '.join(dates_with_timestamps)}]")
        
        return detected_dates
    
    def _detect_weekday_references(self, command: str) -> List[str]:
        """Detect specific weekday references like 'next monday', 'last tuesday'."""
        detected_dates = []
        
        # Check for "next" weekday patterns
        next_pattern = r'\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
        next_matches = re.finditer(next_pattern, command, re.IGNORECASE)
        
        for match in next_matches:
            weekday = match.group(1).lower()
            # Find the matching day in next_week
            for day_data in self.date_context['weeks']['next_week']:
                if day_data['day'].lower().endswith(weekday.lower()):
                    detected_dates.append(f"Next {weekday.title()}: {day_data['date']} {day_data['utc_start_of_day']}")
                    break
        
        # Check for "last" weekday patterns
        last_pattern = r'\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
        last_matches = re.finditer(last_pattern, command, re.IGNORECASE)
        
        for match in last_matches:
            weekday = match.group(1).lower()
            # Find the matching day in last_week
            for day_data in self.date_context['weeks']['last_week']:
                if day_data['day'].lower().endswith(weekday.lower()):
                    detected_dates.append(f"Last {weekday.title()}: {day_data['date']} {day_data['utc_start_of_day']}")
                    break
        
        # Check for "this" weekday patterns (current week)
        this_pattern = r'\bthis\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
        this_matches = re.finditer(this_pattern, command, re.IGNORECASE)
        
        for match in this_matches:
            weekday = match.group(1).lower()
            # Find the matching day in this_week
            for day_data in self.date_context['weeks']['this_week']:
                if day_data['day'].lower() == weekday.lower():
                    detected_dates.append(f"This {weekday.title()}: {day_data['date']} {day_data['utc_start_of_day']}")
                    break
        
        return detected_dates
    
    def _detect_relative_dates(self, command: str) -> List[str]:
        """Detect relative date references like 'today', 'tomorrow', 'yesterday'."""
        detected_dates = []
        
        # Relative date patterns
        relative_patterns = {
            r'\btoday\b': ('Today', self.date_context['current']),
            r'\btomorrow\b': ('Tomorrow', self.date_context['relative_dates']['tomorrow']),
            r'\byesterday\b': ('Yesterday', self.date_context['relative_dates']['yesterday'])
        }
        
        for pattern, (label, date_data) in relative_patterns.items():
            if re.search(pattern, command, re.IGNORECASE):
                if isinstance(date_data, dict) and 'utc_start_of_day' in date_data:
                    detected_dates.append(f"{label}: {date_data['date']} {date_data['utc_start_of_day']}")
                else:
                    # Handle nested structure for current date
                    detected_dates.append(f"{label}: {date_data['date_iso']} {date_data['utc_start_of_day']}")
        
        return detected_dates

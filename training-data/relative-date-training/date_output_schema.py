"""
Date parsing output schema for training data generation.
Defines the structure of what our fine-tuned model should output.
"""

from pydantic import BaseModel
from typing import Optional, Union
from datetime import datetime


class SingleDateOutput(BaseModel):
    """Output for single date expressions like 'tomorrow', 'next Tuesday'"""
    date: str  # ISO format UTC datetime string
    

class DateRangeOutput(BaseModel):
    """Output for date range expressions like 'this weekend', 'next month'"""
    start_date: str  # ISO format UTC datetime string
    end_date: str    # ISO format UTC datetime string


class DateParsingResult(BaseModel):
    """
    Union type for date parsing results.
    Can be either a single date or a date range.
    """
    result: Union[SingleDateOutput, DateRangeOutput]
    result_type: str  # "single" or "range"


def create_single_date_result(date_str: str) -> DateParsingResult:
    """Helper to create single date result"""
    return DateParsingResult(
        result=SingleDateOutput(date=date_str),
        result_type="single"
    )


def create_date_range_result(start_date: str, end_date: str) -> DateParsingResult:
    """Helper to create date range result"""
    return DateParsingResult(
        result=DateRangeOutput(start_date=start_date, end_date=end_date),
        result_type="range"
    )

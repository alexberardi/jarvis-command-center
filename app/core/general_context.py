from datetime import datetime, timedelta
import pytz
import logging

logger = logging.getLogger("uvicorn")

def generate_date_context_object(timezone_str: str = None) -> dict:
    """
    Generate a comprehensive date context object with all calculated dates.
    
    Args:
        timezone_str: Optional timezone string (e.g., "America/New_York")
    
    Returns:
        dict: Object containing all calculated dates and timezone information
    """
    # Get current date/time for context
    logger.info(f"📅 Generating comprehensive date context object with timezone: {timezone_str}")
    
    if timezone_str:
        try:
            # Convert to user's timezone
            user_tz = pytz.timezone(timezone_str)
            utc_now = datetime.now(pytz.UTC)
            now = utc_now.astimezone(user_tz)
            logger.info(f"📅 Successfully converted to {timezone_str}: {now}")
        except Exception as e:
            logger.error(f"❌ Timezone conversion failed: {e}, falling back to local time")
            now = datetime.now()
    else:
        logger.info(f"📅 No timezone provided, using local time")
        now = datetime.now()
    
    # Helper function to get UTC start of day for a given date in user's timezone
    def get_utc_start_of_day(date_obj, user_tz_str):
        if not user_tz_str:
            return date_obj.strftime("%Y-%m-%dT00:00:00")
        
        try:
            user_tz = pytz.timezone(user_tz_str)
            # Create datetime at start of day in user's timezone
            start_of_day = date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Handle timezone-aware vs naive datetimes
            if start_of_day.tzinfo is None:
                # Naive datetime - localize it
                utc_start = user_tz.localize(start_of_day).astimezone(pytz.UTC)
            else:
                # Already timezone-aware - just convert to UTC
                utc_start = start_of_day.astimezone(pytz.UTC)
            
            return utc_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            logger.error(f"❌ Error calculating UTC start of day: {e}")
            return date_obj.strftime("%Y-%m-%dT00:00:00")
    
    # Calculate all the date variations
    tomorrow = now + timedelta(days=1)
    yesterday = now - timedelta(days=1)
    
    # Calculate last_night (yesterday at 7pm local time)
    last_night = yesterday.replace(hour=19, minute=0, second=0, microsecond=0)
    if timezone_str:
        try:
            user_tz = pytz.timezone(timezone_str)
            # Handle timezone-aware vs naive datetimes
            if last_night.tzinfo is None:
                # Naive datetime - localize it
                last_night_utc = user_tz.localize(last_night).astimezone(pytz.UTC)
            else:
                # Already timezone-aware - just convert to UTC
                last_night_utc = last_night.astimezone(pytz.UTC)
            last_night_iso = last_night_utc.isoformat()
        except Exception as e:
            logger.error(f"❌ Error calculating last_night UTC: {e}")
            last_night_iso = last_night.isoformat()
    else:
        last_night_iso = last_night.isoformat()
    
    # Calculate weekend dates (Saturday and Sunday)
    # now.weekday() returns 0 for Monday, 5 for Saturday, 6 for Sunday
    if now.weekday() == 5:  # Today is Saturday
        # This weekend is today and tomorrow
        this_saturday = now
        this_sunday = now + timedelta(days=1)
        # Last weekend is 6 and 7 days ago
        last_saturday = now - timedelta(days=7)
        last_sunday = now - timedelta(days=6)
        # Next weekend is 7 and 8 days from now
        next_saturday = now + timedelta(days=7)
        next_sunday = now + timedelta(days=8)
    elif now.weekday() == 6:  # Today is Sunday
        # This weekend is yesterday and today
        this_saturday = now - timedelta(days=1)
        this_sunday = now
        # Last weekend is 8 and 7 days ago
        last_saturday = now - timedelta(days=8)
        last_sunday = now - timedelta(days=7)
        # Next weekend is 6 and 7 days from now
        next_saturday = now + timedelta(days=6)
        next_sunday = now + timedelta(days=7)
    else:  # Today is Monday-Friday
        # This weekend is the next Saturday and Sunday
        days_until_saturday = (5 - now.weekday()) % 7
        this_saturday = now + timedelta(days=days_until_saturday)
        this_sunday = this_saturday + timedelta(days=1)
        # Last weekend is the previous Saturday and Sunday
        days_since_saturday = now.weekday() + 2  # Monday=0, so Monday is 2 days after Saturday
        last_saturday = now - timedelta(days=days_since_saturday)
        last_sunday = last_saturday + timedelta(days=1)
        # Next weekend is 7 days after this weekend
        next_saturday = this_saturday + timedelta(days=7)
        next_sunday = this_sunday + timedelta(days=7)
    
    # Calculate this week dates (Sunday to Saturday) - the week containing today
    if now.weekday() == 6:  # Today is Sunday
        this_week_start = now
    else:
        # Go back to the Sunday of this week
        this_week_start = now - timedelta(days=now.weekday() + 1)
    this_week_end = this_week_start + timedelta(days=6)
    
    # Calculate next week dates (Sunday to Saturday)
    next_week_start = this_week_start + timedelta(days=7)
    next_week_end = next_week_start + timedelta(days=6)
    
    # Calculate last week dates (Sunday to Saturday)
    last_week_start = this_week_start - timedelta(days=7)
    last_week_end = last_week_start + timedelta(days=6)
    
    # Calculate last month and last year
    last_month = now - timedelta(days=30)
    last_year = now - timedelta(days=365)
    
    # Calculate specific weekdays using week boundaries
    weekday_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    
    # Calculate next weekdays (from next week)
    next_weekdays = {}
    for i, day in enumerate(weekday_names):
        next_weekday_date = next_week_start + timedelta(days=i)
        next_weekdays[f"next_{day}"] = next_weekday_date.strftime("%Y-%m-%d")
    
    # Calculate last weekdays (from last week)
    last_weekdays = {}
    for i, day in enumerate(weekday_names):
        last_weekday_date = last_week_start + timedelta(days=i)
        last_weekdays[f"last_{day}"] = last_weekday_date.strftime("%Y-%m-%d")
    
    # Build the comprehensive date context object with times
    date_context = {
        "current": {
            "date": now.strftime("%A, %B %d %Y"),
            "date_iso": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%I:%M %p"),
            "datetime": now.isoformat(),
            "weekday": now.strftime("%A").lower(),
            "weekday_number": now.weekday(),
            "utc_start_of_day": get_utc_start_of_day(now, timezone_str)
        },
        "relative_dates": {
            "tomorrow": {
                "date": tomorrow.strftime("%Y-%m-%d"),
                "utc_start_of_day": get_utc_start_of_day(tomorrow, timezone_str)
            },
            "yesterday": {
                "date": yesterday.strftime("%Y-%m-%d"),
                "utc_start_of_day": get_utc_start_of_day(yesterday, timezone_str)
            },
            "last_night": {
                "date": yesterday.strftime("%Y-%m-%d"),
                "time": "19:00:00",
                "datetime": last_night_iso
            },
            "day_after_tomorrow": {
                "date": (now + timedelta(days=2)).strftime("%Y-%m-%d"),
                "utc_start_of_day": get_utc_start_of_day(now + timedelta(days=2), timezone_str)
            },
            "day_before_yesterday": {
                "date": (now - timedelta(days=2)).strftime("%Y-%m-%d"),
                "utc_start_of_day": get_utc_start_of_day(now - timedelta(days=2), timezone_str)
            }
        },
        "weekend": {
            "this_weekend": [
                {
                    "day": "Saturday",
                    "date": this_saturday.strftime("%Y-%m-%d"),
                    "utc_start_of_day": get_utc_start_of_day(this_saturday, timezone_str)
                },
                {
                    "day": "Sunday",
                    "date": this_sunday.strftime("%Y-%m-%d"),
                    "utc_start_of_day": get_utc_start_of_day(this_sunday, timezone_str)
                }
            ],
            "next_weekend": [
                {
                    "day": "Saturday",
                    "date": next_saturday.strftime("%Y-%m-%d"),
                    "utc_start_of_day": get_utc_start_of_day(next_saturday, timezone_str)
                },
                {
                    "day": "Sunday",
                    "date": next_sunday.strftime("%Y-%m-%d"),
                    "utc_start_of_day": get_utc_start_of_day(next_sunday, timezone_str)
                }
            ],
            "last_weekend": [
                {
                    "day": "Saturday",
                    "date": last_saturday.strftime("%Y-%m-%d"),
                    "date": last_saturday.strftime("%Y-%m-%d"),
                    "utc_start_of_day": get_utc_start_of_day(last_saturday, timezone_str)
                },
                {
                    "day": "Sunday",
                    "date": last_sunday.strftime("%Y-%m-%d"),
                    "utc_start_of_day": get_utc_start_of_day(last_sunday, timezone_str)
                }
            ]
        },
        "weeks": {
            "this_week": [
                {
                    "day": (this_week_start + timedelta(days=i)).strftime("%A"),
                    "date": (this_week_start + timedelta(days=i)).strftime("%Y-%m-%d"), 
                    "utc_start_of_day": get_utc_start_of_day(this_week_start + timedelta(days=i), timezone_str)
                }
                for i in range(7)
            ],
            "next_week": [
                {
                    "day": f"Next {(next_week_start + timedelta(days=i)).strftime('%A')}",
                    "date": (next_week_start + timedelta(days=i)).strftime("%Y-%m-%d"), 
                    "utc_start_of_day": get_utc_start_of_day(next_week_start + timedelta(days=i), timezone_str)
                }
                for i in range(7)
            ],
            "last_week": [
                {
                    "day": f"Last {(last_week_start + timedelta(days=i)).strftime('%A')}",
                    "date": (last_week_start + timedelta(days=i)).strftime("%Y-%m-%d"), 
                    "utc_start_of_day": get_utc_start_of_day(last_week_start + timedelta(days=i), timezone_str)
                }
                for i in range(7)
            ]
        },
        "months": {
            "this_month": [
                {
                    "date": now.replace(day=1).strftime("%Y-%m-%d"),
                    "utc_start_of_day": get_utc_start_of_day(now.replace(day=1), timezone_str)
                },
                {
                    "date": ((now.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "utc_start_of_day": get_utc_start_of_day(
                        (now.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1),
                        timezone_str
                    )
                }
            ],
            "next_month": [
                {
                    "date": (now.replace(day=1) + timedelta(days=32)).replace(day=1).strftime("%Y-%m-%d"),
                    "utc_start_of_day": get_utc_start_of_day(
                        (now.replace(day=1) + timedelta(days=32)).replace(day=1),
                        timezone_str
                    )
                },
                {
                    "date": (((now.replace(day=1) + timedelta(days=32)).replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "utc_start_of_day": get_utc_start_of_day(
                        ((now.replace(day=1) + timedelta(days=32)).replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1),
                        timezone_str
                    )
                }
            ],
            "last_month": [
                {
                    "date": ((now.replace(day=1) - timedelta(days=1)).replace(day=1)).strftime("%Y-%m-%d"),
                    "utc_start_of_day": get_utc_start_of_day((now.replace(day=1) - timedelta(days=1)).replace(day=1), timezone_str)
                },
                {
                    "date": (now.replace(day=1) - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "utc_start_of_day": get_utc_start_of_day(now.replace(day=1) - timedelta(days=1), timezone_str)
                }
            ]
        },
        "years": {
            "this_year": [
                {
                    "date": now.replace(month=1, day=1).strftime("%Y-%m-%d"),
                    "utc_start_of_day": get_utc_start_of_day(now.replace(month=1, day=1), timezone_str)
                },
                {
                    "date": now.replace(month=12, day=31).strftime("%Y-%m-%d"),
                    "utc_start_of_day": get_utc_start_of_day(now.replace(month=12, day=31), timezone_str)
                }
            ],
            "next_year": [
                {
                    "date": (now + timedelta(days=365)).replace(month=1, day=1).strftime("%Y-%m-%d"),
                    "utc_start_of_day": get_utc_start_of_day((now + timedelta(days=365)).replace(month=1, day=1), timezone_str)
                },
                {
                    "date": (now + timedelta(days=365)).replace(month=12, day=31).strftime("%Y-%m-%d"),
                    "utc_start_of_day": get_utc_start_of_day((now + timedelta(days=365)).replace(month=12, day=31), timezone_str)
                }
            ],
            "last_year": [
                {
                    "date": (now - timedelta(days=365)).replace(month=1, day=1).strftime("%Y-%m-%d"),
                    "utc_start_of_day": get_utc_start_of_day((now - timedelta(days=365)).replace(month=1, day=1), timezone_str)
                },
                {
                    "date": (now - timedelta(days=365)).replace(month=12, day=31).strftime("%Y-%m-%d"),
                    "utc_start_of_day": get_utc_start_of_day((now - timedelta(days=365)).replace(month=12, day=31), timezone_str)
                }
            ]
        },
        "weekdays": {
            **{k: {"date": v, "utc_start_of_day": get_utc_start_of_day(datetime.strptime(v, "%Y-%m-%d"), timezone_str)} for k, v in next_weekdays.items()},
            **{k: {"date": v, "utc_start_of_day": get_utc_start_of_day(datetime.strptime(v, "%Y-%m-%d"), timezone_str)} for k, v in last_weekdays.items()}
        },
        "timezone": {
            "user_timezone": timezone_str,
            "current_timezone": str(now.tzinfo) if now.tzinfo else "local",
            "is_dst": now.dst() != timedelta(0) if now.tzinfo else None
        }
    }
    
    logger.info(f"📅 Generated comprehensive date context object with {len(date_context)} main categories")
    return date_context

def get_general_context(timezone_str: str = None) -> str:
    """
    Get current date/time context for LLM prompts.
    
    Args:
        timezone_str: Optional timezone string (e.g., "America/New_York")
    
    Returns:
        str: Formatted string with current date, time, and comprehensive date context
    """
    # Use the centralized date context object for consistency
    date_context = generate_date_context_object(timezone_str)
    
    # Extract values from the object for the string format
    current_date = date_context["current"]["date"]
    current_date_iso = date_context["current"]["date_iso"]
    current_time = date_context["current"]["time"]
    
    # Get UTC offset for the timezone
    if timezone_str:
        try:
            import pytz
            user_tz = pytz.timezone(timezone_str)
            utc_now = datetime.now(pytz.UTC)
            local_now = utc_now.astimezone(user_tz)
            utc_offset = local_now.strftime("%z")
            timezone_display = f"Your Local Timezone: {timezone_str} (UTC{utc_offset})"
        except Exception as e:
            logger.warning(f"Could not get UTC offset for {timezone_str}: {e}")
            timezone_display = f"Your Local Timezone: {timezone_str}"
    else:
        timezone_display = date_context["timezone"]["current_timezone"]
    
    # Build the formatted string with current date/time and comprehensive date context
    general_context_str = (
        f"Current Date & Time: {current_date} at {current_time}\n"
        f"{timezone_display}\n"
        f"Current date (ISO): {current_date_iso}\n\n"
        f"RELATIVE TO ABSOLUTE DATE MAPPING:\n{_format_date_context_for_prompt(date_context)}"
    )
    
    return general_context_str

def _format_date_context_for_prompt(date_context: dict) -> str:
    """
    Format the date context object for inclusion in LLM prompts.
    
    Args:
        date_context: The date context object from generate_date_context_object
    
    Returns:
        str: Formatted string with key date mappings
    """
    import json
    
    # Create a simplified version for the prompt
    prompt_context = {}
    
    # Current date/time
    prompt_context["now"] = date_context["current"]["datetime"]
    prompt_context["today"] = date_context["current"]["utc_start_of_day"]
    
    # Relative dates
    prompt_context["tomorrow"] = date_context["relative_dates"]["tomorrow"]["utc_start_of_day"]
    prompt_context["yesterday"] = date_context["relative_dates"]["yesterday"]["utc_start_of_day"]
    prompt_context["last night"] = date_context["relative_dates"]["last_night"]["datetime"]
    
    # Weekends
    this_weekend = [day["utc_start_of_day"] for day in date_context["weekend"]["this_weekend"]]
    next_weekend = [day["utc_start_of_day"] for day in date_context["weekend"]["next_weekend"]]
    last_weekend = [day["utc_start_of_day"] for day in date_context["weekend"]["last_weekend"]]
    
    prompt_context["this weekend"] = this_weekend
    prompt_context["next weekend"] = next_weekend
    prompt_context["last weekend"] = last_weekend
    
    # Weeks
    this_week = [day["utc_start_of_day"] for day in date_context["weeks"]["this_week"]]
    next_week = [day["utc_start_of_day"] for day in date_context["weeks"]["next_week"]]
    last_week = [day["utc_start_of_day"] for day in date_context["weeks"]["last_week"]]
    
    prompt_context["this week"] = this_week
    prompt_context["next week"] = next_week
    prompt_context["last week"] = last_week
    
    # Weekdays
    for key, value in date_context["weekdays"].items():
        prompt_context[key] = value["utc_start_of_day"]
    
    # Months
    this_month = [day["utc_start_of_day"] for day in date_context["months"]["this_month"]]
    next_month = [day["utc_start_of_day"] for day in date_context["months"]["next_month"]]
    last_month = [day["utc_start_of_day"] for day in date_context["months"]["last_month"]]
    
    prompt_context["this month"] = this_month
    prompt_context["next month"] = next_month
    prompt_context["last month"] = last_month
    
    # Years
    this_year = [day["utc_start_of_day"] for day in date_context["years"]["this_year"]]
    next_year = [day["utc_start_of_day"] for day in date_context["years"]["next_year"]]
    last_year = [day["utc_start_of_day"] for day in date_context["years"]["last_year"]]
    
    prompt_context["this year"] = this_year
    prompt_context["next year"] = next_year
    prompt_context["last year"] = last_year
    
    return json.dumps(prompt_context, indent=2)

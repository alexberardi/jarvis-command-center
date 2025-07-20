# Command Preprocessing System

This document describes the new command preprocessing functionality that reduces LLM prompt size by filtering commands based on keywords.

## Overview

The command preprocessing system uses keyword-based filtering to reduce the number of commands included in LLM prompts. This can significantly reduce prompt size and improve response times by only including commands that are relevant to the user's voice command.

## Key Features

- **Keyword-based filtering**: Commands define their own keywords for matching
- **Multiple command detection**: Detects commands with conjunctions ("and", "then", "also")
- **Fallback behavior**: Includes commands with no keywords when no matches are found
- **Environment variable toggle**: Easy to enable/disable via `COMMAND_PREPROCESSING_ENABLED`
- **Performance logging**: Logs filtering statistics for monitoring
- **Backward compatibility**: Works with existing system without breaking changes

## Configuration

### Environment Variables

- `COMMAND_PREPROCESSING_ENABLED`: Set to "true" to enable preprocessing, "false" to disable
  - Default: "false" (disabled)
  - Case insensitive

### Example

```bash
# Enable preprocessing
export COMMAND_PREPROCESSING_ENABLED=true

# Disable preprocessing
export COMMAND_PREPROCESSING_ENABLED=false
```

## Usage

### Defining Keywords in Commands

Commands now support an optional `keywords` field in the `CommandDefinition`:

```python
CommandDefinition(
    command_name="turn_on_lights",
    description="Turn on lights in a specific room",
    parameters=[CommandParameter(name="room", type="str")],
    keywords=["light", "lights", "on", "turn on", "switch on", "illuminate", "brighten"]
)
```

### Keywords Best Practices

1. **Include variations**: Add different ways users might express the same intent
2. **Include synonyms**: Add related terms (e.g., "illuminate" for "light")
3. **Include common phrases**: Add multi-word expressions users might use
4. **Keep it relevant**: Only include keywords that truly relate to the command
5. **Use fallback commands**: Some commands should have no keywords to serve as fallbacks

### Example Command Set

```python
commands = [
    CommandDefinition(
        command_name="turn_on_lights",
        description="Turn on lights in a specific room",
        parameters=[CommandParameter(name="room", type="str")],
        keywords=["light", "lights", "on", "turn on", "switch on", "illuminate", "brighten"]
    ),
    CommandDefinition(
        command_name="play_music",
        description="Play music",
        parameters=[CommandParameter(name="song", type="str")],
        keywords=["play", "music", "song", "track", "artist", "album", "jazz", "rock", "classical"]
    ),
    CommandDefinition(
        command_name="emergency_shutdown",
        description="Emergency system shutdown",
        parameters=[],
        keywords=None  # No keywords - fallback command
    )
]
```

## How It Works

### Filtering Process

1. **Exact keyword matching**: Looks for exact keyword matches in the voice command
2. **Regex pattern matching**: Uses regex for more flexible matching if no exact matches
3. **Fallback to no-keywords commands**: If no matches, includes commands with no keywords
4. **Final fallback**: If still no matches, includes all commands
5. **Multiple command detection**: Detects conjunctions and includes additional relevant commands
6. **Command limiting**: Limits results to maximum 8 commands to keep prompts manageable

### Multiple Command Detection

The system detects multiple commands using conjunction words:
- "and", "then", "also", "plus", "as well as", "after that"

Example: "turn on lights and play music" will include both lighting and music commands.

### Statistics and Logging

When preprocessing is enabled, the system logs filtering statistics:

```
Command filtering: 15 -> 3 commands (80.0% reduction, ~180 tokens saved)
```

## Performance Benefits

Typical performance improvements:
- **Token reduction**: 50-90% fewer tokens in prompts
- **Response time**: Faster LLM responses due to smaller prompts
- **Cost savings**: Reduced token usage means lower API costs
- **Accuracy**: More focused prompts can improve command recognition

## Testing

The system includes comprehensive tests:

- `tests/test_command_preprocessing.py`: Complete test suite for all functionality
- `tests/test_command_filter.py`: Specific tests for command filtering logic
- `tests/test_preprocessing_toggle.py`: Tests for environment variable toggle

Run tests with:
```bash
./run-tests.sh
# or
python -m pytest tests/test_command_preprocessing.py -v
```

## Backward Compatibility

The system is fully backward compatible:
- Commands without keywords work normally
- Existing interfaces unchanged
- Default behavior is disabled (no preprocessing)
- Optional `voice_command` parameter in system prompt providers

## Migration Guide

### For Existing Commands

1. Add keywords to your `CommandDefinition` objects:
   ```python
   # Before
   CommandDefinition(
       command_name="turn_on_lights",
       description="Turn on lights",
       parameters=[CommandParameter(name="room", type="str")]
   )
   
   # After
   CommandDefinition(
       command_name="turn_on_lights",
       description="Turn on lights",
       parameters=[CommandParameter(name="room", type="str")],
       keywords=["light", "lights", "on", "turn on"]
   )
   ```

2. Enable preprocessing in your environment:
   ```bash
   export COMMAND_PREPROCESSING_ENABLED=true
   ```

3. Monitor logs for filtering statistics and adjust keywords as needed

### For System Prompt Providers

If you have custom system prompt providers, update them to support the optional `voice_command` parameter:

```python
def build_system_prompt(self, node_context: dict, available_commands: List[CommandDefinition], voice_command: Optional[str] = None) -> str:
    # Your implementation
    pass
```

## Example Usage

See `example_usage.py` for a complete demonstration of the system in action.

## Troubleshooting

### Common Issues

1. **No filtering happening**: Check that `COMMAND_PREPROCESSING_ENABLED=true` is set
2. **Wrong commands filtered**: Review keywords for relevance and completeness
3. **Too many commands filtered**: Add more keywords or create fallback commands
4. **Performance not improved**: Ensure you have enough commands to make filtering worthwhile

### Debug Tools

- `debug_llm_responses.py`: Compare performance with/without preprocessing
- Log output: Monitor filtering statistics in application logs
- Test scripts: Use test files to verify filtering behavior

## Future Enhancements

Potential future improvements:
- Semantic similarity matching using embeddings
- Machine learning-based command relevance scoring
- Dynamic keyword learning from user interactions
- Command popularity-based filtering
- Context-aware filtering based on node location/time 
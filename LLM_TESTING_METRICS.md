# LLM Testing and Metrics Collection System

This document describes the comprehensive testing and metrics collection system for real LLM API calls in the Jarvis Command Center.

## Overview

The system provides multiple ways to test and analyze LLM API performance:

1. **Real LLM API Tests** - Tests that make actual HTTP calls to your LLM proxy
2. **Enhanced Debug Script** - Comprehensive metrics collection and analysis
3. **Automated Test Runner** - Easy-to-use scripts for running specific test suites

## Quick Start

### 1. Set up your LLM API URL

```bash
export JARVIS_LLM_PROXY_API_URL=http://your-llm-proxy-url:port
# Example:
export JARVIS_LLM_PROXY_API_URL=http://10.0.0.69:8000
```

### 2. Run real LLM API tests

```bash
# Using the shell script (recommended)
./run-real-llm-tests.sh

# Or using the Python script directly
python run_real_llm_tests.py

# Save metrics to JSON file
./run-real-llm-tests.sh --save-metrics

# List all available tests without running them
./run-real-llm-tests.sh --list-tests
```

### 3. Use the enhanced debug script

```bash
# Run comprehensive tests with detailed metrics
python tests/debug_llm_responses.py --test-type=real-api

# Run performance tests
python tests/debug_llm_responses.py --test-type=performance

# Run quality tests with edge cases
python tests/debug_llm_responses.py --test-type=quality

# Run all test types
python tests/debug_llm_responses.py --test-type=all --save-metrics
```

## Test Categories

### Real LLM API Tests

These tests make actual HTTP calls to your LLM proxy API and are marked with `@pytest.mark.skipif(not os.getenv("JARVIS_LLM_PROXY_API_URL"))`. They include:

- **Transcription Cleanup Tests**: Test the preprocessing pipeline
- **Basic Command Tests**: Light, temperature, music, timer commands
- **Complex Scenario Tests**: Multi-command, ambiguous, edge cases
- **Performance Tests**: Response time and throughput analysis

### Mocked Tests

These tests use mocked responses and don't make real API calls:

- **Unit Tests**: Test individual components
- **Integration Tests**: Test the full pipeline with mocked LLM responses
- **Edge Case Tests**: Test error handling and malformed responses

## Metrics Collected

### Performance Metrics

- **Response Time**: Total time from request to response (ms)
- **Latency**: Network and processing overhead
- **Throughput**: Tokens per second processed
- **Success Rate**: Percentage of successful responses

### Token Usage Metrics

- **Input Tokens**: System prompt + user command tokens
- **Output Tokens**: LLM response tokens
- **Total Tokens**: Combined input/output usage
- **Token Efficiency**: Tokens per request

### Cost Metrics

- **Per-Request Cost**: Estimated cost for each API call
- **Total Cost**: Cumulative cost for all requests
- **Cost per 1K Requests**: Projected costs at scale
- **Cost Savings**: Comparison between different configurations

### Quality Metrics

- **JSON Parse Success**: Percentage of valid JSON responses
- **Command Extraction Success**: Percentage of successfully extracted commands
- **Expected Command Match**: Accuracy of command recognition
- **Error Analysis**: Types and frequency of errors

### Response Quality Metrics

- **Response Length**: Character count of responses
- **System Prompt Length**: Size of system prompts
- **Model Issues**: Detection of malformed responses
- **Warning Analysis**: Non-critical issues

## File Structure

```
jarvis-command-center/
├── run_real_llm_tests.py          # Main test runner script
├── run-real-llm-tests.sh          # Shell wrapper script
├── tests/
│   ├── debug_llm_responses.py     # Enhanced debug script with metrics
│   └── test_llm_integration.py    # Real LLM API tests
├── LLM_TESTING_METRICS.md         # This documentation
└── real_llm_test_metrics.json     # Generated metrics file
```

## Usage Examples

### Basic Testing

```bash
# Run all real LLM API tests
./run-real-llm-tests.sh

# Run with metrics collection
./run-real-llm-tests.sh --save-metrics
```

### Advanced Debugging

```bash
# Run comprehensive analysis
python tests/debug_llm_responses.py --test-type=all --save-metrics

# Focus on performance
python tests/debug_llm_responses.py --test-type=performance

# Test edge cases
python tests/debug_llm_responses.py --test-type=quality
```

### Custom Metrics Analysis

```bash
# Save metrics to custom file
./run-real-llm-tests.sh --save-metrics --output-file=my_analysis.json

# Analyze the metrics file
python -c "
import json
with open('my_analysis.json') as f:
    data = json.load(f)
print(f'Success Rate: {data[\"success_rate\"]:.1f}%')
print(f'Average Response Time: {data[\"response_times\"][0]:.2f}ms')
"
```

## Configuration

### Environment Variables

- `JARVIS_LLM_PROXY_API_URL`: Your LLM proxy API URL (required for real tests)
- `JARVIS_TRANSCRIPTION_CLEANUP_ENABLED`: Enable/disable transcription cleanup
- `JARVIS_LLM_PROXY_API_VERSION`: API version (default: 0)

### Test Configuration

The tests use OpenAI-style request/response structures:

```python
# Request format
{
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "system_prompt"},
        {"role": "user", "content": "voice_command"}
    ],
    "temperature": 0.1,
    "max_tokens": 1000,
    "stream": False
}

# Response format
{
    "choices": [
        {
            "message": {
                "content": "json_response_string"
            }
        }
    ]
}
```

## Output Formats

### Console Output

The scripts provide detailed console output including:

- Real-time test progress
- Individual test results with metrics
- Comprehensive summary statistics
- Error and warning analysis

### JSON Metrics File

When using `--save-metrics`, a detailed JSON file is created with:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "llm_api_url": "http://10.0.0.69:8000",
  "total_tests": 14,
  "passed": 12,
  "failed": 1,
  "skipped": 1,
  "success_rate": 85.7,
  "response_times": [1250.5, 1180.2, ...],
  "token_usage": [450, 380, ...],
  "costs": [0.0012, 0.0010, ...],
  "test_results": {
    "start_time": "2024-01-15T10:29:45",
    "end_time": "2024-01-15T10:30:15",
    "duration_seconds": 30.5,
    "return_code": 0
  }
}
```

## Troubleshooting

### Common Issues

1. **LLM API URL not set**
   ```bash
   export JARVIS_LLM_PROXY_API_URL=http://your-url:port
   ```

2. **Tests being skipped**
   - Check that `JARVIS_LLM_PROXY_API_URL` is set
   - Verify the URL is accessible

3. **Connection errors**
   - Check network connectivity
   - Verify LLM proxy is running
   - Check firewall settings

4. **High response times**
   - Monitor LLM proxy performance
   - Check network latency
   - Consider model size/configuration

### Debug Mode

For detailed debugging, run with verbose output:

```bash
python run_real_llm_tests.py --save-metrics
# Check the generated JSON file for detailed error information
```

## Integration with CI/CD

### GitHub Actions

Add to your workflow:

```yaml
- name: Run Real LLM Tests
  env:
    JARVIS_LLM_PROXY_API_URL: ${{ secrets.LLM_PROXY_URL }}
  run: |
    ./run-real-llm-tests.sh --save-metrics
    # Upload metrics as artifacts
    cp real_llm_test_metrics.json $GITHUB_WORKSPACE/
```

### Metrics Tracking

Track metrics over time:

```bash
# Run tests and save with timestamp
./run-real-llm-tests.sh --save-metrics --output-file=metrics_$(date +%Y%m%d_%H%M%S).json
```

## Best Practices

1. **Regular Testing**: Run real LLM tests regularly to monitor performance
2. **Metrics Tracking**: Save metrics over time to track improvements
3. **Cost Monitoring**: Monitor token usage and costs
4. **Error Analysis**: Review error patterns and fix issues
5. **Performance Optimization**: Use metrics to identify bottlenecks

## Contributing

When adding new tests:

1. Use `@pytest.mark.skipif(not os.getenv("JARVIS_LLM_PROXY_API_URL"))` for real API tests
2. Add comprehensive metrics collection
3. Include expected command validation
4. Document test purpose and expected behavior
5. Update this documentation

## Support

For issues or questions:

1. Check the troubleshooting section
2. Review the generated metrics files
3. Check the test output for specific error messages
4. Verify your LLM proxy configuration 
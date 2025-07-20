# Test Suite Documentation

This directory contains comprehensive tests for the Jarvis Command Center application. The test suite includes unit tests, integration tests, and end-to-end tests that validate all aspects of the voice command processing system.

## Test Structure Overview

```
tests/
├── README.md                           # This documentation file
├── conftest.py                         # Pytest configuration and fixtures
├── pytest.ini                         # Pytest settings
├── test_*.py                          # Individual test files
└── __pycache__/                       # Python cache directory
```

## Test Categories

### 🧪 **Unit Tests**
Test individual components in isolation with mocked dependencies.

### 🔗 **Integration Tests**
Test how components work together, including database interactions and API calls.

### 🌐 **End-to-End Tests**
Test complete user workflows from HTTP request to response.

### 🚨 **LLM Integration Tests**
Test real LLM interactions and model behavior validation.

---

## Test Files

### **Core Functionality Tests**

#### `test_voice_command_integration.py`
**Purpose**: Tests the main voice command processing endpoint and core functionality.

**Key Tests**:
- `test_successful_command_with_room_context` - Validates successful command processing with room context
- `test_successful_command_multiple_parameters` - Tests commands with multiple parameters (temperature, room, etc.)
- `test_missing_room_parameter` - Handles missing required room parameter gracefully
- `test_malformed_llm_response` - Tests recovery from malformed LLM responses
- `test_different_room_contexts` - Validates commands work across different rooms
- `test_system_prompt_generation` - Ensures system prompts are generated correctly

**What it tests**: The complete voice command processing pipeline from HTTP request to JSON response.

---

#### `test_voice_command_integration_with_db.py`
**Purpose**: Tests voice command processing with real database interactions.

**Key Tests**:
- `test_voice_command_with_db_node` - Tests commands with database-stored node information
- `test_voice_command_with_invalid_api_key` - Validates API key authentication
- `test_voice_command_with_missing_node` - Handles missing node records
- `test_voice_command_with_db_transaction` - Tests database transaction handling

**What it tests**: Database integration, authentication, and data persistence.

---

#### `test_voice_command_database.py`
**Purpose**: Tests database operations and data models.

**Key Tests**:
- `test_node_creation` - Validates node record creation
- `test_node_retrieval` - Tests node data retrieval
- `test_api_key_validation` - Validates API key checking
- `test_database_constraints` - Tests database integrity constraints

**What it tests**: Database models, CRUD operations, and data validation.

---

### **Multi-Command Functionality Tests**

#### `test_multi_command_scenarios.py`
**Purpose**: Tests the new multi-command functionality that allows processing multiple commands in a single request.

**Key Tests**:
- `test_sequential_commands_with_and` - Commands connected with "and" (e.g., "turn on lights and set temperature")
- `test_sequential_commands_with_then` - Commands connected with "then" (e.g., "turn off lights then play music")
- `test_mixed_success_failure_commands` - Scenarios where some commands succeed and others fail
- `test_three_or_more_commands` - Handling of 3+ commands in one request
- `test_commands_with_different_rooms` - Commands affecting different rooms

**What it tests**: Multi-command parsing, execution ordering, and mixed success/failure handling.

---

### **LLM Model Validation Tests**

#### `test_llm_model_validation.py`
**Purpose**: Tests LLM integration robustness and validates handling of real-world LLM model issues.

**Key Tests**:
- `test_llm_returns_extra_content_after_json` - Handles valid JSON followed by extra content (like `<|im_start|>user`)
- `test_llm_returns_explanation_before_json` - Handles explanatory text before JSON
- `test_llm_returns_completely_empty_response` - Handles empty LLM responses
- `test_llm_returns_malformed_json_structure` - Validates error handling for wrong JSON structure
- `test_llm_returns_invalid_json_syntax` - Recovers from syntactically invalid JSON
- `test_llm_returns_nested_json_in_text` - Handles JSON embedded within explanatory text
- `test_llm_returns_multiple_json_objects` - Extracts first JSON when multiple are present
- `test_llm_response_consistency` - Ensures consistent responses for same commands

**What it tests**: LLM response parsing, error recovery, and model behavior validation.

---

### **Edge Cases and Error Handling Tests**

#### `test_edge_cases.py`
**Purpose**: Tests boundary conditions, error scenarios, and unusual input handling.

**Key Tests**:
- `test_empty_voice_command` - Handles empty voice commands
- `test_extremely_long_voice_command` - Handles very long commands (1000+ words)
- `test_voice_command_with_special_characters` - Unicode, emojis, newlines, special characters
- `test_no_available_commands` - Behavior when no commands are available
- `test_command_with_null_parameters` - Commands that don't require parameters
- `test_command_with_extra_parameters` - When LLM returns extra parameters not in schema
- `test_preprocessing_with_no_matching_keywords` - Keyword filtering when no keywords match
- `test_llm_timeout_simulation` - Exception handling for LLM timeouts
- `test_node_context_edge_cases` - Various edge cases in node context (null, empty, whitespace)

**What it tests**: System robustness, error handling, and boundary condition behavior.

---

### **Command Processing Tests**

#### `test_command_preprocessing.py`
**Purpose**: Tests the command preprocessing and filtering functionality.

**Key Tests**:
- `test_keyword_filtering` - Tests keyword-based command filtering
- `test_exact_keyword_matching` - Validates exact keyword matches
- `test_regex_pattern_matching` - Tests regex pattern matching for commands
- `test_fallback_matching` - Tests fallback when no keywords match
- `test_multiple_command_detection` - Detects multiple commands via conjunctions
- `test_preprocessing_toggle` - Tests enabling/disabling preprocessing

**What it tests**: Command preprocessing, keyword filtering, and optimization features.

---

### **LLM Integration Tests**

#### `test_llm_integration.py`
**Purpose**: Tests real LLM interactions and end-to-end command processing.

**Key Tests**:
- `test_real_llm_simple_command` - Tests simple commands with real LLM
- `test_real_llm_complex_command` - Tests complex multi-parameter commands
- `test_real_llm_ambiguous_command` - Tests handling of ambiguous commands
- `test_real_llm_invalid_command` - Tests commands that don't match available options
- `test_real_llm_multiple_commands` - Tests multiple commands in one request
- `test_real_llm_different_rooms` - Tests commands across different rooms
- `test_real_llm_missing_parameters` - Tests handling of missing required parameters
- `test_real_llm_response_format` - Validates LLM response format compliance
- `test_real_llm_consistency` - Tests response consistency across multiple calls
- `test_real_llm_performance` - Tests response time and performance
- `test_real_llm_stress_test` - Stress tests with many concurrent requests

**What it tests**: Real LLM behavior, performance, and end-to-end functionality.

---

## Test Execution

### Running All Tests
```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run with coverage
python -m pytest --cov=app
```

### Running Specific Test Categories
```bash
# Run only unit tests (fast)
python -m pytest tests/test_voice_command_integration.py

# Run only LLM integration tests (slow)
python -m pytest tests/test_llm_integration.py

# Run only edge case tests
python -m pytest tests/test_edge_cases.py
```

### Running Individual Tests
```bash
# Run a specific test
python -m pytest tests/test_voice_command_integration.py::TestVoiceCommandIntegration::test_successful_command_with_room_context

# Run tests matching a pattern
python -m pytest -k "test_multi_command"
```

## Test Data and Fixtures

### Common Test Data
- **Mock Nodes**: Test nodes with various room configurations
- **Mock Commands**: Standard command definitions for testing
- **Mock LLM Responses**: Realistic LLM responses for different scenarios

### Test Fixtures (conftest.py)
- **Database fixtures**: Clean database setup/teardown
- **Authentication fixtures**: Mock API keys and node contexts
- **LLM fixtures**: Mock LLM responses and error scenarios

## Test Coverage

The test suite provides comprehensive coverage across:

- ✅ **HTTP Endpoints**: All API endpoints tested
- ✅ **Database Operations**: CRUD operations and constraints
- ✅ **Authentication**: API key validation and node context
- ✅ **LLM Integration**: Real and mocked LLM interactions
- ✅ **Command Processing**: Single and multi-command scenarios
- ✅ **Error Handling**: Edge cases and failure scenarios
- ✅ **Performance**: Response times and stress testing

## Test Statistics

- **Total Tests**: 103
- **Unit Tests**: ~40
- **Integration Tests**: ~35
- **LLM Integration Tests**: ~11
- **Edge Case Tests**: ~17
- **Average Test Runtime**: ~1.2 minutes for full suite
- **Coverage**: >95% of application code

## Contributing to Tests

When adding new functionality:

1. **Add unit tests** for individual components
2. **Add integration tests** for component interactions
3. **Add edge case tests** for boundary conditions
4. **Update this documentation** with new test descriptions
5. **Ensure all tests pass** before submitting changes

## Debugging Tests

### Common Issues
- **Database connection**: Ensure test database is available
- **LLM service**: Check if LLM proxy is running for integration tests
- **Environment variables**: Verify test environment configuration

### Debugging Commands
```bash
# Run with detailed output
python -m pytest -vvv -s

# Run with pdb on failure
python -m pytest --pdb

# Run only failed tests
python -m pytest --lf
```

---

*This documentation is maintained alongside the test suite. Please update it when adding new tests or modifying existing ones.* 
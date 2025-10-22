"""
Enterprise-grade prompts for test generation.

This module contains comprehensive, well-structured prompts for generating
different types of test cases using LLMs.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TestType(str, Enum):
    """Types of tests that can be generated."""
    
    UNIT = "unit"
    FUNCTIONAL = "functional"
    API = "api"
    INTEGRATION = "integration"
    EDGE_CASE = "edge_case"


class TestGenerationPrompt(BaseModel):
    """
    Structured prompt for test generation.
    
    Attributes:
        test_type: Type of test to generate
        target_code: Code to generate tests for
        context: Additional context
        requirements: Specific requirements
    """
    
    test_type: TestType = Field(..., description="Type of test")
    target_code: str = Field(..., description="Code to test")
    context: Optional[str] = Field(default=None, description="Additional context")
    requirements: Optional[str] = Field(default=None, description="Special requirements")


class PromptTemplates:
    """
    Enterprise-grade prompt templates for test generation.
    
    These prompts are designed to generate high-quality, comprehensive
    test cases covering positive, negative, edge cases, and exceptions.
    """
    
    SYSTEM_PROMPT = """You are an expert Python test engineer specializing in writing comprehensive, 
production-grade test cases. Your tests follow best practices including:

1. **Coverage**: Cover positive cases, negative cases, edge cases, and exception handling
2. **Clarity**: Write clear, self-documenting test names and docstrings
3. **Isolation**: Use mocking for external dependencies and I/O operations
4. **Standards**: Follow pytest conventions and PEP 8 style guidelines
5. **Assertions**: Use specific, meaningful assertions with clear error messages
6. **Fixtures**: Utilize pytest fixtures for reusable test setup
7. **Parameterization**: Use @pytest.mark.parametrize for multiple test cases

Always structure tests with:
- Arrange: Setup test data and mocks
- Act: Execute the code under test
- Assert: Verify expected behavior

Generate only valid, executable Python code. Include all necessary imports.

🔒 CRITICAL SAFETY GUARDRAILS (MANDATORY):
────────────────────────────────────────────────────
1. ✅ DETERMINISM - Tests MUST be deterministic:
   - ❌ NEVER use: time.sleep(), datetime.now(), random() without seed
   - ✅ ALWAYS use: monkeypatch.setattr(), freezegun, mock.patch()
   - ✅ ALWAYS seed random: random.seed(42)
   - ✅ ALWAYS mock time: mock.patch('datetime.datetime.now')

2. ✅ FILE BOUNDARIES - Only write to tests/ directory:
   - ❌ NEVER modify: src/, config/, .env, or any production code
   - ✅ ONLY write: tests/**/*.py
   - ✅ ALWAYS use: temp files with tempfile.TemporaryDirectory()

3. ✅ SECRETS PROTECTION - Never expose sensitive data:
   - ❌ NEVER use real: API keys, passwords, tokens, credentials
   - ✅ ALWAYS use: mock values, "fake_token_123", "test_password"
   - ✅ ALWAYS mock: environment variables with os.environ patching

4. ✅ ISOLATION - Tests must be isolated:
   - ❌ NEVER access: real databases, network, file system
   - ✅ ALWAYS mock: requests, database calls, file I/O
   - ✅ ALWAYS cleanup: use fixtures with yield for teardown

5. ✅ PERFORMANCE - Tests must be fast:
   - ❌ NEVER use: time.sleep(), long-running operations
   - ✅ KEEP tests under 1 second each
   - ✅ USE mocks for slow operations

VIOLATION OF THESE GUARDRAILS = TEST REJECTION
────────────────────────────────────────────────────"""
    
    UNIT_TEST_PROMPT = """Generate comprehensive unit tests for the following Python code:

TARGET CODE:
```python
{target_code}
```

{context_section}

REQUIREMENTS:
1. Generate pytest-based unit tests in a class format
2. Cover ALL of the following scenarios:
   - **Positive Cases**: Normal, expected inputs and behavior
   - **Negative Cases**: Invalid inputs, error conditions
   - **Edge Cases**: Boundary values, empty inputs, None values
   - **Exception Handling**: Verify proper exception raising and handling
3. Use mocking for:
   - File I/O operations
   - Network calls
   - Database operations
   - External API calls
   - System calls
4. Include docstrings for the test class and each test method
5. Use descriptive test names following: test_<function>_<scenario>_<expected_result>
6. Add type hints to all test methods
7. Use pytest fixtures where appropriate
8. Include setup and teardown if needed

{requirements_section}

Generate ONLY the Python test code with proper imports. Start with imports, then the test class."""
    
    FUNCTIONAL_TEST_PROMPT = """Generate comprehensive functional tests for the following code:

TARGET CODE:
```python
{target_code}
```

{context_section}

FUNCTIONAL TEST REQUIREMENTS:
1. Test the complete functionality and integration of components
2. Verify the function/class behavior in realistic scenarios
3. Test the interaction between different parts of the code
4. Validate end-to-end workflows
5. Include tests for:
   - Happy path scenarios
   - Alternative paths
   - Error recovery
   - State transitions
6. Mock external dependencies but test internal interactions
7. Use pytest fixtures for complex setup
8. Include integration points testing

{requirements_section}

Generate complete, executable pytest functional tests."""
    
    API_TEST_PROMPT = """Generate comprehensive API tests for the following REST API code:

TARGET CODE:
```python
{target_code}
```

{context_section}

API TEST REQUIREMENTS:
1. Test all HTTP methods (GET, POST, PUT, DELETE, PATCH)
2. Validate:
   - Response status codes
   - Response body structure and content
   - Response headers
   - Error responses
3. Cover scenarios:
   - Successful requests (2xx)
   - Client errors (4xx): bad requests, unauthorized, not found
   - Server errors (5xx)
   - Edge cases: empty bodies, large payloads, special characters
4. Use pytest fixtures for:
   - Test client setup
   - Mock database/dependencies
   - Sample data
5. Test authentication and authorization if applicable
6. Validate request/response schemas
7. Test rate limiting and pagination if applicable

{requirements_section}

Generate pytest-based API tests using appropriate testing framework (e.g., FastAPI TestClient, Flask test_client)."""
    
    EDGE_CASE_PROMPT = """Generate edge case and boundary tests for the following code:

TARGET CODE:
```python
{target_code}
```

{context_section}

EDGE CASE REQUIREMENTS:
Focus on boundary conditions and unusual inputs:

1. **Boundary Values**:
   - Minimum and maximum values
   - Zero, negative numbers
   - Empty strings, lists, dictionaries
   - Single element collections
   - Very large inputs

2. **Null/None Handling**:
   - None as input
   - Optional parameters
   - Null in nested structures

3. **Type Boundaries**:
   - Type conversion limits
   - Overflow/underflow
   - Precision limits

4. **Special Characters**:
   - Unicode, emojis
   - Control characters
   - SQL/code injection attempts

5. **Concurrency Issues** (if applicable):
   - Race conditions
   - Thread safety

6. **Resource Limits**:
   - Memory constraints
   - Timeout scenarios

{requirements_section}

Generate pytest tests focusing exclusively on edge cases and boundaries."""
    
    MOCK_PATTERN_TEMPLATE = """
# Common mocking patterns to include:

## File I/O Mocking
```python
from unittest.mock import mock_open, patch

@patch('builtins.open', mock_open(read_data='test data'))
def test_with_file_mock():
    # Your test here
    pass
```

## External API Mocking
```python
from unittest.mock import Mock, patch

@patch('requests.get')
def test_api_call(mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'key': 'value'}
    mock_get.return_value = mock_response
    # Your test here
```

## Database Mocking
```python
@patch('your_module.database_connection')
def test_database_operation(mock_db):
    mock_db.query.return_value = [{'id': 1, 'name': 'test'}]
    # Your test here
```
"""
    
    REFINEMENT_PROMPT = """Review and improve the following test code:

ORIGINAL TESTS:
```python
{original_tests}
```

EXECUTION RESULTS:
```
{execution_results}
```

ISSUES IDENTIFIED:
{issues}

INSTRUCTIONS:
1. Fix any failing tests
2. Improve test coverage based on execution results
3. Add missing edge cases
4. Improve assertion messages
5. Optimize mock usage
6. Ensure all imports are correct
7. Follow pytest best practices

Generate the complete, corrected test code."""
    
    @classmethod
    def get_prompt(
        cls,
        test_type: TestType,
        target_code: str,
        context: Optional[str] = None,
        requirements: Optional[str] = None
    ) -> str:
        """
        Generate a complete prompt for test generation.
        
        Args:
            test_type: Type of test to generate
            target_code: Code to generate tests for
            context: Additional context about the code
            requirements: Specific requirements or constraints
            
        Returns:
            Formatted prompt string
            
        Example:
            >>> prompt = PromptTemplates.get_prompt(
            ...     TestType.UNIT,
            ...     "def add(a, b): return a + b",
            ...     context="Simple addition function"
            ... )
        """
        # Format context section
        context_section = ""
        if context:
            context_section = f"""
ADDITIONAL CONTEXT:
{context}
"""
        
        # Format requirements section
        requirements_section = ""
        if requirements:
            requirements_section = f"""
ADDITIONAL REQUIREMENTS:
{requirements}
"""
        
        # Select appropriate template
        template_map = {
            TestType.UNIT: cls.UNIT_TEST_PROMPT,
            TestType.FUNCTIONAL: cls.FUNCTIONAL_TEST_PROMPT,
            TestType.API: cls.API_TEST_PROMPT,
            TestType.EDGE_CASE: cls.EDGE_CASE_PROMPT,
        }
        
        template = template_map.get(test_type, cls.UNIT_TEST_PROMPT)
        
        return template.format(
            target_code=target_code,
            context_section=context_section,
            requirements_section=requirements_section
        )
    
    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for test generation."""
        return cls.SYSTEM_PROMPT
    
    @classmethod
    def get_refinement_prompt(
        cls,
        original_tests: str,
        execution_results: str,
        issues: str
    ) -> str:
        """
        Generate prompt for refining tests based on execution results.
        
        Args:
            original_tests: The original test code
            execution_results: Results from test execution
            issues: Identified issues to fix
            
        Returns:
            Refinement prompt
        """
        return cls.REFINEMENT_PROMPT.format(
            original_tests=original_tests,
            execution_results=execution_results,
            issues=issues
        )


def get_test_type_for_code(code: str) -> TestType:
    """
    Determine appropriate test type based on code analysis.
    
    Args:
        code: Source code to analyze
        
    Returns:
        Recommended TestType
        
    Example:
        >>> code = "def get_user(id): return db.query(...)"
        >>> test_type = get_test_type_for_code(code)
        >>> print(test_type)
        TestType.UNIT
    """
    # Simple heuristics to determine test type
    if any(keyword in code.lower() for keyword in ['@app.', '@router.', 'fastapi', 'flask']):
        return TestType.API
    elif any(keyword in code.lower() for keyword in ['request', 'response', 'http']):
        return TestType.API
    elif 'class ' in code and len(code.split('\n')) > 20:
        return TestType.FUNCTIONAL
    else:
        return TestType.UNIT


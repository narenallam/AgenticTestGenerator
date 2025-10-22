"""Factory for creating test runners."""

from typing import Optional

from src.test_runners.base import BaseTestRunner
from src.test_runners.jest_runner import JestRunner
from src.test_runners.junit_runner import JUnitRunner
from src.test_runners.pytest_runner import PyTestRunner


def create_test_runner(framework: str) -> BaseTestRunner:
    """
    Create test runner for specified framework.
    
    Args:
        framework: Framework name ('pytest', 'jest', 'junit')
        
    Returns:
        BaseTestRunner instance
        
    Raises:
        ValueError: If framework is not supported
    """
    framework = framework.lower()
    
    if framework == "pytest":
        return PyTestRunner()
    elif framework == "jest":
        return JestRunner()
    elif framework == "junit":
        return JUnitRunner()
    else:
        raise ValueError(f"Unsupported framework: {framework}")


def get_runner_for_language(language: str) -> BaseTestRunner:
    """
    Get test runner based on programming language.
    
    Args:
        language: Programming language
        
    Returns:
        BaseTestRunner instance
    """
    language = language.lower()
    
    if language in ["python", "py"]:
        return PyTestRunner()
    elif language in ["javascript", "typescript", "js", "ts"]:
        return JestRunner()
    elif language == "java":
        return JUnitRunner()
    else:
        return PyTestRunner()  # Default to pytest


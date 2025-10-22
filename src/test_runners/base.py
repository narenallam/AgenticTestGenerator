"""
Base test runner interface.

Defines the common interface for all test runners.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class TestRunnerResult(BaseModel):
    """Result from test execution."""
    
    success: bool = Field(..., description="Tests passed")
    framework: str = Field(..., description="Test framework used")
    tests_run: int = Field(default=0, description="Tests executed")
    tests_passed: int = Field(default=0, description="Tests passed")
    tests_failed: int = Field(default=0, description="Tests failed")
    tests_skipped: int = Field(default=0, description="Tests skipped")
    execution_time: float = Field(default=0.0, description="Execution time")
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")
    coverage: Optional[float] = Field(default=None, description="Coverage %")
    exit_code: int = Field(..., description="Exit code")


class BaseTestRunner(ABC):
    """
    Abstract base class for test runners.
    
    All test runners must implement this interface.
    """
    
    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Get framework name."""
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        pass
    
    @abstractmethod
    def run_tests(
        self,
        test_file: Path,
        source_file: Optional[Path] = None,
        with_coverage: bool = False
    ) -> TestRunnerResult:
        """
        Run tests from a file.
        
        Args:
            test_file: Path to test file
            source_file: Path to source file (for coverage)
            with_coverage: Whether to collect coverage
            
        Returns:
            TestRunnerResult
        """
        pass
    
    @abstractmethod
    def run_tests_from_code(
        self,
        test_code: str,
        source_code: Optional[str] = None,
        with_coverage: bool = False
    ) -> TestRunnerResult:
        """
        Run tests from code strings.
        
        Args:
            test_code: Test code
            source_code: Source code (for coverage)
            with_coverage: Whether to collect coverage
            
        Returns:
            TestRunnerResult
        """
        pass
    
    @abstractmethod
    def validate_test_code(self, test_code: str) -> bool:
        """
        Validate test code syntax.
        
        Args:
            test_code: Test code to validate
            
        Returns:
            True if valid
        """
        pass


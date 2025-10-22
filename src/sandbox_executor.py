"""
Sandbox test execution environment.

This module provides isolated test execution with timeout protection
and comprehensive result capturing. Uses Docker by default for security,
falls back to tempfile-based execution if Docker is unavailable.
"""

import ast
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from rich.console import Console

from config.settings import settings

console = Console()

# Try to import Docker sandbox
try:
    from src.sandbox.docker_sandbox import DockerSandbox, DockerTestResult
    DOCKER_AVAILABLE = True
    console.print("[green]✓[/green] Docker sandbox available (secure mode)")
except ImportError:
    DOCKER_AVAILABLE = False
    console.print("[yellow]⚠[/yellow] Docker not available, using tempfile sandbox (less secure)")


class TestResult(BaseModel):
    """
    Test execution result.
    
    Attributes:
        success: Whether tests passed
        stdout: Standard output
        stderr: Standard error
        exit_code: Process exit code
        tests_run: Number of tests executed
        tests_passed: Number of tests passed
        tests_failed: Number of tests failed
        execution_time: Execution time in seconds
        coverage: Code coverage percentage if available
    """
    
    success: bool = Field(..., description="Tests passed")
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")
    exit_code: int = Field(..., description="Exit code")
    tests_run: int = Field(default=0, description="Tests executed")
    tests_passed: int = Field(default=0, description="Tests passed")
    tests_failed: int = Field(default=0, description="Tests failed")
    execution_time: float = Field(default=0.0, description="Execution time")
    coverage: Optional[float] = Field(default=None, description="Coverage %")
    error_details: Optional[str] = Field(default=None, description="Error details")


class SandboxExecutor:
    """
    Sandbox environment for executing tests safely.
    
    This class provides:
    - Isolated test execution (Docker preferred, tempfile fallback)
    - Timeout protection
    - Resource limits
    - Result capture and analysis
    """
    
    def __init__(
        self,
        timeout: int = 30,
        temp_dir: Optional[Path] = None,
        use_docker: bool = True
    ) -> None:
        """
        Initialize sandbox executor.
        
        Args:
            timeout: Maximum execution time in seconds
            temp_dir: Temporary directory for test files (tempfile mode only)
            use_docker: Whether to use Docker if available
        """
        self.timeout = timeout
        self.use_docker = use_docker and DOCKER_AVAILABLE
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp(prefix="test_sandbox_"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Docker sandbox if available
        self.docker_sandbox = None
        if self.use_docker:
            try:
                from src.sandbox.docker_sandbox import create_docker_sandbox
                self.docker_sandbox = create_docker_sandbox(
                    timeout=timeout,
                    mem_limit="512m",
                    network_disabled=True
                )
                console.print(f"[green]✓[/green] Docker sandbox initialized (secure mode)")
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] Docker initialization failed: {e}")
                console.print("[yellow]⚠[/yellow] Falling back to tempfile sandbox")
                self.use_docker = False
        
        if not self.use_docker:
            console.print(f"[yellow]⚠[/yellow] Tempfile sandbox initialized: {self.temp_dir}")
    
    def execute_tests(
        self,
        test_code: str,
        source_code: Optional[str] = None,
        additional_deps: Optional[List[str]] = None
    ) -> TestResult:
        """
        Execute test code in sandbox environment.
        
        Args:
            test_code: The test code to execute
            source_code: Source code being tested (if separate)
            additional_deps: Additional dependencies to install
            
        Returns:
            TestResult object with execution results
            
        Example:
            >>> executor = SandboxExecutor()
            >>> result = executor.execute_tests(test_code)
            >>> print(f"Tests passed: {result.success}")
        """
        console.print("[cyan]Executing tests in sandbox...[/cyan]")
        
        # Validate test code syntax
        if not self._validate_syntax(test_code):
            return TestResult(
                success=False,
                exit_code=1,
                error_details="Test code has syntax errors"
            )
        
        # Use Docker if available
        if self.use_docker and self.docker_sandbox:
            return self._execute_docker(test_code, source_code, additional_deps)
        
        # Fall back to tempfile execution
        return self._execute_tempfile(test_code, source_code, additional_deps)
    
    def _execute_docker(
        self,
        test_code: str,
        source_code: Optional[str],
        additional_deps: Optional[List[str]]
    ) -> TestResult:
        """Execute tests using Docker sandbox."""
        docker_result = self.docker_sandbox.execute_tests(
            test_code=test_code,
            source_code=source_code,
            requirements=additional_deps or [],
            framework="pytest"
        )
        
        # Convert DockerTestResult to TestResult
        result = TestResult(
            success=docker_result.success,
            stdout=docker_result.stdout,
            stderr=docker_result.stderr,
            exit_code=docker_result.exit_code,
            tests_run=docker_result.tests_run,
            tests_passed=docker_result.tests_passed,
            tests_failed=docker_result.tests_failed,
            execution_time=docker_result.execution_time,
            coverage=docker_result.coverage,
            error_details=docker_result.error_details
        )
        
        console.print(
            f"[{'green' if result.success else 'red'}]"
            f"Tests {'passed' if result.success else 'failed'}: "
            f"{result.tests_passed}/{result.tests_run}[/] (Docker)"
        )
        
        return result
    
    def _execute_tempfile(
        self,
        test_code: str,
        source_code: Optional[str],
        additional_deps: Optional[List[str]]
    ) -> TestResult:
        """Execute tests using tempfile-based sandbox (fallback)."""
        # Create temporary test file
        test_file = self.temp_dir / "test_generated.py"
        test_file.write_text(test_code, encoding='utf-8')
        
        # Create source file if provided
        if source_code:
            source_file = self.temp_dir / "source_module.py"
            source_file.write_text(source_code, encoding='utf-8')
        
        # Install additional dependencies if needed
        if additional_deps:
            self._install_dependencies(additional_deps)
        
        # Execute tests using pytest
        result = self._run_pytest(test_file)
        
        console.print(
            f"[{'green' if result.success else 'red'}]"
            f"Tests {'passed' if result.success else 'failed'}: "
            f"{result.tests_passed}/{result.tests_run}[/] (tempfile)"
        )
        
        return result
    
    def execute_with_coverage(
        self,
        test_code: str,
        source_code: str
    ) -> TestResult:
        """
        Execute tests with coverage analysis.
        
        Args:
            test_code: Test code
            source_code: Source code to measure coverage for
            
        Returns:
            TestResult with coverage information
        """
        # Use Docker if available
        if self.use_docker and self.docker_sandbox:
            docker_result = self.docker_sandbox.execute_with_coverage(
                test_code=test_code,
                source_code=source_code,
                framework="pytest"
            )
            
            # Convert to TestResult
            return TestResult(
                success=docker_result.success,
                stdout=docker_result.stdout,
                stderr=docker_result.stderr,
                exit_code=docker_result.exit_code,
                tests_run=docker_result.tests_run,
                tests_passed=docker_result.tests_passed,
                tests_failed=docker_result.tests_failed,
                execution_time=docker_result.execution_time,
                coverage=docker_result.coverage,
                error_details=docker_result.error_details
            )
        
        # Fall back to tempfile execution
        # Write source code
        source_file = self.temp_dir / "source_module.py"
        source_file.write_text(source_code, encoding='utf-8')
        
        # Write test code
        test_file = self.temp_dir / "test_generated.py"
        test_file.write_text(test_code, encoding='utf-8')
        
        # Run pytest with coverage
        cmd = [
            "pytest",
            str(test_file),
            f"--cov={source_file.stem}",
            "--cov-report=term-missing",
            "-v",
            "--tb=short"
        ]
        
        result = self._execute_command(cmd)
        
        # Parse coverage from output
        coverage = self._parse_coverage(result.stdout)
        result.coverage = coverage
        
        return result
    
    def _validate_syntax(self, code: str) -> bool:
        """
        Validate Python syntax.
        
        Args:
            code: Python code to validate
            
        Returns:
            True if syntax is valid
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            console.print(f"[red]Syntax error: {e}[/red]")
            return False
    
    def _run_pytest(self, test_file: Path) -> TestResult:
        """
        Run pytest on the test file.
        
        Args:
            test_file: Path to test file
            
        Returns:
            TestResult object
        """
        cmd = [
            "pytest",
            str(test_file),
            "-v",
            "--tb=short",
            "--no-header",
            f"--timeout={self.timeout}"
        ]
        
        return self._execute_command(cmd)
    
    def _execute_command(self, cmd: List[str]) -> TestResult:
        """
        Execute a command and capture results.
        
        Args:
            cmd: Command to execute
            
        Returns:
            TestResult object
        """
        import time
        
        start_time = time.time()
        
        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.temp_dir)
            )
            
            execution_time = time.time() - start_time
            
            # Parse pytest output
            tests_run, tests_passed, tests_failed = self._parse_pytest_output(
                process.stdout,
                process.stderr
            )
            
            return TestResult(
                success=process.returncode == 0,
                stdout=process.stdout,
                stderr=process.stderr,
                exit_code=process.returncode,
                tests_run=tests_run,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                execution_time=execution_time
            )
        
        except subprocess.TimeoutExpired:
            return TestResult(
                success=False,
                exit_code=-1,
                stderr=f"Test execution timed out after {self.timeout} seconds",
                execution_time=self.timeout,
                error_details="Timeout"
            )
        
        except Exception as e:
            return TestResult(
                success=False,
                exit_code=-1,
                stderr=str(e),
                error_details=f"Execution error: {e}"
            )
    
    def _parse_pytest_output(
        self,
        stdout: str,
        stderr: str
    ) -> Tuple[int, int, int]:
        """
        Parse pytest output to extract test statistics.
        
        Args:
            stdout: Standard output
            stderr: Standard error
            
        Returns:
            Tuple of (tests_run, tests_passed, tests_failed)
        """
        import re
        
        output = stdout + stderr
        
        # Look for pytest summary line
        # Example: "5 passed in 0.12s" or "3 failed, 2 passed in 1.23s"
        passed_match = re.search(r'(\d+) passed', output)
        failed_match = re.search(r'(\d+) failed', output)
        
        tests_passed = int(passed_match.group(1)) if passed_match else 0
        tests_failed = int(failed_match.group(1)) if failed_match else 0
        tests_run = tests_passed + tests_failed
        
        return tests_run, tests_passed, tests_failed
    
    def _parse_coverage(self, output: str) -> Optional[float]:
        """
        Parse coverage percentage from pytest-cov output.
        
        Args:
            output: Pytest output with coverage
            
        Returns:
            Coverage percentage or None
        """
        import re
        
        # Look for coverage line: "TOTAL    100    20    80%"
        match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', output)
        if match:
            return float(match.group(1))
        
        return None
    
    def _install_dependencies(self, deps: List[str]) -> None:
        """
        Install additional dependencies in sandbox.
        
        Args:
            deps: List of package names
        """
        for dep in deps:
            try:
                subprocess.run(
                    ["pip", "install", "-q", dep],
                    check=True,
                    capture_output=True,
                    timeout=60
                )
                console.print(f"[green]✓[/green] Installed {dep}")
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to install {dep}: {e}[/yellow]")
    
    def cleanup(self) -> None:
        """Clean up temporary files and Docker resources."""
        import shutil
        
        # Clean up Docker if used
        if self.use_docker and self.docker_sandbox:
            try:
                self.docker_sandbox.cleanup()
            except Exception as e:
                console.print(f"[yellow]Warning: Docker cleanup failed: {e}[/yellow]")
        
        # Clean up tempfile directory
        try:
            shutil.rmtree(self.temp_dir)
            console.print(f"[green]✓[/green] Cleaned up sandbox: {self.temp_dir}")
        except Exception as e:
            console.print(f"[yellow]Warning: Cleanup failed: {e}[/yellow]")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


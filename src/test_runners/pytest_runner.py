"""PyTest test runner for Python tests."""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from src.test_runners.base import BaseTestRunner, TestRunnerResult


class PyTestRunner(BaseTestRunner):
    """Test runner for PyTest framework."""
    
    @property
    def framework_name(self) -> str:
        return "pytest"
    
    @property
    def supported_extensions(self) -> List[str]:
        return [".py"]
    
    def run_tests(
        self,
        test_file: Path,
        source_file: Optional[Path] = None,
        with_coverage: bool = False
    ) -> TestRunnerResult:
        """Run PyTest tests from file."""
        cmd = ["pytest", str(test_file), "-v", "--tb=short"]
        
        if with_coverage and source_file:
            cmd.extend([f"--cov={source_file.stem}", "--cov-report=term-missing"])
        
        return self._execute_command(cmd)
    
    def run_tests_from_code(
        self,
        test_code: str,
        source_code: Optional[str] = None,
        with_coverage: bool = False
    ) -> TestRunnerResult:
        """Run PyTest tests from code strings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write test file
            test_file = temp_path / "test_generated.py"
            test_file.write_text(test_code, encoding='utf-8')
            
            # Write source file if provided
            source_file = None
            if source_code:
                source_file = temp_path / "source_module.py"
                source_file.write_text(source_code, encoding='utf-8')
            
            return self.run_tests(test_file, source_file, with_coverage)
    
    def validate_test_code(self, test_code: str) -> bool:
        """Validate Python test syntax."""
        import ast
        try:
            ast.parse(test_code)
            return True
        except SyntaxError:
            return False
    
    def _execute_command(self, cmd: List[str]) -> TestRunnerResult:
        """Execute pytest command."""
        import re
        import time
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            execution_time = time.time() - start_time
            
            # Parse output
            tests_run, tests_passed, tests_failed = self._parse_output(result.stdout)
            coverage = self._parse_coverage(result.stdout)
            
            return TestRunnerResult(
                success=result.returncode == 0,
                framework="pytest",
                tests_run=tests_run,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                execution_time=execution_time,
                stdout=result.stdout,
                stderr=result.stderr,
                coverage=coverage,
                exit_code=result.returncode
            )
        
        except subprocess.TimeoutExpired:
            return TestRunnerResult(
                success=False,
                framework="pytest",
                stdout="",
                stderr="Test execution timed out",
                exit_code=-1
            )
    
    def _parse_output(self, output: str) -> tuple:
        """Parse pytest output for test statistics."""
        import re
        
        passed_match = re.search(r'(\d+) passed', output)
        failed_match = re.search(r'(\d+) failed', output)
        
        tests_passed = int(passed_match.group(1)) if passed_match else 0
        tests_failed = int(failed_match.group(1)) if failed_match else 0
        tests_run = tests_passed + tests_failed
        
        return tests_run, tests_passed, tests_failed
    
    def _parse_coverage(self, output: str) -> Optional[float]:
        """Parse coverage from pytest-cov output."""
        import re
        
        match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', output)
        return float(match.group(1)) if match else None


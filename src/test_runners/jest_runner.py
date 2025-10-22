"""Jest test runner for JavaScript/TypeScript tests."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from src.test_runners.base import BaseTestRunner, TestRunnerResult


class JestRunner(BaseTestRunner):
    """Test runner for Jest framework (JavaScript/TypeScript)."""
    
    @property
    def framework_name(self) -> str:
        return "jest"
    
    @property
    def supported_extensions(self) -> List[str]:
        return [".js", ".ts", ".jsx", ".tsx"]
    
    def run_tests(
        self,
        test_file: Path,
        source_file: Optional[Path] = None,
        with_coverage: bool = False
    ) -> TestRunnerResult:
        """Run Jest tests from file."""
        cmd = ["jest", str(test_file), "--no-coverage", "--json"]
        
        if with_coverage:
            cmd[cmd.index("--no-coverage")] = "--coverage"
        
        return self._execute_command(cmd)
    
    def run_tests_from_code(
        self,
        test_code: str,
        source_code: Optional[str] = None,
        with_coverage: bool = False
    ) -> TestRunnerResult:
        """Run Jest tests from code strings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write test file
            test_file = temp_path / "test.spec.js"
            test_file.write_text(test_code, encoding='utf-8')
            
            # Write source file if provided
            if source_code:
                source_file = temp_path / "source.js"
                source_file.write_text(source_code, encoding='utf-8')
            
            # Create minimal package.json
            package_json = {
                "name": "test-run",
                "version": "1.0.0",
                "scripts": {"test": "jest"}
            }
            (temp_path / "package.json").write_text(json.dumps(package_json))
            
            return self.run_tests(test_file, None, with_coverage)
    
    def validate_test_code(self, test_code: str) -> bool:
        """Validate JavaScript/TypeScript syntax (simplified)."""
        # Basic syntax check - in production, use a proper parser
        return "describe(" in test_code or "test(" in test_code or "it(" in test_code
    
    def _execute_command(self, cmd: List[str]) -> TestRunnerResult:
        """Execute Jest command."""
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
            
            # Parse JSON output
            try:
                data = json.loads(result.stdout)
                tests_run = data.get("numTotalTests", 0)
                tests_passed = data.get("numPassedTests", 0)
                tests_failed = data.get("numFailedTests", 0)
                coverage_pct = None
                
                if "coverageMap" in data:
                    coverage_pct = self._calculate_coverage(data["coverageMap"])
            
            except json.JSONDecodeError:
                tests_run, tests_passed, tests_failed = 0, 0, 0
                coverage_pct = None
            
            return TestRunnerResult(
                success=result.returncode == 0,
                framework="jest",
                tests_run=tests_run,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                execution_time=execution_time,
                stdout=result.stdout,
                stderr=result.stderr,
                coverage=coverage_pct,
                exit_code=result.returncode
            )
        
        except subprocess.TimeoutExpired:
            return TestRunnerResult(
                success=False,
                framework="jest",
                stdout="",
                stderr="Test execution timed out",
                exit_code=-1
            )
    
    def _calculate_coverage(self, coverage_map: dict) -> Optional[float]:
        """Calculate average coverage from Jest coverage map."""
        if not coverage_map:
            return None
        
        total_lines = 0
        covered_lines = 0
        
        for file_data in coverage_map.values():
            statements = file_data.get("s", {})
            total_lines += len(statements)
            covered_lines += sum(1 for count in statements.values() if count > 0)
        
        return (covered_lines / total_lines * 100) if total_lines > 0 else None


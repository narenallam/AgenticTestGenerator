"""JUnit test runner for Java tests."""

import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

from src.test_runners.base import BaseTestRunner, TestRunnerResult


class JUnitRunner(BaseTestRunner):
    """Test runner for JUnit framework (Java)."""
    
    @property
    def framework_name(self) -> str:
        return "junit"
    
    @property
    def supported_extensions(self) -> List[str]:
        return [".java"]
    
    def run_tests(
        self,
        test_file: Path,
        source_file: Optional[Path] = None,
        with_coverage: bool = False
    ) -> TestRunnerResult:
        """Run JUnit tests from file."""
        # Compile and run Java tests
        temp_dir = test_file.parent
        
        # Compile
        compile_cmd = [
            "javac",
            "-cp", ".:junit-platform-console-standalone.jar",
            str(test_file)
        ]
        
        if source_file:
            compile_cmd.append(str(source_file))
        
        subprocess.run(compile_cmd, capture_output=True, timeout=30)
        
        # Run tests
        test_class = test_file.stem
        run_cmd = [
            "java",
            "-jar", "junit-platform-console-standalone.jar",
            "-cp", str(temp_dir),
            "--select-class", test_class,
            "--reports-dir", str(temp_dir / "reports")
        ]
        
        return self._execute_command(run_cmd, temp_dir)
    
    def run_tests_from_code(
        self,
        test_code: str,
        source_code: Optional[str] = None,
        with_coverage: bool = False
    ) -> TestRunnerResult:
        """Run JUnit tests from code strings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract class name from test code
            import re
            class_match = re.search(r'public class (\w+)', test_code)
            class_name = class_match.group(1) if class_match else "GeneratedTest"
            
            # Write test file
            test_file = temp_path / f"{class_name}.java"
            test_file.write_text(test_code, encoding='utf-8')
            
            # Write source file if provided
            source_file = None
            if source_code:
                src_class_match = re.search(r'public class (\w+)', source_code)
                src_class_name = src_class_match.group(1) if src_class_match else "Source"
                source_file = temp_path / f"{src_class_name}.java"
                source_file.write_text(source_code, encoding='utf-8')
            
            return self.run_tests(test_file, source_file, with_coverage)
    
    def validate_test_code(self, test_code: str) -> bool:
        """Validate Java test syntax (simplified)."""
        return ("@Test" in test_code and
                "public class" in test_code and
                "import org.junit" in test_code)
    
    def _execute_command(self, cmd: List[str], temp_dir: Path) -> TestRunnerResult:
        """Execute JUnit command."""
        import time
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(temp_dir)
            )
            
            execution_time = time.time() - start_time
            
            # Parse XML report
            tests_run, tests_passed, tests_failed = self._parse_junit_xml(temp_dir / "reports")
            
            return TestRunnerResult(
                success=result.returncode == 0,
                framework="junit",
                tests_run=tests_run,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                execution_time=execution_time,
                stdout=result.stdout,
                stderr=result.stderr,
                coverage=None,
                exit_code=result.returncode
            )
        
        except subprocess.TimeoutExpired:
            return TestRunnerResult(
                success=False,
                framework="junit",
                stdout="",
                stderr="Test execution timed out",
                exit_code=-1
            )
    
    def _parse_junit_xml(self, reports_dir: Path) -> tuple:
        """Parse JUnit XML reports."""
        tests_run = 0
        tests_failed = 0
        
        try:
            for xml_file in reports_dir.glob("TEST-*.xml"):
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                tests_run += int(root.get("tests", 0))
                tests_failed += int(root.get("failures", 0)) + int(root.get("errors", 0))
        
        except Exception:
            pass
        
        tests_passed = tests_run - tests_failed
        return tests_run, tests_passed, tests_failed


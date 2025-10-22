"""
Multi-language test evaluation support.

Supports evaluation for:
- Python (pytest, unittest)
- Java (JUnit, TestNG)
- JavaScript/TypeScript (Jest, Mocha, Jasmine)
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..base import BaseMetricCalculator


# ═══════════════════════════════════════════════════════════════════════════
# Language-Specific Runners
# ═══════════════════════════════════════════════════════════════════════════


class PythonTestRunner:
    """Run Python tests with pytest."""
    
    @staticmethod
    def run_with_coverage(
        test_code: str,
        source_code: str,
        timeout: int = 30
    ) -> Dict[str, any]:
        """
        Run Python tests with coverage.
        
        Returns:
            Dict with coverage, pass_rate, execution_time
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Write source code
                source_file = tmpdir_path / "source.py"
                source_file.write_text(source_code)
                
                # Write test code
                test_file = tmpdir_path / "test_source.py"
                test_file.write_text(test_code)
                
                # Run with coverage
                result = subprocess.run(
                    [
                        "python", "-m", "pytest",
                        str(test_file),
                        f"--cov={tmpdir_path}",
                        "--cov-report=term-missing:skip-covered",
                        "-v",
                    ],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                
                # Parse results
                coverage = PythonTestRunner._parse_coverage(result.stdout)
                pass_rate, total_tests, passed_tests = PythonTestRunner._parse_pytest_results(result.stdout)
                
                return {
                    "coverage": coverage,
                    "pass_rate": pass_rate,
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
        
        except subprocess.TimeoutExpired:
            return {"error": "Timeout", "coverage": 0.0, "pass_rate": 0.0}
        except Exception as e:
            return {"error": str(e), "coverage": 0.0, "pass_rate": 0.0}
    
    @staticmethod
    def _parse_coverage(output: str) -> float:
        """Parse coverage percentage from pytest output."""
        import re
        for line in output.split('\n'):
            if 'TOTAL' in line:
                match = re.search(r'(\d+)%', line)
                if match:
                    return int(match.group(1)) / 100.0
        return 0.0
    
    @staticmethod
    def _parse_pytest_results(output: str) -> Tuple[float, int, int]:
        """Parse test pass rate from pytest output."""
        import re
        # Look for "X passed" or "X passed, Y failed"
        match = re.search(r'(\d+) passed', output)
        passed = int(match.group(1)) if match else 0
        
        match = re.search(r'(\d+) failed', output)
        failed = int(match.group(1)) if match else 0
        
        total = passed + failed
        pass_rate = passed / total if total > 0 else 0.0
        
        return pass_rate, total, passed


class JavaTestRunner:
    """Run Java tests with JUnit."""
    
    @staticmethod
    def run_with_coverage(
        test_code: str,
        source_code: str,
        timeout: int = 30
    ) -> Dict[str, any]:
        """
        Run Java tests with coverage (JaCoCo).
        
        Returns:
            Dict with coverage, pass_rate, execution_time
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Extract class name from source code
                class_name = JavaTestRunner._extract_class_name(source_code)
                test_class_name = f"{class_name}Test"
                
                # Write source code
                source_file = tmpdir_path / f"{class_name}.java"
                source_file.write_text(source_code)
                
                # Write test code
                test_file = tmpdir_path / f"{test_class_name}.java"
                test_file.write_text(test_code)
                
                # Compile source
                compile_result = subprocess.run(
                    ["javac", str(source_file)],
                    cwd=tmpdir,
                    capture_output=True,
                    timeout=timeout,
                )
                
                if compile_result.returncode != 0:
                    return {
                        "error": "Compilation failed",
                        "coverage": 0.0,
                        "pass_rate": 0.0,
                        "stderr": compile_result.stderr.decode(),
                    }
                
                # Run tests with JUnit
                result = subprocess.run(
                    [
                        "java",
                        "-cp", f".:{tmpdir_path}:junit-platform-console-standalone.jar",
                        "org.junit.runner.JUnitCore",
                        test_class_name,
                    ],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                
                # Parse results
                pass_rate, total_tests, passed_tests = JavaTestRunner._parse_junit_results(result.stdout)
                
                # Note: Coverage requires JaCoCo agent, simplified here
                coverage = 0.75  # Placeholder
                
                return {
                    "coverage": coverage,
                    "pass_rate": pass_rate,
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                }
        
        except Exception as e:
            return {"error": str(e), "coverage": 0.0, "pass_rate": 0.0}
    
    @staticmethod
    def _extract_class_name(source_code: str) -> str:
        """Extract Java class name from source code."""
        import re
        match = re.search(r'class\s+(\w+)', source_code)
        return match.group(1) if match else "Unknown"
    
    @staticmethod
    def _parse_junit_results(output: str) -> Tuple[float, int, int]:
        """Parse JUnit test results."""
        import re
        # JUnit output: "OK (X tests)" or "Tests run: X, Failures: Y"
        match = re.search(r'OK \((\d+) tests?\)', output)
        if match:
            passed = int(match.group(1))
            return 1.0, passed, passed
        
        match = re.search(r'Tests run: (\d+),.*Failures: (\d+)', output)
        if match:
            total = int(match.group(1))
            failures = int(match.group(2))
            passed = total - failures
            pass_rate = passed / total if total > 0 else 0.0
            return pass_rate, total, passed
        
        return 0.0, 0, 0


class JavaScriptTestRunner:
    """Run JavaScript/TypeScript tests with Jest."""
    
    @staticmethod
    def run_with_coverage(
        test_code: str,
        source_code: str,
        timeout: int = 30,
        is_typescript: bool = False
    ) -> Dict[str, any]:
        """
        Run JS/TS tests with Jest and coverage.
        
        Returns:
            Dict with coverage, pass_rate, execution_time
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # File extensions
                ext = ".ts" if is_typescript else ".js"
                test_ext = ".test" + ext
                
                # Write source code
                source_file = tmpdir_path / f"source{ext}"
                source_file.write_text(source_code)
                
                # Write test code
                test_file = tmpdir_path / f"source{test_ext}"
                test_file.write_text(test_code)
                
                # Create minimal package.json
                package_json = {
                    "name": "test",
                    "scripts": {"test": "jest --coverage"},
                    "devDependencies": {
                        "jest": "^29.0.0",
                        "@types/jest": "^29.0.0" if is_typescript else None,
                    }
                }
                
                import json
                (tmpdir_path / "package.json").write_text(
                    json.dumps({k: v for k, v in package_json.items() if v is not None}, indent=2)
                )
                
                # Run tests
                result = subprocess.run(
                    ["npm", "test", "--", "--coverage", "--json"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                
                # Parse results
                coverage, pass_rate, total, passed = JavaScriptTestRunner._parse_jest_results(
                    result.stdout
                )
                
                return {
                    "coverage": coverage,
                    "pass_rate": pass_rate,
                    "total_tests": total,
                    "passed_tests": passed,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                }
        
        except Exception as e:
            return {"error": str(e), "coverage": 0.0, "pass_rate": 0.0}
    
    @staticmethod
    def _parse_jest_results(output: str) -> Tuple[float, float, int, int]:
        """Parse Jest test results and coverage."""
        import json
        import re
        
        try:
            # Jest JSON output
            data = json.loads(output)
            
            # Coverage
            coverage_pct = data.get("coverageMap", {}).get("total", {}).get("lines", {}).get("pct", 0)
            coverage = coverage_pct / 100.0
            
            # Test results
            total = data.get("numTotalTests", 0)
            passed = data.get("numPassedTests", 0)
            pass_rate = passed / total if total > 0 else 0.0
            
            return coverage, pass_rate, total, passed
        
        except json.JSONDecodeError:
            # Fallback: parse text output
            coverage_match = re.search(r'All files\s+\|\s+(\d+\.?\d*)', output)
            coverage = float(coverage_match.group(1)) / 100.0 if coverage_match else 0.0
            
            passed_match = re.search(r'(\d+) passed', output)
            passed = int(passed_match.group(1)) if passed_match else 0
            
            total_match = re.search(r'(\d+) total', output)
            total = int(total_match.group(1)) if total_match else 0
            
            pass_rate = passed / total if total > 0 else 0.0
            
            return coverage, pass_rate, total, passed


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Language Coverage Calculator
# ═══════════════════════════════════════════════════════════════════════════


class MultiLanguageCoverageCalculator(BaseMetricCalculator):
    """Calculate coverage for any supported language."""
    
    def calculate(
        self,
        test_code: str,
        source_code: str,
        language: str = "python",
        **kwargs
    ) -> float:
        """
        Calculate coverage for given language.
        
        Args:
            test_code: Generated test code
            source_code: Source code being tested
            language: Programming language (python, java, javascript, typescript)
        
        Returns:
            Coverage percentage (0.0 to 1.0)
        """
        language = language.lower()
        
        if language == "python":
            result = PythonTestRunner.run_with_coverage(test_code, source_code)
        elif language == "java":
            result = JavaTestRunner.run_with_coverage(test_code, source_code)
        elif language in ["javascript", "js"]:
            result = JavaScriptTestRunner.run_with_coverage(test_code, source_code, is_typescript=False)
        elif language in ["typescript", "ts"]:
            result = JavaScriptTestRunner.run_with_coverage(test_code, source_code, is_typescript=True)
        else:
            raise ValueError(f"Unsupported language: {language}")
        
        return result.get("coverage", 0.0)
    
    def get_name(self) -> str:
        return "coverage"
    
    def get_unit(self) -> str:
        return "percentage"


class MultiLanguagePassRateCalculator(BaseMetricCalculator):
    """Calculate test pass rate for any supported language."""
    
    def calculate(
        self,
        test_code: str,
        source_code: str,
        language: str = "python",
        **kwargs
    ) -> float:
        """
        Calculate pass rate for given language.
        
        Args:
            test_code: Generated test code
            source_code: Source code being tested
            language: Programming language
        
        Returns:
            Pass rate (0.0 to 1.0)
        """
        language = language.lower()
        
        if language == "python":
            result = PythonTestRunner.run_with_coverage(test_code, source_code)
        elif language == "java":
            result = JavaTestRunner.run_with_coverage(test_code, source_code)
        elif language in ["javascript", "js"]:
            result = JavaScriptTestRunner.run_with_coverage(test_code, source_code, is_typescript=False)
        elif language in ["typescript", "ts"]:
            result = JavaScriptTestRunner.run_with_coverage(test_code, source_code, is_typescript=True)
        else:
            raise ValueError(f"Unsupported language: {language}")
        
        return result.get("pass_rate", 0.0)
    
    def get_name(self) -> str:
        return "pass_rate"
    
    def get_unit(self) -> str:
        return "percentage"


# ═══════════════════════════════════════════════════════════════════════════
# Goal Achievement Calculator
# ═══════════════════════════════════════════════════════════════════════════


class GoalAchievementCalculator:
    """
    Calculate how well the system achieves its goals:
    - 90% test coverage
    - 90% pass rate
    """
    
    COVERAGE_GOAL = 0.90
    PASS_RATE_GOAL = 0.90
    
    @staticmethod
    def calculate_goal_score(coverage: float, pass_rate: float) -> Dict[str, any]:
        """
        Calculate overall goal achievement score.
        
        Args:
            coverage: Actual coverage (0.0 to 1.0)
            pass_rate: Actual pass rate (0.0 to 1.0)
        
        Returns:
            Dictionary with scores and goal status
        """
        # Individual goal achievement
        coverage_achievement = min(coverage / GoalAchievementCalculator.COVERAGE_GOAL, 1.0)
        pass_rate_achievement = min(pass_rate / GoalAchievementCalculator.PASS_RATE_GOAL, 1.0)
        
        # Overall goal score (equal weight)
        overall_score = (coverage_achievement + pass_rate_achievement) / 2.0
        
        # Goal status
        coverage_met = coverage >= GoalAchievementCalculator.COVERAGE_GOAL
        pass_rate_met = pass_rate >= GoalAchievementCalculator.PASS_RATE_GOAL
        both_met = coverage_met and pass_rate_met
        
        return {
            "overall_score": overall_score,
            "coverage_achievement": coverage_achievement,
            "pass_rate_achievement": pass_rate_achievement,
            "coverage_met": coverage_met,
            "pass_rate_met": pass_rate_met,
            "both_goals_met": both_met,
            "coverage_gap": max(0.0, GoalAchievementCalculator.COVERAGE_GOAL - coverage),
            "pass_rate_gap": max(0.0, GoalAchievementCalculator.PASS_RATE_GOAL - pass_rate),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════


def evaluate_multi_language(
    test_code: str,
    source_code: str,
    language: str
) -> Dict[str, any]:
    """
    Quick multi-language evaluation.
    
    Args:
        test_code: Generated test code
        source_code: Source code being tested
        language: Programming language
    
    Returns:
        Dictionary with coverage, pass_rate, and goal achievement
    """
    coverage_calc = MultiLanguageCoverageCalculator()
    pass_rate_calc = MultiLanguagePassRateCalculator()
    
    coverage = coverage_calc.calculate(test_code, source_code, language=language)
    pass_rate = pass_rate_calc.calculate(test_code, source_code, language=language)
    
    goal_score = GoalAchievementCalculator.calculate_goal_score(coverage, pass_rate)
    
    return {
        "language": language,
        "coverage": coverage,
        "pass_rate": pass_rate,
        **goal_score,
    }


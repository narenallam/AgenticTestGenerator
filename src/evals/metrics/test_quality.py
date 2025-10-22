"""
Test quality metrics evaluation.

Measures the quality of generated tests across multiple dimensions:
- Correctness
- Coverage
- Completeness
- Determinism
- Assertions
- Mocking
- Readability
"""

import ast
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..base import BaseEvaluator, BaseMetricCalculator, EvalLevel, EvalResult, generate_eval_id


# ═══════════════════════════════════════════════════════════════════════════
# Individual Metric Calculators
# ═══════════════════════════════════════════════════════════════════════════


class CorrectnessCalculator(BaseMetricCalculator):
    """Calculate test correctness (syntax + execution)."""
    
    def calculate(self, test_code: str, source_code: str) -> float:
        """
        Calculate correctness score.
        
        Args:
            test_code: Generated test code
            source_code: Source code being tested
        
        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        
        # Check 1: Syntax validity (0.5 points)
        try:
            ast.parse(test_code)
            score += 0.5
        except SyntaxError:
            return 0.0  # Syntax errors are fatal
        
        # Check 2: Execution (0.5 points)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Write source code
                source_file = tmpdir_path / "source.py"
                source_file.write_text(source_code)
                
                # Write test code
                test_file = tmpdir_path / "test_source.py"
                test_file.write_text(test_code)
                
                # Run tests
                result = subprocess.run(
                    ["python", "-m", "pytest", str(test_file), "-v"],
                    cwd=tmpdir,
                    capture_output=True,
                    timeout=10,
                )
                
                # If tests pass or fail (but execute), add points
                if result.returncode in [0, 1]:  # 0 = pass, 1 = fail
                    score += 0.5
        
        except Exception:
            pass  # Execution failed
        
        return score
    
    def get_name(self) -> str:
        return "correctness"
    
    def get_unit(self) -> str:
        return "ratio"


class CoverageCalculator(BaseMetricCalculator):
    """Calculate code coverage achieved by tests."""
    
    def calculate(self, test_code: str, source_code: str) -> float:
        """
        Calculate coverage score.
        
        Args:
            test_code: Generated test code
            source_code: Source code being tested
        
        Returns:
            Coverage percentage (0.0 to 1.0)
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
                        "--cov-report=term-missing",
                    ],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                
                # Parse coverage from output
                # Look for "TOTAL ... XX%"
                for line in result.stdout.split('\n'):
                    if 'TOTAL' in line:
                        # Extract percentage
                        match = re.search(r'(\d+)%', line)
                        if match:
                            coverage = int(match.group(1)) / 100.0
                            return coverage
                
                return 0.0
        
        except Exception:
            return 0.0
    
    def get_name(self) -> str:
        return "coverage"
    
    def get_unit(self) -> str:
        return "percentage"


class CompletenessCalculator(BaseMetricCalculator):
    """Calculate test completeness (edge cases, error paths)."""
    
    def calculate(self, test_code: str, source_code: str) -> float:
        """
        Calculate completeness score.
        
        Args:
            test_code: Generated test code
            source_code: Source code being tested
        
        Returns:
            Score between 0.0 and 1.0
        """
        try:
            test_tree = ast.parse(test_code)
            source_tree = ast.parse(source_code)
        except SyntaxError:
            return 0.0
        
        score = 0.0
        
        # Count test functions
        test_functions = [
            node for node in ast.walk(test_tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')
        ]
        
        # Check 1: Multiple test cases (0.3 points)
        if len(test_functions) >= 3:
            score += 0.3
        elif len(test_functions) >= 2:
            score += 0.15
        elif len(test_functions) >= 1:
            score += 0.05
        
        # Check 2: Tests exceptions (0.3 points)
        has_exception_test = any(
            isinstance(node, ast.Raise) or
            (isinstance(node, ast.Name) and 'raises' in str(node))
            for node in ast.walk(test_tree)
        )
        if has_exception_test:
            score += 0.3
        
        # Check 3: Edge case tests (0.2 points)
        # Look for empty lists, None, 0, negative numbers, etc.
        edge_case_indicators = ['None', '[]', '{}', '0', '-1', 'empty']
        test_code_lower = test_code.lower()
        if any(indicator.lower() in test_code_lower for indicator in edge_case_indicators):
            score += 0.2
        
        # Check 4: Parametrized tests (0.2 points)
        if '@pytest.mark.parametrize' in test_code or '@parameterized' in test_code:
            score += 0.2
        
        return min(score, 1.0)
    
    def get_name(self) -> str:
        return "completeness"
    
    def get_unit(self) -> str:
        return "ratio"


class DeterminismCalculator(BaseMetricCalculator):
    """Check if tests are deterministic."""
    
    NONDETERMINISTIC_PATTERNS = [
        r'time\.sleep',
        r'datetime\.now\(\)',
        r'random\.',
        r'uuid\.uuid4',
        r'requests\.',
    ]
    
    MOCK_PATTERNS = [
        r'mock\.',
        r'@patch',
        r'monkeypatch',
        r'freezegun',
        r'\.seed\(',
    ]
    
    def calculate(self, test_code: str, source_code: str = "") -> float:
        """
        Calculate determinism score.
        
        Args:
            test_code: Generated test code
            source_code: Source code (unused here)
        
        Returns:
            Score between 0.0 and 1.0
        """
        score = 1.0
        
        # Check for non-deterministic patterns
        violations = 0
        for pattern in self.NONDETERMINISTIC_PATTERNS:
            if re.search(pattern, test_code):
                violations += 1
        
        # Check if mocking is used
        has_mocking = any(
            re.search(pattern, test_code)
            for pattern in self.MOCK_PATTERNS
        )
        
        # If violations but mocking is used, it's OK
        if violations > 0 and not has_mocking:
            score = max(0.0, 1.0 - (violations * 0.2))
        
        return score
    
    def get_name(self) -> str:
        return "determinism"
    
    def get_unit(self) -> str:
        return "ratio"


class AssertionCalculator(BaseMetricCalculator):
    """Count and evaluate assertions in tests."""
    
    def calculate(self, test_code: str, source_code: str = "") -> float:
        """
        Calculate assertion quality score.
        
        Args:
            test_code: Generated test code
            source_code: Source code (unused here)
        
        Returns:
            Score between 0.0 and 1.0
        """
        try:
            tree = ast.parse(test_code)
        except SyntaxError:
            return 0.0
        
        # Count assertions
        assertions = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                assertions += 1
            elif isinstance(node, ast.Call):
                # Check for pytest/unittest assertions
                func_name = ""
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id
                
                if func_name.startswith('assert'):
                    assertions += 1
        
        # Score based on assertion count
        # 0 assertions = 0.0
        # 1-2 assertions = 0.5
        # 3+ assertions = 1.0
        if assertions == 0:
            return 0.0
        elif assertions <= 2:
            return 0.5
        else:
            return 1.0
    
    def get_name(self) -> str:
        return "assertions"
    
    def get_unit(self) -> str:
        return "ratio"


class MockingCalculator(BaseMetricCalculator):
    """Check if proper mocking is used."""
    
    def calculate(self, test_code: str, source_code: str) -> float:
        """
        Calculate mocking score.
        
        Args:
            test_code: Generated test code
            source_code: Source code being tested
        
        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        
        # Check if source code has external dependencies
        external_indicators = [
            'requests', 'urllib', 'http',
            'database', 'db', 'sql',
            'open(', 'file',
            'datetime', 'time',
            'random',
        ]
        
        needs_mocking = any(
            indicator in source_code.lower()
            for indicator in external_indicators
        )
        
        if not needs_mocking:
            # No mocking needed, perfect score
            return 1.0
        
        # Check if mocking is used
        mock_indicators = [
            'mock', '@patch', 'monkeypatch',
            'Mock()', 'MagicMock',
            'freezegun', 'responses',
        ]
        
        has_mocking = any(
            indicator in test_code
            for indicator in mock_indicators
        )
        
        if has_mocking:
            score = 1.0
        else:
            score = 0.3  # Partial credit for attempting tests
        
        return score
    
    def get_name(self) -> str:
        return "mocking"
    
    def get_unit(self) -> str:
        return "ratio"


# ═══════════════════════════════════════════════════════════════════════════
# Test Quality Evaluator
# ═══════════════════════════════════════════════════════════════════════════


class TestQualityEvaluator(BaseEvaluator):
    """Comprehensive test quality evaluation."""
    
    def __init__(self):
        """Initialize test quality evaluator."""
        super().__init__(name="test_quality", level=EvalLevel.COMPONENT)
        
        # Initialize metric calculators
        self.calculators = {
            "correctness": CorrectnessCalculator(),
            "coverage": CoverageCalculator(),
            "completeness": CompletenessCalculator(),
            "determinism": DeterminismCalculator(),
            "assertions": AssertionCalculator(),
            "mocking": MockingCalculator(),
        }
        
        # Metric weights for overall score
        self.weights = {
            "correctness": 0.30,   # Most important
            "coverage": 0.25,
            "completeness": 0.20,
            "determinism": 0.10,
            "assertions": 0.10,
            "mocking": 0.05,
        }
    
    def evaluate(
        self,
        test_code: str,
        source_code: str,
        eval_id: Optional[str] = None,
    ) -> EvalResult:
        """
        Evaluate test quality.
        
        Args:
            test_code: Generated test code
            source_code: Source code being tested
            eval_id: Optional evaluation ID
        
        Returns:
            EvalResult with all metrics
        """
        if eval_id is None:
            eval_id = generate_eval_id("test_quality")
        
        result = self.create_result(eval_id)
        
        try:
            # Calculate all metrics
            for name, calculator in self.calculators.items():
                try:
                    value = calculator.calculate(test_code, source_code)
                    result.add_metric(
                        name=name,
                        value=value,
                        unit=calculator.get_unit(),
                        description=f"{name.capitalize()} score",
                    )
                except Exception as e:
                    result.warnings.append(f"Failed to calculate {name}: {e}")
                    result.add_metric(name=name, value=0.0, unit=calculator.get_unit())
            
            # Calculate overall score
            result.calculate_score(weights=self.weights)
            
            # Add recommendations
            result.recommendations.extend(self._generate_recommendations(result))
            
            result.mark_completed()
        
        except Exception as e:
            result.mark_failed(str(e))
        
        return result
    
    def _generate_recommendations(self, result: EvalResult) -> List[str]:
        """Generate improvement recommendations based on metrics."""
        recommendations = []
        
        for name, metric in result.metrics.items():
            if metric.value < 0.7:
                if name == "correctness":
                    recommendations.append("⚠️ Fix syntax errors and ensure tests execute")
                elif name == "coverage":
                    recommendations.append("⚠️ Add more tests to increase code coverage")
                elif name == "completeness":
                    recommendations.append("⚠️ Add tests for edge cases and error paths")
                elif name == "determinism":
                    recommendations.append("⚠️ Use mocks/fixtures for non-deterministic operations")
                elif name == "assertions":
                    recommendations.append("⚠️ Add more assertions to verify behavior")
                elif name == "mocking":
                    recommendations.append("⚠️ Mock external dependencies properly")
        
        if not recommendations:
            recommendations.append("✅ Test quality is excellent!")
        
        return recommendations


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════


def quick_eval(test_code: str, source_code: str) -> Dict[str, float]:
    """
    Quick evaluation of test quality.
    
    Args:
        test_code: Generated test code
        source_code: Source code being tested
    
    Returns:
        Dictionary of metric scores
    """
    evaluator = TestQualityEvaluator()
    result = evaluator.evaluate(test_code, source_code)
    
    return {
        "overall": result.score or 0.0,
        **{name: metric.value for name, metric in result.metrics.items()}
    }


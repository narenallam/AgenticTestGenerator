"""
Lightweight classifiers for test intelligence.

This module provides fast, accurate classifiers for:
- Failure triage (categorizing test failures)
- Framework detection (pytest, unittest, jest, junit)
- Flaky test prediction (likelihood of flakiness)
"""

import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from rich.console import Console

console = Console()


class FailureCategory(str, Enum):
    """Categories of test failures."""
    
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    ASSERTION_ERROR = "assertion_error"
    TIMEOUT = "timeout"
    DEPENDENCY_ERROR = "dependency_error"
    LOGIC_ERROR = "logic_error"
    RUNTIME_ERROR = "runtime_error"
    NETWORK_ERROR = "network_error"
    FILE_ERROR = "file_error"
    UNKNOWN = "unknown"


class TestFramework(str, Enum):
    """Supported test frameworks."""
    
    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    JUNIT = "junit"
    MOCHA = "mocha"
    UNKNOWN = "unknown"


class FailureTriageResult(BaseModel):
    """Result of failure triage classification."""
    
    category: FailureCategory = Field(..., description="Failure category")
    confidence: float = Field(..., description="Classification confidence (0-1)")
    error_message: str = Field(..., description="Error message")
    suggested_fix: Optional[str] = Field(default=None, description="Suggested fix")
    requires_refinement: bool = Field(..., description="Requires test refinement")


class FrameworkDetectionResult(BaseModel):
    """Result of framework detection."""
    
    framework: TestFramework = Field(..., description="Detected framework")
    confidence: float = Field(..., description="Detection confidence (0-1)")
    language: str = Field(..., description="Programming language")
    indicators: List[str] = Field(default_factory=list, description="Detection indicators")


class FlakyPredictionResult(BaseModel):
    """Result of flaky test prediction."""
    
    is_likely_flaky: bool = Field(..., description="Test is likely flaky")
    flakiness_score: float = Field(..., description="Flakiness score (0-1)")
    risk_factors: List[str] = Field(default_factory=list, description="Risk factors")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


class FailureTriageClassifier:
    """
    Classifier for categorizing test failures.
    
    Uses pattern matching and heuristics for fast, accurate classification.
    """
    
    # Error patterns for classification
    PATTERNS = {
        FailureCategory.SYNTAX_ERROR: [
            r"SyntaxError:",
            r"IndentationError:",
            r"TabError:",
            r"invalid syntax",
            r"unexpected EOF"
        ],
        FailureCategory.IMPORT_ERROR: [
            r"ImportError:",
            r"ModuleNotFoundError:",
            r"cannot import name",
            r"No module named",
            r"import .* failed"
        ],
        FailureCategory.ASSERTION_ERROR: [
            r"AssertionError:",
            r"assert .* failed",
            r"Expected .*, got",
            r"should (?:equal|be|have)",
            r"Expected: .* Actual:"
        ],
        FailureCategory.TIMEOUT: [
            r"TimeoutError:",
            r"timeout exceeded",
            r"timed out",
            r"execution time limit",
            r"killed after .* seconds"
        ],
        FailureCategory.DEPENDENCY_ERROR: [
            r"ConnectionError:",
            r"Database",
            r"connection refused",
            r"could not connect",
            r"service unavailable"
        ],
        FailureCategory.NETWORK_ERROR: [
            r"NetworkError:",
            r"URLError:",
            r"HTTPError:",
            r"connection reset",
            r"network unreachable"
        ],
        FailureCategory.FILE_ERROR: [
            r"FileNotFoundError:",
            r"PermissionError:",
            r"IOError:",
            r"No such file",
            r"permission denied"
        ],
        FailureCategory.RUNTIME_ERROR: [
            r"RuntimeError:",
            r"MemoryError:",
            r"RecursionError:",
            r"OverflowError:",
            r"maximum recursion depth"
        ]
    }
    
    # Suggested fixes for each category
    FIXES = {
        FailureCategory.SYNTAX_ERROR: "Fix syntax errors in the generated test code",
        FailureCategory.IMPORT_ERROR: "Add missing imports or install required packages",
        FailureCategory.ASSERTION_ERROR: "Review assertion logic and expected values",
        FailureCategory.TIMEOUT: "Optimize test execution or increase timeout",
        FailureCategory.DEPENDENCY_ERROR: "Mock external dependencies",
        FailureCategory.NETWORK_ERROR: "Mock network calls and external APIs",
        FailureCategory.FILE_ERROR: "Mock file I/O operations or fix file paths",
        FailureCategory.RUNTIME_ERROR: "Fix recursive logic or memory issues"
    }
    
    def classify(
        self,
        error_message: str,
        stderr: Optional[str] = None
    ) -> FailureTriageResult:
        """
        Classify a test failure.
        
        Args:
            error_message: Primary error message
            stderr: Standard error output (optional)
            
        Returns:
            FailureTriageResult with classification
            
        Example:
            >>> classifier = FailureTriageClassifier()
            >>> result = classifier.classify("ImportError: No module named 'foo'")
            >>> print(f"Category: {result.category}, Confidence: {result.confidence}")
        """
        full_error = f"{error_message}\n{stderr or ''}"
        
        # Try to match patterns
        best_match = None
        best_confidence = 0.0
        
        for category, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, full_error, re.IGNORECASE):
                    confidence = 0.8 + (0.2 * len(re.findall(pattern, full_error, re.IGNORECASE)))
                    confidence = min(confidence, 0.99)
                    
                    if confidence > best_confidence:
                        best_match = category
                        best_confidence = confidence
        
        # Default to LOGIC_ERROR if no match
        if not best_match:
            best_match = FailureCategory.LOGIC_ERROR
            best_confidence = 0.5
        
        # Determine if refinement is needed
        requires_refinement = best_match in [
            FailureCategory.SYNTAX_ERROR,
            FailureCategory.IMPORT_ERROR,
            FailureCategory.ASSERTION_ERROR,
            FailureCategory.LOGIC_ERROR
        ]
        
        return FailureTriageResult(
            category=best_match,
            confidence=best_confidence,
            error_message=error_message,
            suggested_fix=self.FIXES.get(best_match),
            requires_refinement=requires_refinement
        )
    
    def batch_classify(
        self,
        errors: List[str]
    ) -> List[FailureTriageResult]:
        """
        Classify multiple failures.
        
        Args:
            errors: List of error messages
            
        Returns:
            List of classification results
        """
        return [self.classify(error) for error in errors]


class FrameworkDetector:
    """
    Detector for identifying test frameworks and programming languages.
    
    Uses file patterns, imports, and syntax to detect frameworks.
    """
    
    # Indicators for each framework
    INDICATORS = {
        TestFramework.PYTEST: {
            "patterns": [
                r"import pytest",
                r"from pytest",
                r"@pytest\.",
                r"def test_",
                r"pytest\.mark\.",
                r"def conftest"
            ],
            "file_patterns": ["test_*.py", "*_test.py", "conftest.py"],
            "language": "Python"
        },
        TestFramework.UNITTEST: {
            "patterns": [
                r"import unittest",
                r"from unittest",
                r"class \w+\(unittest\.TestCase\)",
                r"def setUp\(",
                r"def tearDown\("
            ],
            "file_patterns": ["test*.py", "*test.py"],
            "language": "Python"
        },
        TestFramework.JEST: {
            "patterns": [
                r"describe\(",
                r"it\(",
                r"test\(",
                r"expect\(",
                r"from ['\"]@jest",
                r"import .* from ['\"]@jest"
            ],
            "file_patterns": ["*.test.js", "*.test.ts", "*.spec.js", "*.spec.ts"],
            "language": "JavaScript/TypeScript"
        },
        TestFramework.JUNIT: {
            "patterns": [
                r"import org\.junit",
                r"@Test",
                r"@Before",
                r"@After",
                r"Assert\.",
                r"assertEquals"
            ],
            "file_patterns": ["*Test.java", "Test*.java"],
            "language": "Java"
        },
        TestFramework.MOCHA: {
            "patterns": [
                r"describe\(",
                r"it\(",
                r"before\(",
                r"after\(",
                r"from ['\"]mocha",
                r"require\(['\"]mocha"
            ],
            "file_patterns": ["*.test.js", "*.spec.js"],
            "language": "JavaScript"
        }
    }
    
    def detect_from_code(self, code: str) -> FrameworkDetectionResult:
        """
        Detect framework from code content.
        
        Args:
            code: Test code content
            
        Returns:
            FrameworkDetectionResult
            
        Example:
            >>> detector = FrameworkDetector()
            >>> result = detector.detect_from_code("import pytest\\ndef test_foo(): pass")
            >>> print(result.framework)
        """
        best_framework = TestFramework.UNKNOWN
        best_confidence = 0.0
        best_indicators = []
        best_language = "Unknown"
        
        for framework, info in self.INDICATORS.items():
            matches = []
            for pattern in info["patterns"]:
                if re.search(pattern, code, re.MULTILINE):
                    matches.append(pattern)
            
            if matches:
                # Calculate confidence based on number of matches
                confidence = min(0.5 + (len(matches) * 0.15), 0.99)
                
                if confidence > best_confidence:
                    best_framework = framework
                    best_confidence = confidence
                    best_indicators = matches
                    best_language = info["language"]
        
        return FrameworkDetectionResult(
            framework=best_framework,
            confidence=best_confidence,
            language=best_language,
            indicators=best_indicators
        )
    
    def detect_from_file(self, file_path: Path) -> FrameworkDetectionResult:
        """
        Detect framework from file path and content.
        
        Args:
            file_path: Path to test file
            
        Returns:
            FrameworkDetectionResult
        """
        # First try file pattern matching
        file_name = file_path.name
        
        for framework, info in self.INDICATORS.items():
            for pattern in info["file_patterns"]:
                if Path(file_name).match(pattern):
                    # Read file and verify with content
                    try:
                        code = file_path.read_text(encoding='utf-8')
                        content_result = self.detect_from_code(code)
                        
                        if content_result.framework == framework or content_result.framework == TestFramework.UNKNOWN:
                            return FrameworkDetectionResult(
                                framework=framework,
                                confidence=0.8,
                                language=info["language"],
                                indicators=[f"File pattern: {pattern}"]
                            )
                    except:
                        pass
        
        # Fall back to content detection
        try:
            code = file_path.read_text(encoding='utf-8')
            return self.detect_from_code(code)
        except:
            return FrameworkDetectionResult(
                framework=TestFramework.UNKNOWN,
                confidence=0.0,
                language="Unknown",
                indicators=[]
            )
    
    def detect_project_framework(self, project_dir: Path) -> FrameworkDetectionResult:
        """
        Detect primary framework for entire project.
        
        Args:
            project_dir: Project root directory
            
        Returns:
            FrameworkDetectionResult for primary framework
        """
        framework_counts: Dict[TestFramework, int] = {}
        
        # Search for test files
        test_patterns = ["test_*.py", "*_test.py", "*.test.js", "*.test.ts", "*Test.java"]
        
        for pattern in test_patterns:
            for file_path in project_dir.rglob(pattern):
                result = self.detect_from_file(file_path)
                if result.framework != TestFramework.UNKNOWN:
                    framework_counts[result.framework] = framework_counts.get(result.framework, 0) + 1
        
        if not framework_counts:
            return FrameworkDetectionResult(
                framework=TestFramework.UNKNOWN,
                confidence=0.0,
                language="Unknown",
                indicators=[]
            )
        
        # Get most common framework
        primary_framework = max(framework_counts, key=framework_counts.get)
        total_files = sum(framework_counts.values())
        confidence = framework_counts[primary_framework] / total_files
        
        return FrameworkDetectionResult(
            framework=primary_framework,
            confidence=confidence,
            language=self.INDICATORS[primary_framework]["language"],
            indicators=[f"Found in {framework_counts[primary_framework]}/{total_files} test files"]
        )


class FlakyTestPredictor:
    """
    Predictor for identifying potentially flaky tests.
    
    Uses heuristics to predict test flakiness based on code patterns.
    """
    
    # Flakiness risk patterns
    RISK_PATTERNS = {
        "timing_dependent": [
            r"time\.sleep\(",
            r"setTimeout\(",
            r"setInterval\(",
            r"Thread\.sleep\(",
            r"await.*delay"
        ],
        "random_values": [
            r"random\.",
            r"Math\.random\(",
            r"Random\(",
            r"UUID\.randomUUID",
            r"faker\."
        ],
        "external_dependencies": [
            r"requests\.",
            r"fetch\(",
            r"axios\.",
            r"http\.",
            r"urllib"
        ],
        "file_io": [
            r"open\(",
            r"File\(",
            r"readFile",
            r"writeFile",
            r"fs\."
        ],
        "datetime_dependent": [
            r"datetime\.now\(",
            r"Date\.now\(",
            r"new Date\(",
            r"System\.currentTimeMillis",
            r"time\.time\("
        ],
        "concurrent_execution": [
            r"threading\.",
            r"multiprocessing\.",
            r"async def",
            r"await ",
            r"Thread\(",
            r"ExecutorService"
        ]
    }
    
    # Recommendations for each risk factor
    RECOMMENDATIONS = {
        "timing_dependent": "Use fixed time values or mock time-dependent functions",
        "random_values": "Use seeded random generators or fixed test values",
        "external_dependencies": "Mock external API calls and network requests",
        "file_io": "Use in-memory file systems or mocked file operations",
        "datetime_dependent": "Mock datetime functions with fixed values",
        "concurrent_execution": "Use deterministic synchronization or avoid concurrency in tests"
    }
    
    def predict(self, test_code: str) -> FlakyPredictionResult:
        """
        Predict if a test is likely to be flaky.
        
        Args:
            test_code: Test code to analyze
            
        Returns:
            FlakyPredictionResult
            
        Example:
            >>> predictor = FlakyTestPredictor()
            >>> result = predictor.predict("def test(): time.sleep(random.random())")
            >>> print(f"Flaky: {result.is_likely_flaky}, Score: {result.flakiness_score}")
        """
        risk_factors = []
        recommendations = []
        flakiness_score = 0.0
        
        for risk_type, patterns in self.RISK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, test_code):
                    risk_factors.append(risk_type)
                    recommendations.append(self.RECOMMENDATIONS[risk_type])
                    flakiness_score += 0.15
                    break  # Only count each risk type once
        
        # Cap flakiness score at 0.95
        flakiness_score = min(flakiness_score, 0.95)
        
        # Additional checks
        # Check for no mocking (higher risk)
        has_mocking = bool(re.search(r"@mock\.|@patch\.|jest\.mock|Mockito\.", test_code))
        if not has_mocking and risk_factors:
            flakiness_score += 0.1
            recommendations.append("Add mocking for external dependencies")
        
        # Check for no assertions (suspicious)
        has_assertions = bool(re.search(r"assert |assertEquals|expect\(", test_code))
        if not has_assertions:
            flakiness_score += 0.05
        
        flakiness_score = min(flakiness_score, 0.95)
        
        is_likely_flaky = flakiness_score >= 0.3
        
        return FlakyPredictionResult(
            is_likely_flaky=is_likely_flaky,
            flakiness_score=flakiness_score,
            risk_factors=list(set(risk_factors)),
            recommendations=list(set(recommendations))
        )
    
    def batch_predict(
        self,
        test_codes: List[str]
    ) -> List[FlakyPredictionResult]:
        """
        Predict flakiness for multiple tests.
        
        Args:
            test_codes: List of test code strings
            
        Returns:
            List of prediction results
        """
        return [self.predict(code) for code in test_codes]


# Factory functions

def create_failure_classifier() -> FailureTriageClassifier:
    """Create a failure triage classifier instance."""
    return FailureTriageClassifier()


def create_framework_detector() -> FrameworkDetector:
    """Create a framework detector instance."""
    return FrameworkDetector()


def create_flaky_predictor() -> FlakyTestPredictor:
    """Create a flaky test predictor instance."""
    return FlakyTestPredictor()


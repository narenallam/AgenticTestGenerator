"""
Base classes and types for the evaluation framework.

This module provides foundational abstractions for building enterprise-grade
evaluations for the agentic test generation system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════


class EvalStatus(str, Enum):
    """Evaluation run status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EvalLevel(str, Enum):
    """Evaluation level in the hierarchy."""
    UNIT = "unit"              # Function-level
    COMPONENT = "component"    # Module-level
    AGENT = "agent"           # Agent-level
    SYSTEM = "system"         # End-to-end
    BUSINESS = "business"     # ROI/metrics


class QualityLevel(str, Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"   # 90-100%
    GOOD = "good"            # 80-89%
    FAIR = "fair"            # 70-79%
    POOR = "poor"            # <70%


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════


class EvalMetric(BaseModel):
    """A single evaluation metric result."""
    
    name: str = Field(..., description="Metric name (e.g., 'coverage', 'correctness')")
    value: float = Field(..., description="Metric value (0.0 to 1.0 for ratios)")
    unit: str = Field(default="ratio", description="Unit (ratio, count, seconds, etc.)")
    description: str = Field(default="", description="Human-readable description")
    threshold: Optional[float] = Field(default=None, description="Pass threshold")
    passed: Optional[bool] = Field(default=None, description="Whether threshold was met")
    
    def __post_init__(self):
        """Determine if threshold was passed."""
        if self.threshold is not None and self.passed is None:
            self.passed = self.value >= self.threshold


class EvalResult(BaseModel):
    """Result of an evaluation run."""
    
    eval_id: str = Field(..., description="Unique evaluation identifier")
    eval_name: str = Field(..., description="Name of the evaluation")
    eval_level: EvalLevel = Field(..., description="Evaluation level")
    status: EvalStatus = Field(default=EvalStatus.PENDING, description="Evaluation status")
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)
    duration_seconds: Optional[float] = Field(default=None)
    
    # Scores
    score: Optional[float] = Field(default=None, description="Overall score (0.0 to 1.0)")
    quality_level: Optional[QualityLevel] = Field(default=None)
    
    # Detailed metrics
    metrics: Dict[str, EvalMetric] = Field(default_factory=dict)
    
    # Context
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    errors: List[str] = Field(default_factory=list, description="Error messages if failed")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    recommendations: List[str] = Field(default_factory=list, description="Improvement suggestions")
    
    def add_metric(self, name: str, value: float, **kwargs) -> None:
        """Add a metric to the result."""
        metric = EvalMetric(name=name, value=value, **kwargs)
        self.metrics[name] = metric
    
    def calculate_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate weighted overall score from metrics.
        
        Args:
            weights: Metric name to weight mapping. If None, equal weights.
        
        Returns:
            Weighted score between 0.0 and 1.0
        """
        if not self.metrics:
            return 0.0
        
        if weights is None:
            # Equal weights
            weights = {name: 1.0 / len(self.metrics) for name in self.metrics}
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate weighted score
        score = sum(
            metric.value * normalized_weights.get(name, 0.0)
            for name, metric in self.metrics.items()
        )
        
        self.score = score
        self.quality_level = self._determine_quality_level(score)
        
        return score
    
    @staticmethod
    def _determine_quality_level(score: float) -> QualityLevel:
        """Determine quality level from score."""
        if score >= 0.90:
            return QualityLevel.EXCELLENT
        elif score >= 0.80:
            return QualityLevel.GOOD
        elif score >= 0.70:
            return QualityLevel.FAIR
        else:
            return QualityLevel.POOR
    
    def mark_completed(self) -> None:
        """Mark evaluation as completed."""
        self.status = EvalStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def mark_failed(self, error: str) -> None:
        """Mark evaluation as failed."""
        self.status = EvalStatus.FAILED
        self.errors.append(error)
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()


class EvalDataset(BaseModel):
    """An evaluation dataset entry."""
    
    id: str = Field(..., description="Unique dataset entry ID")
    name: str = Field(..., description="Human-readable name")
    
    # Source code
    source_code: str = Field(..., description="Source code to generate tests for")
    language: str = Field(default="python", description="Programming language")
    
    # Reference (gold standard)
    reference_tests: Optional[str] = Field(default=None, description="Human-written reference tests")
    
    # Metadata
    complexity: str = Field(default="medium", description="simple, medium, complex")
    category: str = Field(default="synthetic", description="synthetic, real_world, adversarial")
    tags: List[str] = Field(default_factory=list, description="Tags (e.g., ['async', 'database'])")
    
    # Expected outcomes
    expected_coverage: Optional[float] = Field(default=None, description="Expected coverage %")
    expected_test_count: Optional[int] = Field(default=None, description="Expected number of tests")
    known_bugs: List[str] = Field(default_factory=list, description="Known bugs to detect")
    
    # Additional context
    description: str = Field(default="", description="Description of what code does")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvalSuite(BaseModel):
    """A collection of related evaluations."""
    
    suite_id: str = Field(..., description="Unique suite identifier")
    name: str = Field(..., description="Suite name")
    description: str = Field(default="", description="Suite description")
    
    # Evaluations
    eval_results: List[EvalResult] = Field(default_factory=list)
    
    # Aggregate scores
    overall_score: Optional[float] = Field(default=None)
    quality_level: Optional[QualityLevel] = Field(default=None)
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)
    duration_seconds: Optional[float] = Field(default=None)
    
    # Summary
    total_evals: int = Field(default=0)
    passed_evals: int = Field(default=0)
    failed_evals: int = Field(default=0)
    
    def add_result(self, result: EvalResult) -> None:
        """Add an evaluation result to the suite."""
        self.eval_results.append(result)
        self.total_evals += 1
        
        if result.status == EvalStatus.COMPLETED:
            if result.score and result.score >= 0.70:
                self.passed_evals += 1
            else:
                self.failed_evals += 1
        elif result.status == EvalStatus.FAILED:
            self.failed_evals += 1
    
    def calculate_overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate overall suite score."""
        completed = [r for r in self.eval_results if r.status == EvalStatus.COMPLETED and r.score is not None]
        
        if not completed:
            self.overall_score = 0.0
            return 0.0
        
        if weights is None:
            # Equal weights
            score = sum(r.score for r in completed) / len(completed)
        else:
            # Weighted by eval name
            total_weight = 0.0
            weighted_sum = 0.0
            for result in completed:
                weight = weights.get(result.eval_name, 1.0)
                weighted_sum += result.score * weight
                total_weight += weight
            
            score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        self.overall_score = score
        self.quality_level = EvalResult._determine_quality_level(score)
        
        return score
    
    def mark_completed(self) -> None:
        """Mark suite as completed."""
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()


# ═══════════════════════════════════════════════════════════════════════════
# Abstract Base Classes
# ═══════════════════════════════════════════════════════════════════════════


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""
    
    def __init__(self, name: str, level: EvalLevel):
        """
        Initialize evaluator.
        
        Args:
            name: Evaluator name
            level: Evaluation level
        """
        self.name = name
        self.level = level
    
    @abstractmethod
    def evaluate(self, **kwargs) -> EvalResult:
        """
        Run evaluation.
        
        Returns:
            EvalResult with metrics and scores
        """
        pass
    
    def create_result(self, eval_id: str) -> EvalResult:
        """Create a new evaluation result."""
        return EvalResult(
            eval_id=eval_id,
            eval_name=self.name,
            eval_level=self.level,
            status=EvalStatus.RUNNING,
        )


class BaseMetricCalculator(ABC):
    """Abstract base class for metric calculators."""
    
    @abstractmethod
    def calculate(self, **kwargs) -> float:
        """
        Calculate metric value.
        
        Returns:
            Metric value (typically 0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get metric name."""
        pass
    
    @abstractmethod
    def get_unit(self) -> str:
        """Get metric unit."""
        pass


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    @abstractmethod
    def load(self, dataset_path: Path) -> List[EvalDataset]:
        """
        Load dataset from file.
        
        Args:
            dataset_path: Path to dataset file/directory
        
        Returns:
            List of evaluation dataset entries
        """
        pass
    
    @abstractmethod
    def save(self, dataset: List[EvalDataset], output_path: Path) -> None:
        """
        Save dataset to file.
        
        Args:
            dataset: List of evaluation dataset entries
            output_path: Where to save
        """
        pass


class BaseReporter(ABC):
    """Abstract base class for result reporters."""
    
    @abstractmethod
    def generate_report(self, suite: EvalSuite) -> str:
        """
        Generate evaluation report.
        
        Args:
            suite: Evaluation suite with results
        
        Returns:
            Formatted report string
        """
        pass
    
    @abstractmethod
    def save_report(self, suite: EvalSuite, output_path: Path) -> None:
        """
        Save report to file.
        
        Args:
            suite: Evaluation suite with results
            output_path: Where to save report
        """
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════


def generate_eval_id(prefix: str = "eval") -> str:
    """Generate a unique evaluation ID."""
    from datetime import datetime
    import uuid
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{short_uuid}"


def compare_scores(current: float, baseline: float, threshold: float = 0.05) -> Dict[str, Any]:
    """
    Compare current score against baseline.
    
    Args:
        current: Current score
        baseline: Baseline score
        threshold: Regression threshold (default: 5%)
    
    Returns:
        Dictionary with comparison results
    """
    delta = current - baseline
    delta_percent = (delta / baseline * 100) if baseline > 0 else 0.0
    
    is_regression = delta < -threshold
    is_improvement = delta > threshold
    
    return {
        "current": current,
        "baseline": baseline,
        "delta": delta,
        "delta_percent": delta_percent,
        "is_regression": is_regression,
        "is_improvement": is_improvement,
        "status": "regression" if is_regression else "improvement" if is_improvement else "stable"
    }


"""
Enterprise-grade evaluation system for agentic test generation.

This module provides comprehensive evaluation capabilities across:
- Test Quality (correctness, coverage, completeness)
- Agent Performance (Planner, Coder, Critic)
- Safety & Guardrails (PII, secrets, injection, determinism)
- Multi-Language Support (Python, Java, JavaScript, TypeScript)
- Goal Achievement (90% coverage, 90% pass rate)
"""

from .base import (
    BaseDatasetLoader,
    BaseEvaluator,
    BaseMetricCalculator,
    BaseReporter,
    EvalDataset,
    EvalLevel,
    EvalMetric,
    EvalResult,
    EvalStatus,
    EvalSuite,
    QualityLevel,
    compare_scores,
    generate_eval_id,
)
from .runner import EvalRunner

__all__ = [
    # Base classes
    "BaseEvaluator",
    "BaseMetricCalculator",
    "BaseDatasetLoader",
    "BaseReporter",
    
    # Data models
    "EvalResult",
    "EvalMetric",
    "EvalDataset",
    "EvalSuite",
    
    # Enums
    "EvalStatus",
    "EvalLevel",
    "QualityLevel",
    
    # Main runner
    "EvalRunner",
    
    # Utility functions
    "generate_eval_id",
    "compare_scores",
]

__version__ = "1.0.0"


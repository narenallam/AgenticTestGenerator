"""Reporting and result storage for evaluations."""

from .report_generator import (
    ConsoleReporter,
    JSONReporter,
    MarkdownReporter,
    MultiReporter,
)
from .result_storage import RegressionDetector, ResultStorage, TrendAnalyzer

__all__ = [
    # Storage
    "ResultStorage",
    "RegressionDetector",
    "TrendAnalyzer",
    
    # Reporters
    "ConsoleReporter",
    "MarkdownReporter",
    "JSONReporter",
    "MultiReporter",
]


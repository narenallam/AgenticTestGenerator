"""
Result storage and persistence for evaluations.

Stores evaluation results in SQLite for:
- Historical tracking
- Regression detection
- Trend analysis
- Reporting
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..base import EvalResult, EvalSuite, EvalStatus, QualityLevel


# ═══════════════════════════════════════════════════════════════════════════
# Result Storage
# ═══════════════════════════════════════════════════════════════════════════


class ResultStorage:
    """Store and retrieve evaluation results."""
    
    def __init__(self, db_path: Path):
        """
        Initialize result storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Create database schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Evaluation results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS eval_results (
                    id TEXT PRIMARY KEY,
                    eval_name TEXT NOT NULL,
                    eval_level TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    duration_seconds REAL,
                    score REAL,
                    quality_level TEXT,
                    metrics TEXT,
                    metadata TEXT,
                    errors TEXT,
                    warnings TEXT,
                    recommendations TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Evaluation suites table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS eval_suites (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    overall_score REAL,
                    quality_level TEXT,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    duration_seconds REAL,
                    total_evals INTEGER,
                    passed_evals INTEGER,
                    failed_evals INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Suite-to-result mapping
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS suite_results (
                    suite_id TEXT NOT NULL,
                    result_id TEXT NOT NULL,
                    PRIMARY KEY (suite_id, result_id),
                    FOREIGN KEY (suite_id) REFERENCES eval_suites(id),
                    FOREIGN KEY (result_id) REFERENCES eval_results(id)
                )
            """)
            
            # Baselines table for regression detection
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS baselines (
                    eval_name TEXT PRIMARY KEY,
                    baseline_score REAL NOT NULL,
                    baseline_metrics TEXT NOT NULL,
                    set_at TIMESTAMP NOT NULL,
                    result_id TEXT,
                    FOREIGN KEY (result_id) REFERENCES eval_results(id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_name ON eval_results(eval_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_started_at ON eval_results(started_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_score ON eval_results(score)")
            
            conn.commit()
    
    def save_result(self, result: EvalResult) -> None:
        """Save evaluation result to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO eval_results (
                    id, eval_name, eval_level, status,
                    started_at, completed_at, duration_seconds,
                    score, quality_level,
                    metrics, metadata, errors, warnings, recommendations
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.eval_id,
                result.eval_name,
                result.eval_level.value,
                result.status.value,
                result.started_at,
                result.completed_at,
                result.duration_seconds,
                result.score,
                result.quality_level.value if result.quality_level else None,
                json.dumps({name: metric.model_dump() for name, metric in result.metrics.items()}),
                json.dumps(result.metadata),
                json.dumps(result.errors),
                json.dumps(result.warnings),
                json.dumps(result.recommendations),
            ))
            
            conn.commit()
    
    def save_suite(self, suite: EvalSuite) -> None:
        """Save evaluation suite to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Save suite
            cursor.execute("""
                INSERT OR REPLACE INTO eval_suites (
                    id, name, description,
                    overall_score, quality_level,
                    started_at, completed_at, duration_seconds,
                    total_evals, passed_evals, failed_evals
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                suite.suite_id,
                suite.name,
                suite.description,
                suite.overall_score,
                suite.quality_level.value if suite.quality_level else None,
                suite.started_at,
                suite.completed_at,
                suite.duration_seconds,
                suite.total_evals,
                suite.passed_evals,
                suite.failed_evals,
            ))
            
            # Save individual results and mappings
            for result in suite.eval_results:
                self.save_result(result)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO suite_results (suite_id, result_id)
                    VALUES (?, ?)
                """, (suite.suite_id, result.eval_id))
            
            conn.commit()
    
    def get_result(self, eval_id: str) -> Optional[EvalResult]:
        """Retrieve evaluation result by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM eval_results WHERE id = ?", (eval_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_result(row, cursor)
    
    def get_latest_result(self, eval_name: str) -> Optional[EvalResult]:
        """Get the most recent result for an evaluation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM eval_results
                WHERE eval_name = ?
                ORDER BY started_at DESC
                LIMIT 1
            """, (eval_name,))
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_result(row, cursor)
    
    def get_results_by_eval_name(
        self,
        eval_name: str,
        limit: int = 100
    ) -> List[EvalResult]:
        """Get all results for a specific evaluation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM eval_results
                WHERE eval_name = ?
                ORDER BY started_at DESC
                LIMIT ?
            """, (eval_name, limit))
            
            rows = cursor.fetchall()
            
            return [self._row_to_result(row, cursor) for row in rows]
    
    def set_baseline(self, eval_name: str, result: EvalResult) -> None:
        """Set baseline for regression detection."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO baselines (
                    eval_name, baseline_score, baseline_metrics,
                    set_at, result_id
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                eval_name,
                result.score or 0.0,
                json.dumps({name: metric.value for name, metric in result.metrics.items()}),
                datetime.utcnow(),
                result.eval_id,
            ))
            
            conn.commit()
    
    def get_baseline(self, eval_name: str) -> Optional[Dict]:
        """Get baseline for an evaluation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM baselines WHERE eval_name = ?", (eval_name,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return {
                "eval_name": row[0],
                "baseline_score": row[1],
                "baseline_metrics": json.loads(row[2]),
                "set_at": row[3],
                "result_id": row[4],
            }
    
    def _row_to_result(self, row, cursor) -> EvalResult:
        """Convert database row to EvalResult."""
        from ..base import EvalMetric, EvalLevel, EvalStatus, QualityLevel
        
        # Parse metrics
        metrics_data = json.loads(row[9])
        metrics = {
            name: EvalMetric(**data)
            for name, data in metrics_data.items()
        }
        
        result = EvalResult(
            eval_id=row[0],
            eval_name=row[1],
            eval_level=EvalLevel(row[2]),
            status=EvalStatus(row[3]),
            started_at=datetime.fromisoformat(row[4]),
            completed_at=datetime.fromisoformat(row[5]) if row[5] else None,
            duration_seconds=row[6],
            score=row[7],
            quality_level=QualityLevel(row[8]) if row[8] else None,
            metrics=metrics,
            metadata=json.loads(row[10]),
            errors=json.loads(row[11]),
            warnings=json.loads(row[12]),
            recommendations=json.loads(row[13]),
        )
        
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Regression Detector
# ═══════════════════════════════════════════════════════════════════════════


class RegressionDetector:
    """Detect regressions in evaluation scores."""
    
    def __init__(self, storage: ResultStorage, threshold: float = 0.05):
        """
        Initialize regression detector.
        
        Args:
            storage: ResultStorage instance
            threshold: Regression threshold (default: 5%)
        """
        self.storage = storage
        self.threshold = threshold
    
    def check_regression(
        self,
        result: EvalResult,
        baseline: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Check if result shows regression from baseline.
        
        Args:
            result: Current evaluation result
            baseline: Baseline to compare against (if None, uses stored baseline)
        
        Returns:
            Dictionary with regression analysis
        """
        if baseline is None:
            baseline = self.storage.get_baseline(result.eval_name)
        
        if not baseline:
            return {
                "has_regression": False,
                "reason": "No baseline found",
                "regressions": [],
            }
        
        regressions = []
        
        # Check overall score
        current_score = result.score or 0.0
        baseline_score = baseline["baseline_score"]
        
        if current_score < baseline_score - self.threshold:
            regressions.append({
                "metric": "overall_score",
                "current": current_score,
                "baseline": baseline_score,
                "delta": current_score - baseline_score,
                "delta_percent": ((current_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0.0,
            })
        
        # Check individual metrics
        baseline_metrics = baseline["baseline_metrics"]
        for metric_name, metric in result.metrics.items():
            if metric_name in baseline_metrics:
                baseline_value = baseline_metrics[metric_name]
                current_value = metric.value
                
                if current_value < baseline_value - self.threshold:
                    regressions.append({
                        "metric": metric_name,
                        "current": current_value,
                        "baseline": baseline_value,
                        "delta": current_value - baseline_value,
                        "delta_percent": ((current_value - baseline_value) / baseline_value * 100) if baseline_value > 0 else 0.0,
                    })
        
        return {
            "has_regression": len(regressions) > 0,
            "reason": f"Found {len(regressions)} regression(s)" if regressions else "No regressions",
            "regressions": regressions,
            "threshold": self.threshold,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Trend Analyzer
# ═══════════════════════════════════════════════════════════════════════════


class TrendAnalyzer:
    """Analyze trends in evaluation results over time."""
    
    def __init__(self, storage: ResultStorage):
        """Initialize trend analyzer."""
        self.storage = storage
    
    def analyze_trend(self, eval_name: str, window: int = 10) -> Dict[str, any]:
        """
        Analyze trend for an evaluation.
        
        Args:
            eval_name: Evaluation name
            window: Number of recent results to analyze
        
        Returns:
            Dictionary with trend analysis
        """
        results = self.storage.get_results_by_eval_name(eval_name, limit=window)
        
        if len(results) < 2:
            return {
                "trend": "insufficient_data",
                "direction": "unknown",
                "recent_score": results[0].score if results else None,
            }
        
        # Extract scores
        scores = [r.score for r in results if r.score is not None]
        scores.reverse()  # Chronological order
        
        if not scores:
            return {
                "trend": "no_scores",
                "direction": "unknown",
            }
        
        # Simple linear regression
        n = len(scores)
        x = list(range(n))
        y = scores
        
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0.0
        
        # Determine trend
        if abs(slope) < 0.01:
            trend = "stable"
            direction = "flat"
        elif slope > 0:
            trend = "improving"
            direction = "up"
        else:
            trend = "declining"
            direction = "down"
        
        return {
            "trend": trend,
            "direction": direction,
            "slope": slope,
            "recent_score": scores[-1],
            "oldest_score": scores[0],
            "change": scores[-1] - scores[0],
            "num_data_points": n,
        }


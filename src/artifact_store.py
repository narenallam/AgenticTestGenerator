"""
Artifact store for test history and metrics tracking.

Provides:
- Test artifact storage (generated tests, execution logs)
- Version history tracking
- Metrics collection and aggregation
- Query and analysis capabilities
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

console = Console()


class TestArtifact(BaseModel):
    """A single test artifact."""
    
    id: Optional[int] = Field(default=None, description="Artifact ID")
    timestamp: datetime = Field(default_factory=datetime.now)
    test_code: str = Field(..., description="Generated test code")
    source_file: str = Field(..., description="Source file path")
    function_name: Optional[str] = Field(default=None)
    framework: str = Field(default="pytest")
    coverage: Optional[float] = Field(default=None)
    tests_passed: int = Field(default=0)
    tests_failed: int = Field(default=0)
    execution_time: float = Field(default=0.0)
    quality_score: Optional[float] = Field(default=None)
    llm_provider: Optional[str] = Field(default=None)
    generation_iterations: int = Field(default=1)
    metadata: Dict = Field(default_factory=dict)


class MetricsSummary(BaseModel):
    """Aggregated metrics summary."""
    
    total_tests_generated: int = Field(default=0)
    total_functions_tested: int = Field(default=0)
    average_coverage: float = Field(default=0.0)
    average_quality_score: float = Field(default=0.0)
    success_rate: float = Field(default=0.0)
    total_execution_time: float = Field(default=0.0)
    most_used_framework: str = Field(default="unknown")
    generation_trends: List[Dict] = Field(default_factory=list)


class ArtifactStore:
    """
    Persistent storage for test artifacts and metrics.
    
    Uses SQLite for efficient storage and querying.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize artifact store.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path or Path.home() / ".cache" / "genai_test_agent" / "artifacts.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        self._initialize_db()
        
        console.print(f"[green]✓[/green] Artifact store initialized: {self.db_path}")
    
    def _initialize_db(self) -> None:
        """Create database schema."""
        cursor = self.conn.cursor()
        
        # Artifacts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                test_code TEXT NOT NULL,
                source_file TEXT NOT NULL,
                function_name TEXT,
                framework TEXT NOT NULL,
                coverage REAL,
                tests_passed INTEGER,
                tests_failed INTEGER,
                execution_time REAL,
                quality_score REAL,
                llm_provider TEXT,
                generation_iterations INTEGER,
                metadata TEXT
            )
        """)
        
        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                artifact_id INTEGER,
                metadata TEXT,
                FOREIGN KEY (artifact_id) REFERENCES artifacts(id)
            )
        """)
        
        # Create indices
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_artifacts_source_file
            ON artifacts(source_file)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_artifacts_timestamp
            ON artifacts(timestamp)
        """)
        
        self.conn.commit()
    
    def store_artifact(self, artifact: TestArtifact) -> int:
        """
        Store a test artifact.
        
        Args:
            artifact: TestArtifact to store
            
        Returns:
            Artifact ID
            
        Example:
            >>> store = ArtifactStore()
            >>> artifact = TestArtifact(test_code="...", source_file="foo.py")
            >>> artifact_id = store.store_artifact(artifact)
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO artifacts (
                timestamp, test_code, source_file, function_name,
                framework, coverage, tests_passed, tests_failed,
                execution_time, quality_score, llm_provider,
                generation_iterations, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            artifact.timestamp.isoformat(),
            artifact.test_code,
            artifact.source_file,
            artifact.function_name,
            artifact.framework,
            artifact.coverage,
            artifact.tests_passed,
            artifact.tests_failed,
            artifact.execution_time,
            artifact.quality_score,
            artifact.llm_provider,
            artifact.generation_iterations,
            json.dumps(artifact.metadata)
        ))
        
        self.conn.commit()
        artifact_id = cursor.lastrowid
        
        console.print(f"  → Artifact stored: ID={artifact_id}")
        
        return artifact_id
    
    def get_artifact(self, artifact_id: int) -> Optional[TestArtifact]:
        """Retrieve an artifact by ID."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM artifacts WHERE id = ?
        """, (artifact_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return self._row_to_artifact(row)
    
    def query_artifacts(
        self,
        source_file: Optional[str] = None,
        function_name: Optional[str] = None,
        min_coverage: Optional[float] = None,
        limit: int = 100
    ) -> List[TestArtifact]:
        """
        Query artifacts with filters.
        
        Args:
            source_file: Filter by source file
            function_name: Filter by function name
            min_coverage: Minimum coverage threshold
            limit: Maximum results
            
        Returns:
            List of matching artifacts
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM artifacts WHERE 1=1"
        params = []
        
        if source_file:
            query += " AND source_file = ?"
            params.append(source_file)
        
        if function_name:
            query += " AND function_name = ?"
            params.append(function_name)
        
        if min_coverage:
            query += " AND coverage >= ?"
            params.append(min_coverage)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        return [self._row_to_artifact(row) for row in cursor.fetchall()]
    
    def get_metrics_summary(
        self,
        since: Optional[datetime] = None
    ) -> MetricsSummary:
        """
        Get aggregated metrics summary.
        
        Args:
            since: Optional start date for aggregation
            
        Returns:
            MetricsSummary with aggregated data
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM artifacts"
        params = []
        
        if since:
            query += " WHERE timestamp >= ?"
            params.append(since.isoformat())
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        if not rows:
            return MetricsSummary()
        
        # Aggregate metrics
        total_tests = len(rows)
        unique_functions = len(set(row["function_name"] for row in rows if row["function_name"]))
        
        coverages = [row["coverage"] for row in rows if row["coverage"] is not None]
        avg_coverage = sum(coverages) / len(coverages) if coverages else 0.0
        
        quality_scores = [row["quality_score"] for row in rows if row["quality_score"] is not None]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        passed = sum(row["tests_passed"] for row in rows)
        failed = sum(row["tests_failed"] for row in rows)
        success_rate = passed / (passed + failed) if (passed + failed) > 0 else 0.0
        
        total_time = sum(row["execution_time"] for row in rows)
        
        # Most used framework
        frameworks = [row["framework"] for row in rows]
        most_used = max(set(frameworks), key=frameworks.count) if frameworks else "unknown"
        
        return MetricsSummary(
            total_tests_generated=total_tests,
            total_functions_tested=unique_functions,
            average_coverage=avg_coverage,
            average_quality_score=avg_quality,
            success_rate=success_rate,
            total_execution_time=total_time,
            most_used_framework=most_used,
            generation_trends=[]
        )
    
    def get_coverage_trend(self, days: int = 30) -> List[Dict]:
        """Get coverage trend over time."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                DATE(timestamp) as date,
                AVG(coverage) as avg_coverage,
                COUNT(*) as count
            FROM artifacts
            WHERE coverage IS NOT NULL
                AND timestamp >= datetime('now', ? || ' days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, (f"-{days}",))
        
        return [
            {
                "date": row["date"],
                "avg_coverage": row["avg_coverage"],
                "count": row["count"]
            }
            for row in cursor.fetchall()
        ]
    
    def display_summary(self) -> None:
        """Display metrics summary in a table."""
        summary = self.get_metrics_summary()
        
        table = Table(title="📊 Artifact Store Metrics")
        
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Tests Generated", str(summary.total_tests_generated))
        table.add_row("Functions Tested", str(summary.total_functions_tested))
        table.add_row("Average Coverage", f"{summary.average_coverage:.1f}%")
        table.add_row("Average Quality Score", f"{summary.average_quality_score:.1f}/100")
        table.add_row("Success Rate", f"{summary.success_rate:.1%}")
        table.add_row("Total Execution Time", f"{summary.total_execution_time:.1f}s")
        table.add_row("Most Used Framework", summary.most_used_framework)
        
        console.print(table)
    
    def _row_to_artifact(self, row) -> TestArtifact:
        """Convert database row to TestArtifact."""
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        
        return TestArtifact(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            test_code=row["test_code"],
            source_file=row["source_file"],
            function_name=row["function_name"],
            framework=row["framework"],
            coverage=row["coverage"],
            tests_passed=row["tests_passed"],
            tests_failed=row["tests_failed"],
            execution_time=row["execution_time"],
            quality_score=row["quality_score"],
            llm_provider=row["llm_provider"],
            generation_iterations=row["generation_iterations"],
            metadata=metadata
        )
    
    def export_artifacts(
        self,
        output_path: Path,
        source_file: Optional[str] = None
    ) -> None:
        """
        Export artifacts to JSON file.
        
        Args:
            output_path: Output file path
            source_file: Optional filter by source file
        """
        artifacts = self.query_artifacts(source_file=source_file, limit=1000)
        
        data = [
            {
                "id": a.id,
                "timestamp": a.timestamp.isoformat(),
                "source_file": a.source_file,
                "function_name": a.function_name,
                "coverage": a.coverage,
                "tests_passed": a.tests_passed,
                "tests_failed": a.tests_failed,
                "quality_score": a.quality_score
            }
            for a in artifacts
        ]
        
        output_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        console.print(f"[green]✓[/green] Exported {len(data)} artifacts to {output_path}")
    
    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_artifact_store(db_path: Optional[Path] = None) -> ArtifactStore:
    """Factory function to create artifact store."""
    return ArtifactStore(db_path=db_path)


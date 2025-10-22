"""
Audit Logger for comprehensive event tracking.

Provides structured logging with SQLite persistence for compliance.
"""

import json
import sqlite3
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from rich.console import Console

console = Console()


class EventType(str, Enum):
    """Types of audit events."""
    TOOL_CALL = "tool_call"
    POLICY_DECISION = "policy_decision"
    SAFETY_VIOLATION = "safety_violation"
    HITL_APPROVAL = "hitl_approval"
    BUDGET_LIMIT = "budget_limit"
    ERROR = "error"
    SESSION_START = "session_start"
    SESSION_END = "session_end"


class Severity(str, Enum):
    """Event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEvent(BaseModel):
    """A single audit event."""
    
    id: Optional[int] = Field(default=None, description="Event ID (auto-assigned)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")
    event_type: EventType = Field(..., description="Event type")
    severity: Severity = Field(default=Severity.INFO, description="Severity level")
    
    # Context
    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(default="system", description="User identifier")
    
    # Event details
    actor: str = Field(..., description="Actor (agent, tool, user)")
    action: str = Field(..., description="Action performed")
    result: str = Field(..., description="Result (ALLOW/DENY/SUCCESS/FAIL)")
    
    # Additional data
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    reason: Optional[str] = Field(default=None, description="Reason for action/decision")
    duration_ms: Optional[float] = Field(default=None, description="Duration in milliseconds")


class AuditLogger:
    """
    Comprehensive audit logging with SQLite persistence.
    
    Features:
    - Structured event logging
    - SQLite persistence with indexing
    - Query capabilities
    - JSON export
    - Privacy controls
    - Automatic rotation
    
    Example:
        >>> logger = AuditLogger()
        >>> logger.log_tool_call(
        ...     tool="generate_tests",
        ...     params={"max_iterations": 5},
        ...     result="SUCCESS",
        ...     duration_ms=1234.5
        ... )
        >>> events = logger.query(event_type="tool_call", limit=10)
    """
    
    def __init__(self, db_path: str = "./data/audit_logs.db"):
        """
        Initialize audit logger.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        console.print("[bold green]📋 Audit Logger Initialized[/bold green]")
    
    def _init_database(self):
        """Initialize SQLite database and schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                result TEXT NOT NULL,
                metadata TEXT,
                reason TEXT,
                duration_ms REAL
            )
        """)
        
        # Create indices for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON audit_events(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_type 
            ON audit_events(event_type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session 
            ON audit_events(session_id)
        """)
        
        conn.commit()
        conn.close()
    
    def log_event(self, event: AuditEvent):
        """
        Log an audit event.
        
        Args:
            event: Event to log
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO audit_events (
                timestamp, event_type, severity, session_id, user_id,
                actor, action, result, metadata, reason, duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.timestamp.isoformat(),
            event.event_type.value,
            event.severity.value,
            event.session_id,
            event.user_id,
            event.actor,
            event.action,
            event.result,
            json.dumps(event.metadata),
            event.reason,
            event.duration_ms
        ))
        
        conn.commit()
        event_id = cursor.lastrowid
        conn.close()
        
        # Console output for important events
        if event.severity in (Severity.WARNING, Severity.ERROR, Severity.CRITICAL):
            console.print(
                f"[yellow]⚠️  {event.event_type.value}: {event.action} -> {event.result}[/yellow]"
            )
        
        return event_id
    
    def log_tool_call(
        self,
        session_id: str,
        tool: str,
        params: Dict[str, Any],
        result: str,
        duration_ms: float,
        error: Optional[str] = None
    ):
        """Log a tool call."""
        event = AuditEvent(
            event_type=EventType.TOOL_CALL,
            severity=Severity.INFO if result == "SUCCESS" else Severity.WARNING,
            session_id=session_id,
            actor="agent",
            action=f"call_{tool}",
            result=result,
            metadata={
                "tool": tool,
                "params": params,
                "error": error
            },
            duration_ms=duration_ms
        )
        self.log_event(event)
    
    def log_policy_decision(
        self,
        session_id: str,
        tool: str,
        decision: str,
        reason: str,
        risk_tier: str,
        rule_id: Optional[str] = None
    ):
        """Log a policy decision."""
        severity = {
            "ALLOW": Severity.INFO,
            "DENY": Severity.WARNING,
            "REVIEW": Severity.WARNING
        }.get(decision, Severity.INFO)
        
        event = AuditEvent(
            event_type=EventType.POLICY_DECISION,
            severity=severity,
            session_id=session_id,
            actor="policy_engine",
            action=f"evaluate_{tool}",
            result=decision,
            reason=reason,
            metadata={
                "tool": tool,
                "risk_tier": risk_tier,
                "rule_id": rule_id
            }
        )
        self.log_event(event)
    
    def log_safety_violation(
        self,
        session_id: str,
        violation_type: str,
        severity: Severity,
        action_taken: str,
        details: Dict[str, Any]
    ):
        """Log a safety violation."""
        event = AuditEvent(
            event_type=EventType.SAFETY_VIOLATION,
            severity=severity,
            session_id=session_id,
            actor="guardrails",
            action=violation_type,
            result=action_taken,
            metadata=details
        )
        self.log_event(event)
    
    def log_hitl_approval(
        self,
        session_id: str,
        action: str,
        decision: str,
        reason: Optional[str] = None,
        response_time_ms: Optional[float] = None
    ):
        """Log a human-in-the-loop approval."""
        event = AuditEvent(
            event_type=EventType.HITL_APPROVAL,
            severity=Severity.INFO,
            session_id=session_id,
            actor="user",
            action=action,
            result=decision,
            reason=reason,
            duration_ms=response_time_ms
        )
        self.log_event(event)
    
    def log_budget_limit(
        self,
        session_id: str,
        limit_type: str,
        current: float,
        maximum: float,
        action: str
    ):
        """Log a budget limit event."""
        event = AuditEvent(
            event_type=EventType.BUDGET_LIMIT,
            severity=Severity.WARNING,
            session_id=session_id,
            actor="budget_tracker",
            action=limit_type,
            result=action,
            metadata={
                "current": current,
                "maximum": maximum,
                "percentage": (current / maximum * 100) if maximum > 0 else 0
            }
        )
        self.log_event(event)
    
    def query(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """
        Query audit events.
        
        Args:
            session_id: Filter by session
            event_type: Filter by event type
            severity: Filter by severity
            start_time: Start timestamp
            end_time: End timestamp
            limit: Max results
            
        Returns:
            List of matching events
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        events = []
        for row in rows:
            events.append(AuditEvent(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                event_type=EventType(row[2]),
                severity=Severity(row[3]),
                session_id=row[4],
                user_id=row[5],
                actor=row[6],
                action=row[7],
                result=row[8],
                metadata=json.loads(row[9]) if row[9] else {},
                reason=row[10],
                duration_ms=row[11]
            ))
        
        return events
    
    def export_session(self, session_id: str, format: str = "json") -> str:
        """
        Export audit trail for a session.
        
        Args:
            session_id: Session to export
            format: Export format (json only for now)
            
        Returns:
            Formatted audit trail
        """
        events = self.query(session_id=session_id, limit=10000)
        
        if format == "json":
            return json.dumps([
                {
                    "timestamp": e.timestamp.isoformat(),
                    "event_type": e.event_type.value,
                    "severity": e.severity.value,
                    "actor": e.actor,
                    "action": e.action,
                    "result": e.result,
                    "reason": e.reason,
                    "metadata": e.metadata,
                    "duration_ms": e.duration_ms
                }
                for e in events
            ], indent=2)
        
        return ""
    
    def get_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        query = """
            SELECT 
                event_type,
                severity,
                result,
                COUNT(*) as count,
                AVG(duration_ms) as avg_duration
            FROM audit_events
        """
        
        if session_id:
            query += " WHERE session_id = ?"
            cursor.execute(query + " GROUP BY event_type, severity, result", (session_id,))
        else:
            cursor.execute(query + " GROUP BY event_type, severity, result")
        
        rows = cursor.fetchall()
        conn.close()
        
        summary = {
            "total_events": sum(row[3] for row in rows),
            "by_type": {},
            "by_severity": {},
            "by_result": {}
        }
        
        for row in rows:
            event_type, severity, result, count, avg_duration = row
            
            if event_type not in summary["by_type"]:
                summary["by_type"][event_type] = 0
            summary["by_type"][event_type] += count
            
            if severity not in summary["by_severity"]:
                summary["by_severity"][severity] = 0
            summary["by_severity"][severity] += count
            
            if result not in summary["by_result"]:
                summary["by_result"][result] = 0
            summary["by_result"][result] += count
        
        return summary


def create_audit_logger(db_path: str = "./data/audit_logs.db") -> AuditLogger:
    """Factory function to create audit logger."""
    return AuditLogger(db_path)


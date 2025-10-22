"""
Distributed tracing with correlation IDs.

Implements span tracking with parent-child relationships
for end-to-end request flow visibility.
"""

import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from tinydb import TinyDB

from .config import get_config
from .logger import get_logger, set_trace_id

logger = get_logger()


# Context variables for trace propagation
current_trace_var: ContextVar[Optional['Trace']] = ContextVar("current_trace", default=None)
current_span_var: ContextVar[Optional['Span']] = ContextVar("current_span", default=None)


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════


class SpanStatus(str, Enum):
    """Span status."""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class Span:
    """Represents a single unit of work."""
    
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    attributes: Dict[str, any] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)
    error: Optional[str] = None
    
    def finish(self, status: SpanStatus = SpanStatus.OK, error: Optional[str] = None) -> None:
        """Finish the span."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        self.error = error
    
    def set_attribute(self, key: str, value: any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        """Add an event to the span."""
        self.events.append({
            'name': name,
            'timestamp': time.time(),
            'attributes': attributes or {}
        })
    
    def to_dict(self) -> Dict:
        """Convert span to dictionary."""
        return {
            'span_id': self.span_id,
            'trace_id': self.trace_id,
            'parent_span_id': self.parent_span_id,
            'operation': self.operation,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            'duration_ms': self.duration_ms,
            'status': self.status.value,
            'attributes': self.attributes,
            'events': self.events,
            'error': self.error,
        }


@dataclass
class Trace:
    """Represents a complete request trace."""
    
    trace_id: str
    root_span_id: str
    start_time: float
    end_time: Optional[float] = None
    spans: List[Span] = field(default_factory=list)
    
    def add_span(self, span: Span) -> None:
        """Add a span to the trace."""
        self.spans.append(span)
    
    def finish(self) -> None:
        """Finish the trace."""
        self.end_time = time.time()
    
    def get_duration_ms(self) -> Optional[float]:
        """Get total trace duration."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None
    
    def to_dict(self) -> Dict:
        """Convert trace to dictionary."""
        return {
            'trace_id': self.trace_id,
            'root_span_id': self.root_span_id,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            'duration_ms': self.get_duration_ms(),
            'span_count': len(self.spans),
            'spans': [span.to_dict() for span in self.spans],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Tracer
# ═══════════════════════════════════════════════════════════════════════════


class Tracer:
    """Distributed tracer."""
    
    def __init__(self):
        """Initialize tracer."""
        self.config = get_config()
        self.db = TinyDB(self.config.get_traces_db_path())
        self.table = self.db.table('traces')
    
    def start_trace(self, operation: str) -> Trace:
        """Start a new trace."""
        trace_id = f"trace_{uuid.uuid4().hex[:16]}"
        root_span_id = f"span_{uuid.uuid4().hex[:16]}"
        
        trace = Trace(
            trace_id=trace_id,
            root_span_id=root_span_id,
            start_time=time.time(),
        )
        
        # Set in context
        current_trace_var.set(trace)
        set_trace_id(trace_id)
        
        # Start root span
        root_span = self.start_span(operation, parent_span_id=None)
        
        logger.bind(trace_id=trace_id).debug(f"Trace started: {operation}")
        
        return trace
    
    def start_span(self, operation: str, parent_span_id: Optional[str] = None) -> Span:
        """Start a new span."""
        trace = current_trace_var.get()
        if not trace:
            # No active trace, start one
            trace = self.start_trace(operation)
        
        # Get parent span if not specified
        if parent_span_id is None:
            parent_span = current_span_var.get()
            parent_span_id = parent_span.span_id if parent_span else None
        
        span = Span(
            span_id=f"span_{uuid.uuid4().hex[:16]}",
            trace_id=trace.trace_id,
            parent_span_id=parent_span_id,
            operation=operation,
            start_time=time.time(),
        )
        
        # Add to trace
        trace.add_span(span)
        
        # Set as current span
        current_span_var.set(span)
        
        logger.bind(
            trace_id=trace.trace_id,
            span_id=span.span_id,
            operation=operation
        ).debug(f"Span started: {operation}")
        
        return span
    
    def finish_span(
        self,
        span: Span,
        status: SpanStatus = SpanStatus.OK,
        error: Optional[str] = None
    ) -> None:
        """Finish a span."""
        span.finish(status=status, error=error)
        
        logger.bind(
            trace_id=span.trace_id,
            span_id=span.span_id,
            duration_ms=span.duration_ms
        ).debug(f"Span finished: {span.operation} ({span.duration_ms:.2f}ms)")
    
    def finish_trace(self, trace: Trace) -> None:
        """Finish and store a trace."""
        trace.finish()
        
        # Save to database
        if self.config.tracing_enabled:
            self.table.insert(trace.to_dict())
        
        logger.bind(trace_id=trace.trace_id).info(
            f"Trace completed: {trace.get_duration_ms():.2f}ms, {len(trace.spans)} spans"
        )
        
        # Clear from context
        current_trace_var.set(None)
        current_span_var.set(None)


# ═══════════════════════════════════════════════════════════════════════════
# Context Managers
# ═══════════════════════════════════════════════════════════════════════════


class trace_context:
    """Context manager for tracing."""
    
    def __init__(self, operation: str, **attributes):
        """Initialize trace context."""
        self.operation = operation
        self.attributes = attributes
        self.tracer = get_tracer()
        self.trace = None
    
    def __enter__(self):
        """Enter trace context."""
        self.trace = self.tracer.start_trace(self.operation)
        
        # Set attributes on root span
        root_span = current_span_var.get()
        if root_span:
            for key, value in self.attributes.items():
                root_span.set_attribute(key, value)
        
        return self.trace
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit trace context."""
        if exc_type is not None:
            # Error occurred
            root_span = current_span_var.get()
            if root_span:
                self.tracer.finish_span(
                    root_span,
                    status=SpanStatus.ERROR,
                    error=str(exc_val)
                )
        else:
            # Success
            root_span = current_span_var.get()
            if root_span:
                self.tracer.finish_span(root_span)
        
        self.tracer.finish_trace(self.trace)


class span_context:
    """Context manager for spans."""
    
    def __init__(self, operation: str, **attributes):
        """Initialize span context."""
        self.operation = operation
        self.attributes = attributes
        self.tracer = get_tracer()
        self.span = None
    
    def __enter__(self):
        """Enter span context."""
        self.span = self.tracer.start_span(self.operation)
        
        # Set attributes
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)
        
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit span context."""
        if exc_type is not None:
            # Error occurred
            self.tracer.finish_span(
                self.span,
                status=SpanStatus.ERROR,
                error=str(exc_val)
            )
        else:
            # Success
            self.tracer.finish_span(self.span)


# ═══════════════════════════════════════════════════════════════════════════
# Global Tracer
# ═══════════════════════════════════════════════════════════════════════════


_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Get global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def get_current_trace() -> Optional[Trace]:
    """Get current trace from context."""
    return current_trace_var.get()


def get_current_span() -> Optional[Span]:
    """Get current span from context."""
    return current_span_var.get()


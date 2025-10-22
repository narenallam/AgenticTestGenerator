"""
Enterprise Observability System.

Comprehensive logging, metrics, tracing, and alerting for the
agentic test generation system.

**Components**:
- Logger: Structured logging with Loguru (console, file, database)
- Metrics: Prometheus-compatible metrics (Counter, Gauge, Histogram)
- Tracer: Distributed tracing with correlation IDs
- Instrumentation: Decorators for automatic observability

**Quick Start**:

```python
from src.observability import observe, get_logger, record_test_generation

logger = get_logger()

@observe(operation="test_generation", trace=True)
def generate_tests(source_code: str) -> str:
    logger.info("Generating tests", language="python")
    # ... generation logic ...
    return tests

# Record metrics
record_test_generation(
    language="python",
    framework="pytest",
    coverage=0.92,
    pass_rate=0.95,
    test_count=15,
    duration_seconds=3.5
)
```

**Migration to Prometheus/Grafana**:
- Metrics are already Prometheus-compatible
- Export endpoint available via `get_registry().export_prometheus()`
- Ready for Prometheus scraping
"""

# Configuration
from .config import get_config, set_config, update_config, ObservabilityConfig

# Logger
from .logger import (
    get_logger,
    bind_context,
    set_trace_id,
    get_trace_id,
    set_session_id,
    get_session_id,
)

# Metrics
from .metrics import (
    counter,
    gauge,
    histogram,
    get_registry,
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
)

# Tracer
from .tracer import (
    get_tracer,
    get_current_trace,
    get_current_span,
    trace_context,
    span_context,
    Trace,
    Span,
    SpanStatus,
)

# Instrumentation
from .instrumentation import (
    observe,
    observe_llm_call,
    observe_agent,
    record_test_generation,
    record_guardrail_event,
)

__all__ = [
    # Configuration
    "get_config",
    "set_config",
    "update_config",
    "ObservabilityConfig",
    
    # Logger
    "get_logger",
    "bind_context",
    "set_trace_id",
    "get_trace_id",
    "set_session_id",
    "get_session_id",
    
    # Metrics
    "counter",
    "gauge",
    "histogram",
    "get_registry",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    
    # Tracer
    "get_tracer",
    "get_current_trace",
    "get_current_span",
    "trace_context",
    "span_context",
    "Trace",
    "Span",
    "SpanStatus",
    
    # Instrumentation
    "observe",
    "observe_llm_call",
    "observe_agent",
    "record_test_generation",
    "record_guardrail_event",
]

__version__ = "1.0.0"


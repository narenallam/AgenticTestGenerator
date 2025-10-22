# 🔭 Enterprise Observability System - Design Document

**Purpose**: Comprehensive observability for agentic test generation  
**Current**: Local implementation with Loguru + TinyDB  
**Future**: Prometheus + Grafana + Cloud (OpenTelemetry-ready)  
**Date**: October 23, 2024

---

## 🎯 Goals

1. ✅ **Structured Logging**: Rich, queryable logs with context
2. ✅ **Metrics Collection**: Time-series data for performance tracking
3. ✅ **Tracing**: Request flow across agents and tools
4. ✅ **Alerting**: Anomaly detection and threshold-based alerts
5. ✅ **Dashboards**: Real-time visualization of system health
6. ✅ **Migration-Ready**: OpenTelemetry-compatible, easy Prometheus migration

---

## 📊 Observability Pillars

### 1. Logs (Loguru)

**What**: Structured, contextual event records

**Levels**:
- `TRACE`: Detailed debugging (LLM prompts/responses)
- `DEBUG`: Development debugging
- `INFO`: General operational events
- `SUCCESS`: Successful operations
- `WARNING`: Non-critical issues
- `ERROR`: Errors that don't stop execution
- `CRITICAL`: System-level failures

**Structure**:
```json
{
  "timestamp": "2024-10-23T10:30:45.123Z",
  "level": "INFO",
  "message": "Test generation started",
  "context": {
    "session_id": "sess_abc123",
    "agent": "coder",
    "function": "generate_tests",
    "source_file": "module.py",
    "language": "python"
  },
  "metadata": {
    "host": "localhost",
    "pid": 12345,
    "thread": "MainThread"
  }
}
```

**Storage**:
- Console (colored, formatted)
- File (daily rotation, 7-day retention)
- TinyDB (queryable, structured)

### 2. Metrics (Prometheus-Compatible)

**What**: Time-series numerical data

**Categories**:

**A. Request Metrics**
- `test_generation_requests_total`: Counter of total requests
- `test_generation_duration_seconds`: Histogram of request latency
- `test_generation_errors_total`: Counter of errors by type

**B. LLM Metrics**
- `llm_calls_total`: Counter by provider/model
- `llm_tokens_total`: Counter by type (input/output)
- `llm_latency_seconds`: Histogram by provider
- `llm_cost_dollars`: Counter by provider
- `llm_errors_total`: Counter by provider/error_type

**C. Test Generation Metrics**
- `tests_generated_total`: Counter by language/framework
- `test_coverage_ratio`: Gauge (0.0-1.0)
- `test_pass_rate_ratio`: Gauge (0.0-1.0)
- `test_execution_duration_seconds`: Histogram

**D. Agent Metrics**
- `agent_iterations_total`: Counter by agent
- `agent_tool_calls_total`: Counter by agent/tool
- `agent_decision_time_seconds`: Histogram by agent

**E. Guardrails Metrics**
- `guardrails_checks_total`: Counter by type
- `guardrails_violations_total`: Counter by type
- `guardrails_blocks_total`: Counter by type
- `pii_detections_total`: Counter
- `secret_detections_total`: Counter

**F. System Metrics**
- `system_cpu_percent`: Gauge
- `system_memory_mb`: Gauge
- `system_disk_usage_percent`: Gauge
- `active_sessions`: Gauge

**Storage**:
- TinyDB (time-series records)
- Prometheus exposition format (for scraping)

### 3. Traces (Correlation IDs)

**What**: Request flow tracking across components

**Structure**:
```python
{
  "trace_id": "trace_xyz789",
  "span_id": "span_abc123",
  "parent_span_id": "span_parent",
  "operation": "generate_tests",
  "start_time": "2024-10-23T10:30:45.000Z",
  "duration_ms": 1250,
  "status": "success",
  "attributes": {
    "agent": "coder",
    "language": "python",
    "coverage": 0.89
  }
}
```

**Implementation**:
- Context variables for trace propagation
- Span tracking for each operation
- Parent-child relationships

### 4. Alerts

**What**: Automated notifications for anomalies

**Rules**:
- `high_error_rate`: Error rate > 5% in 5 minutes
- `low_coverage`: Coverage < 80% for 3 consecutive runs
- `llm_quota_exceeded`: LLM costs > daily budget
- `guardrails_violations`: Multiple violations in 1 minute
- `slow_response`: p99 latency > 60 seconds
- `system_overload`: CPU > 90% for 5 minutes

**Channels**:
- Console (immediate)
- Log file (persistent)
- Future: Email, Slack, PagerDuty

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                           │
│  (Agents, Tools, Guardrails, Evals)                            │
└────────────────────┬────────────────────────────────────────────┘
                     │ Instrumented with decorators
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Observability Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Logger     │  │   Metrics    │  │   Tracer     │         │
│  │  (Loguru)    │  │  Collector   │  │  (Context)   │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          ↓                  ↓                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Storage Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Console    │  │   TinyDB     │  │   Files      │         │
│  │   (stdout)   │  │   (metrics)  │  │   (logs)     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Visualization Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Console    │  │   Text       │  │   Prometheus │         │
│  │   Monitor    │  │   Dashboard  │  │   Exporter   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Implementation Components

### 1. Core Infrastructure

**`src/observability/logger.py`**
- Loguru setup with multiple sinks
- Custom formatters
- Context binding
- Intercept standard logging

**`src/observability/metrics.py`**
- Metric types: Counter, Gauge, Histogram, Summary
- Prometheus-compatible names
- Label management
- Time-series storage in TinyDB

**`src/observability/tracer.py`**
- Context manager for spans
- Trace ID generation
- Parent-child tracking
- Trace storage

**`src/observability/config.py`**
- Centralized configuration
- Feature flags (enable/disable components)
- Retention policies
- Alert thresholds

### 2. Instrumentation

**Decorators**:
```python
@observe(
    metric="test_generation_duration_seconds",
    log_level="INFO",
    trace=True
)
def generate_tests(source_code: str) -> str:
    # Automatically tracked
    pass
```

**Context Managers**:
```python
with observe_span("llm_call", model="gpt-4"):
    response = llm.generate(prompt)
```

**Manual Logging**:
```python
logger.bind(
    session_id=session.id,
    agent="planner"
).info("Task decomposition complete", tasks=5)
```

### 3. Collectors

**`MetricsCollector`**:
- In-memory counters/gauges
- Periodic flush to TinyDB
- Prometheus exposition endpoint

**`LogCollector`**:
- Buffers logs for batch writes
- Filters by level
- Structured indexing

**`TraceCollector`**:
- Span lifecycle management
- Trace assembly
- Export to storage

### 4. Dashboards

**Console Monitor** (Real-time):
```
┌─────────────────────────────────────────────────────────────┐
│ Agentic Test Generator - Live Monitor                      │
├─────────────────────────────────────────────────────────────┤
│ Requests:  125 (12/min)    Errors: 3 (2.4%)               │
│ Avg Latency: 3.2s          p99: 12.5s                     │
│ Coverage: 88.5%            Pass Rate: 91.2%                │
│ LLM Calls: 450             Tokens: 1.2M ($2.34)           │
│ CPU: 45%  Memory: 2.1GB    Active Sessions: 5             │
├─────────────────────────────────────────────────────────────┤
│ Recent Events:                                              │
│ [10:45:23] INFO  Test generation started (session_abc123)  │
│ [10:45:25] SUCCESS Test generated: 15 tests, 92% coverage  │
│ [10:45:26] WARNING LLM retry attempt 1/3                   │
└─────────────────────────────────────────────────────────────┘
```

**Text Dashboard** (Summary):
- Daily/hourly statistics
- Top errors
- Performance trends
- Cost breakdown

**Prometheus Exporter** (HTTP endpoint):
```
# HELP test_generation_requests_total Total test generation requests
# TYPE test_generation_requests_total counter
test_generation_requests_total{language="python"} 125

# HELP llm_tokens_total Total LLM tokens used
# TYPE llm_tokens_total counter
llm_tokens_total{provider="ollama",type="input"} 850000
llm_tokens_total{provider="ollama",type="output"} 350000
```

---

## 🚀 Migration Path

### Phase 1: Local (Current)

**Stack**: Loguru + TinyDB + Console
- Structured logging to console and files
- Metrics stored in TinyDB (time-series)
- Simple text-based dashboards
- Local alerting

### Phase 2: Prometheus + Grafana (Local)

**Changes**:
1. Replace TinyDB with Prometheus (time-series DB)
2. Add Prometheus client library
3. Expose `/metrics` endpoint
4. Configure Prometheus scraping
5. Import Grafana dashboards

**Migration**:
- Metrics format already Prometheus-compatible
- Update storage backend only
- Keep Loguru for logs (add Loki for centralized logs)

### Phase 3: Cloud (Future)

**Options**:
- **AWS**: CloudWatch + X-Ray
- **GCP**: Cloud Monitoring + Cloud Trace
- **Azure**: Application Insights
- **Vendor**: Datadog, New Relic, Honeycomb

**Migration**:
- Switch to OpenTelemetry SDK
- Configure cloud exporters
- Keep instrumentation unchanged
- Update dashboard imports

---

## 📊 Key Metrics Dashboard

### Test Generation Performance
- Requests/minute (line chart)
- Latency distribution (histogram)
- Error rate (line chart)
- Success rate (gauge)

### Goal Achievement
- Coverage over time (line chart, target: 90%)
- Pass rate over time (line chart, target: 90%)
- Tests generated per language (bar chart)

### LLM Performance
- Calls by provider (pie chart)
- Token usage (stacked area chart)
- Cost tracking (line chart)
- Latency by provider (box plot)

### Agent Activity
- Iterations by agent (bar chart)
- Tool usage frequency (heatmap)
- Decision time (histogram)

### Guardrails
- Checks vs violations (stacked bar)
- PII detections (counter)
- Secret detections (counter)
- Blocks by type (pie chart)

### System Health
- CPU usage (line chart)
- Memory usage (line chart)
- Active sessions (gauge)
- Disk usage (gauge)

---

## 🎯 Success Criteria

**Observability Coverage**:
- ✅ 100% of agents instrumented
- ✅ 100% of LLM calls tracked
- ✅ 100% of guardrails logged
- ✅ 100% of errors captured
- ✅ 95%+ of operations traced

**Performance Impact**:
- ❌ <1% latency overhead
- ❌ <5% memory overhead
- ❌ <1% CPU overhead
- ✅ Async/non-blocking writes

**Data Quality**:
- ✅ Structured, queryable logs
- ✅ Accurate metric values
- ✅ Complete traces
- ✅ Retention policies enforced

---

## 🔐 Security & Privacy

**PII Handling**:
- Scrub PII from logs automatically
- Configurable sensitive fields
- Audit trail for access

**Data Retention**:
- Logs: 7 days local, 30 days cloud
- Metrics: 30 days local, 90 days cloud
- Traces: 3 days local, 14 days cloud

**Access Control**:
- Read-only dashboards
- Admin-only configuration
- Encrypted storage (cloud)

---

## 📚 Best Practices

1. **Structured Logging**: Always use `.bind()` for context
2. **Metric Naming**: Follow Prometheus conventions (`_total`, `_seconds`)
3. **Trace Propagation**: Pass trace context to all async operations
4. **Error Handling**: Log errors with full context before re-raising
5. **Performance**: Use sampling for high-frequency events
6. **Testing**: Mock observability in unit tests
7. **Documentation**: Update runbooks with metric meanings

---

**Status**: Design Complete | Ready for Implementation


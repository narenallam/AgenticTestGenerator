# 🛡️ Guardrails Quick Reference Card

**Quick lookup for guardrail components, checkpoints, and configurations**

---

## 📍 Guardrail Checkpoints (Where They Run)

| Checkpoint | Location | Guardrails Executed | When |
|------------|----------|---------------------|------|
| **1. Input Validation** | `orchestrator.py:349` | Input Guardrails, Secrets Detection | Before LLM receives prompt |
| **2. Tool Authorization** | `guard_manager.py:153` | Schema, Policy, HITL, Budget, Audit | Before each tool call |
| **3. Output Validation** | `orchestrator.py:485` | Output Guardrails, Code Safety, File Boundaries | After test generation |
| **4. Self-Verification** | `constitutional_ai.py` | Constitutional AI principles | Optional post-generation |

---

## 🔟 Guardrail Components (What They Do)

| # | Component | File | Purpose | Enterprise Critical |
|---|-----------|------|---------|---------------------|
| 1 | **Policy Engine** | `policy_engine.py` | ALLOW/DENY/REVIEW decisions, risk tiers | ✅ YES |
| 2 | **Input Guardrails** | `input_guardrails.py` | PII detection, prompt injection, toxic content | ✅ YES |
| 3 | **Output Guardrails** | `output_guardrails.py` | Code safety, dangerous patterns (eval, exec) | ✅ YES |
| 4 | **Constitutional AI** | `constitutional_ai.py` | Self-verification, principle-based evaluation | ⚠️ Recommended |
| 5 | **Budget Tracker** | `budget_tracker.py` | Token/cost/time limits, quota enforcement | ✅ YES |
| 6 | **HITL Manager** | `hitl_manager.py` | Human approval workflows for high-risk actions | ✅ YES |
| 7 | **Audit Logger** | `audit_logger.py` | Compliance event logging (SOC 2, GDPR) | ✅ YES |
| 8 | **Schema Validator** | `schema_validator.py` | Parameter validation, type checking | ✅ YES |
| 9 | **Secrets Scrubber** | `secrets_scrubber.py` | API key leak prevention, credential protection | ✅ YES |
| 10 | **File Boundary** | `file_boundary.py` | Directory restrictions, path traversal prevention | ✅ YES |

---

## ⚙️ Configuration (How to Enable)

### Environment Variables

```bash
# .env
ENABLE_GUARDRAILS=true
ENABLE_HITL=true
ENABLE_AUDIT_LOGGING=true
ENABLE_BUDGET_TRACKING=true
ENABLE_PII_REDACTION=true

# Budget Limits
MAX_TOKENS_PER_DAY=1000000
MAX_COST_PER_MONTH=100.0
MAX_SESSION_TIME_SECONDS=900
```

### Programmatic Configuration

```python
from src.guardrails import create_guard_manager

# Full protection (95% coverage)
guard_manager = create_guard_manager(
    session_id="session_123",
    interactive=True,              # Enable HITL
    enable_input_guards=True,      # PII, prompt injection
    enable_output_guards=True,     # Code safety
    enable_constitutional_ai=True, # Self-verification
    enable_budget_tracking=True    # Cost control
)

# Minimal protection (60% coverage - NOT RECOMMENDED)
guard_manager = create_guard_manager(
    session_id="session_123",
    interactive=False,
    enable_all=False  # Only core guardrails
)
```

---

## 🎯 Risk Tiers (Policy Engine)

| Risk Tier | Examples | Action Required | Approval Needed |
|-----------|----------|-----------------|-----------------|
| **LOW** | Read files, list directories | Auto-allow | ❌ No |
| **MEDIUM** | Write to `tests/` | Auto-allow + log | ❌ No |
| **HIGH** | Write to `src/`, code repairs | HITL approval | ✅ Yes (5 min timeout) |
| **CRITICAL** | External APIs, delete files | Strict HITL | ✅ Yes (mandatory) |

---

## 🚨 Threat Detection (Input Guardrails)

### PII Types Detected

```python
EMAIL, PHONE, SSN, CREDIT_CARD, IP_ADDRESS,
API_KEY, PASSWORD, NAME, ADDRESS
```

### Threat Levels

```python
SAFE      → No issues detected
LOW       → Minor concerns (e.g., informal language)
MEDIUM    → Potential PII or mild injection attempts
HIGH      → Clear PII or injection attempts
CRITICAL  → Severe security threats
```

---

## 💣 Dangerous Patterns (Output Guardrails)

### Code Safety Issues Detected

| Pattern | Description | Severity | Action |
|---------|-------------|----------|--------|
| `eval()`, `exec()` | Code execution | CRITICAL | ❌ Block |
| `os.system()`, `subprocess` | Shell execution | CRITICAL | ❌ Block |
| `open(..., 'w')` | File writes | HIGH | ⚠️ Review |
| `requests.get()` | Network calls | MEDIUM | ⚠️ Log |
| `import pickle` | Unsafe deserialization | HIGH | ❌ Block |
| `while True:` (no break) | Infinite loop | MEDIUM | ⚠️ Warn |

---

## 📊 Budget Limits (Budget Tracker)

### Budget Types

```python
TOKEN → Token count (e.g., 1M tokens/day)
COST  → Dollar cost (e.g., $100/month)
TIME  → Duration (e.g., 15 min/session)
CALL  → API calls (e.g., 1000 calls/hour)
```

### Default Limits

```python
TOKENS: 1,000,000 per day
COST:   $100 per month  
TIME:   900 seconds (15 min) per session
```

### Status Levels

```python
OK       → <80% of limit
WARNING  → 80-99% of limit (send alert)
EXCEEDED → 100%+ of limit (block action)
```

---

## 📝 Audit Events (Audit Logger)

### Event Types

```python
TOOL_CALL         → Every tool invocation
POLICY_DECISION   → ALLOW/DENY/REVIEW
SAFETY_VIOLATION  → Security incidents
HITL_APPROVAL     → Human approvals
BUDGET_LIMIT      → Budget exceeded
ERROR             → System errors
SESSION_START/END → Lifecycle events
```

### Severity Levels

```python
INFO     → Normal operations
WARNING  → Potential issues
ERROR    → Failures
CRITICAL → Security incidents
```

---

## 🏢 Compliance Mapping

| Regulation | Guardrails Required | Database | Status |
|------------|---------------------|----------|--------|
| **SOC 2** | Audit Logger, HITL, Access Control | ✅ SQLite | ✅ Compliant |
| **GDPR** | Input Guardrails (PII), Audit Logger | ✅ SQLite | ✅ Compliant |
| **CCPA** | PII Redaction, Data Minimization | ✅ SQLite | ✅ Compliant |
| **HIPAA** | Encrypted Audit Logs, Access Control | ⚠️ Plaintext | ⚠️ Needs encryption |
| **SOX** | Audit Trails, HITL Approvals | ✅ SQLite | ✅ Compliant |
| **PCI-DSS** | Secrets Scrubber, Audit Logger | ✅ SQLite | ✅ Compliant |

---

## 🎯 Recommended Configuration by Environment

### Development

```python
guard_manager = create_guard_manager(
    interactive=False,              # No HITL prompts
    enable_input_guards=True,       # Catch issues early
    enable_output_guards=True,      # Code safety
    enable_constitutional_ai=False, # Skip for speed
    enable_budget_tracking=True     # Track usage
)
```

### Staging

```python
guard_manager = create_guard_manager(
    interactive=True,               # Test HITL workflows
    enable_input_guards=True,
    enable_output_guards=True,
    enable_constitutional_ai=True,  # Test quality
    enable_budget_tracking=True
)
```

### Production

```python
guard_manager = create_guard_manager(
    interactive=True,               # Human oversight
    enable_input_guards=True,       # MANDATORY
    enable_output_guards=True,      # MANDATORY
    enable_constitutional_ai=True,  # Quality assurance
    enable_budget_tracking=True     # MANDATORY
)

# Set stricter budgets
guard_manager.budget_tracker.set_limit(BudgetType.TOKEN, 500_000, "day")
guard_manager.budget_tracker.set_limit(BudgetType.COST, 50.0, "month")
```

---

## 🚨 Common Violations & Resolutions

| Violation | Cause | Resolution |
|-----------|-------|------------|
| **Input PII detected** | User provided email/phone | Enable `enable_pii_redaction=True` |
| **Prompt injection** | Malicious input | Use sanitized input from guardrail result |
| **Budget exceeded** | Too many tokens | Increase limit or optimize prompts |
| **HITL timeout** | User didn't respond | Adjust timeout or use lower risk tier |
| **Code safety: eval()** | Generated unsafe code | Review and regenerate with safety constraints |
| **File boundary violation** | Writing outside `tests/` | Add Planner justification or fix tool params |
| **Secrets detected** | API key in code | Enable secrets scrubber, check .gitignore |

---

## 📞 Quick Actions

### Check if guardrails are enabled

```python
print(f"Input guards: {guard_manager.input_guardrails is not None}")
print(f"Output guards: {guard_manager.output_guardrails is not None}")
print(f"Budget tracking: {guard_manager.budget_tracker is not None}")
```

### Get session summary

```python
summary = guard_manager.get_session_summary()
print(f"Total events: {summary['total_events']}")
print(f"Violations: {summary['safety_violations']}")
```

### Check budget status

```python
budget_summary = guard_manager.get_budget_summary()
print(f"Token usage: {budget_summary['token_usage']}")
print(f"Cost: ${budget_summary['total_cost']:.2f}")
```

### Query audit logs

```python
from src.guardrails.audit_logger import EventType, Severity

events = guard_manager.audit_logger.query_events(
    session_id="session_123",
    event_type=EventType.SAFETY_VIOLATION,
    severity=Severity.HIGH
)
```

---

## 🔗 Additional Resources

- **Full Documentation**: `GUARDRAILS_README.md` (865 lines)
- **Implementation**: `src/guardrails/` directory
- **Examples**: `examples/` directory
- **Tests**: `tests/` directory

---

**Last Updated**: November 28, 2025  
**Version**: 1.0


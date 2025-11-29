# 📚 Documentation Index

**AgenticTestGenerator - Complete Documentation Library**

---

## 🎯 Core Documentation

| Document | Size | Purpose |
|----------|------|---------|
| **README.md** | 1504 lines | Main project documentation |
| **ARCHITECTURE.md** | 1886 lines | System architecture & design patterns |

---

## 🛡️ Guardrails Documentation

| Document | Size | Purpose |
|----------|------|---------|
| **GUARDRAILS_README.md** | 29 KB | Comprehensive guardrails guide |
| **GUARDRAILS_QUICK_REFERENCE.md** | 9 KB | Quick lookup tables & configs |
| **GUARDRAIL_LIBRARIES_COMPARISON.md** | - | Library alternatives comparison |

**Topics Covered**:
- 10 guardrail components (Policy, Input, Output, Constitutional AI, Budget, HITL, Audit, Schema, Secrets, File Boundaries)
- 4 checkpoints in agent flow
- Enterprise recommendations
- Compliance mapping (SOC 2, GDPR, CCPA, HIPAA, SOX, PCI-DSS)

---

## 📊 EVALS & Observability Documentation

| Document | Size | Purpose |
|----------|------|---------|
| **EVALS_OBSERVABILITY_COMPLETE.md** | 52 KB | Complete EVALS & monitoring guide |
| **EVALS_EXPLAINED.md** | 15 KB | EVALS system overview |
| **QUICKSTART_OBSERVABILITY.md** | 4.2 KB | 5-minute quick start |

**Topics Covered**:
- EVALS system (5-level framework)
- Metrics, KPIs, KPMs definitions
- How EVALS check efficiency & correctness
- Prometheus + Grafana setup
- Alert configuration
- External library recommendations (OpenTelemetry, Jaeger, Sentry, DataDog, etc.)
- Enterprise gaps & solutions

---

## ⚙️ Configuration Files

### Observability Stack

```
config/
├── prometheus/
│   └── prometheus.yml          # Prometheus configuration
├── grafana/
│   ├── datasources/
│   │   └── prometheus.yml      # Grafana datasource
│   └── dashboards/
│       └── dashboard-provider.yml
├── alertmanager/
│   └── alertmanager.yml        # Alert routing & notifications
└── alerts/
    └── agentic_alerts.yml      # 15+ alert rules
```

### Infrastructure

- **docker-compose.yml** - Full observability stack (Prometheus, Grafana, Alertmanager, Jaeger)

---

## 🚀 Quick Navigation

### Getting Started
1. **Setup**: See `README.md` → Installation
2. **Architecture**: See `ARCHITECTURE.md`
3. **Run EVALS**: See `QUICKSTART_OBSERVABILITY.md`

### Security & Compliance
1. **Guardrails Overview**: `GUARDRAILS_README.md`
2. **Quick Reference**: `GUARDRAILS_QUICK_REFERENCE.md`
3. **Library Alternatives**: `GUARDRAIL_LIBRARIES_COMPARISON.md`

### Monitoring & Evaluation
1. **EVALS Guide**: `EVALS_OBSERVABILITY_COMPLETE.md`
2. **Quick Start**: `QUICKSTART_OBSERVABILITY.md`
3. **EVALS Explained**: `EVALS_EXPLAINED.md`

---

## 📂 File Organization

```
AgenticTestGenerator/
├── README.md                              # ⭐ Start here
├── ARCHITECTURE.md                        # System design
├── DOCUMENTATION_INDEX.md                 # This file
│
├── GUARDRAILS_README.md                   # 🛡️ Security & compliance
├── GUARDRAILS_QUICK_REFERENCE.md
├── GUARDRAIL_LIBRARIES_COMPARISON.md
│
├── EVALS_OBSERVABILITY_COMPLETE.md        # 📊 Monitoring & evaluation
├── EVALS_EXPLAINED.md
├── QUICKSTART_OBSERVABILITY.md
│
├── docker-compose.yml                     # ⚙️ Observability stack
├── config/                                # Configuration files
│   ├── prometheus/
│   ├── grafana/
│   ├── alertmanager/
│   └── alerts/
│
├── src/                                   # Source code
│   ├── evals/                            # Evaluation system
│   ├── guardrails/                       # Security guardrails
│   ├── observability/                    # Metrics, logs, traces
│   └── ...
│
└── tests/                                 # Unit tests
```

---

## 📊 Documentation Statistics

- **Total Documentation**: ~109 KB (5 major documents)
- **Configuration Files**: 6 files
- **Docker Compose**: 1 file (4 services)
- **Alert Rules**: 15+ pre-configured alerts
- **Metrics Defined**: 40+ operational metrics
- **KPIs Tracked**: 6 key performance indicators
- **External Libraries Reviewed**: 7 recommendations

---

## 🎯 Key Concepts

### Guardrails (95% Coverage)
- **Core** (60%): Policy, Schema, Audit, HITL
- **Input/Output** (+20%): PII, Injection, Code Safety
- **Constitutional AI** (+10%): Self-verification
- **Budget** (+5%): Cost control

### EVALS (5 Levels)
1. **UNIT**: Function-level testing
2. **COMPONENT**: Module-level metrics
3. **AGENT**: Planner/Coder/Critic performance
4. **SYSTEM**: Safety & guardrails
5. **BUSINESS**: ROI & goal achievement (90/90)

### Observability (4 Pillars)
1. **Metrics**: Prometheus (40+ metrics)
2. **Logs**: TinyDB → Structured JSON
3. **Traces**: Span tracking
4. **Alerts**: 15+ rules (Prometheus Alertmanager)

---

## 🚀 Common Tasks

### Run Evaluation
```bash
python -m src.evals.runner --dataset mixed
```

### Start Observability Stack
```bash
docker-compose up -d
python -m src.observability.prometheus_exporter --port 9090
```

### Access UIs
- Prometheus: http://localhost:9091
- Grafana: http://localhost:3000 (admin/admin123)
- Jaeger: http://localhost:16686

### Query Metrics
```bash
curl http://localhost:9090/metrics
```

---

## 📞 Support

- **Issues**: Report via GitHub Issues
- **Documentation**: See individual README files
- **Configuration**: See `config/` directory
- **Examples**: See `examples/` directory

---

**Last Updated**: November 29, 2025  
**Version**: 1.0  
**Total Documentation**: 109 KB across 5 documents

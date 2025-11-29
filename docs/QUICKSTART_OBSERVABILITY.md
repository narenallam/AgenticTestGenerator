# 🚀 Observability Quick Start

**Get Prometheus + Grafana running in 5 minutes**

---

## ⚡ Fast Track (Docker)

```bash
# Step 1: Start observability stack
docker-compose up -d

# Step 2: Verify services
docker-compose ps

# Step 3: Start metrics exporter
python -m src.observability.prometheus_exporter --port 9090

# Step 4: Access UIs
open http://localhost:9091  # Prometheus
open http://localhost:3000  # Grafana (admin/admin123)
open http://localhost:16686 # Jaeger (optional)

# Step 5: Run evaluation to generate metrics
python -m src.evals.runner --dataset mixed

# Step 6: Check metrics
curl http://localhost:9090/metrics | head -n 50
```

---

## 🎯 What You Get

| Service | URL | Purpose |
|---------|-----|---------|
| **Prometheus Exporter** | http://localhost:9090/metrics | Metrics endpoint |
| **Prometheus Server** | http://localhost:9091 | Metrics DB & Query UI |
| **Grafana** | http://localhost:3000 | Dashboards & Visualization |
| **Alertmanager** | http://localhost:9093 | Alert routing (optional) |
| **Jaeger** | http://localhost:16686 | Distributed tracing (optional) |

---

## 📊 Key Metrics to Monitor

### Test Quality
- `test_coverage_ratio` - Code coverage (target: ≥0.90)
- `test_pass_rate_ratio` - Test pass rate (target: ≥0.90)
- `test_quality_score` - Overall quality (target: ≥0.80)

### Performance
- `llm_call_duration_seconds` - LLM latency (target: p99 <60s)
- `test_generation_duration_seconds` - Test gen time
- `agent_iterations_total` - Agent activity

### Cost
- `llm_tokens_total` - Token consumption
- `llm_cost_total` - Estimated cost

### Security
- `guardrails_violations_total` - Security incidents
- `guardrails_blocks_total` - Blocked actions

---

## 🔍 PromQL Query Examples

```promql
# Coverage over time
test_coverage_ratio

# LLM latency percentiles
histogram_quantile(0.50, llm_call_duration_seconds_bucket)  # p50
histogram_quantile(0.95, llm_call_duration_seconds_bucket)  # p95
histogram_quantile(0.99, llm_call_duration_seconds_bucket)  # p99

# Token usage rate (tokens/second)
rate(llm_tokens_total[5m])

# Error rate
rate(test_generation_errors_total[5m])

# Agent activity
rate(agent_iterations_total{agent="planner"}[5m])

# Cost per hour
increase(llm_cost_total[1h])

# Guardrail effectiveness (% blocked)
rate(guardrails_blocks_total[5m]) / rate(guardrails_checks_total[5m])
```

---

## 🚨 Alerts

Pre-configured alerts in `config/alerts/agentic_alerts.yml`:

| Alert | Threshold | Severity |
|-------|-----------|----------|
| LowTestCoverage | <80% | WARNING |
| LowPassRate | <80% | WARNING |
| HighTestGenerationErrorRate | >0.1/sec | CRITICAL |
| HighLLMLatency | p99 >60s | WARNING |
| LLMBudgetExceeded | >$100 | CRITICAL |
| HighGuardrailViolationRate | >5/sec | WARNING |
| SecretsDetected | >0/hour | CRITICAL |

---

## 🛠️ Troubleshooting

### Exporter not exposing metrics

```bash
# Check if exporter is running
ps aux | grep prometheus_exporter

# Check port is available
lsof -i :9090

# Restart exporter
pkill -f prometheus_exporter
python -m src.observability.prometheus_exporter --port 9090
```

### Prometheus can't scrape

```bash
# Check Prometheus targets
open http://localhost:9091/targets

# For Docker, use host.docker.internal instead of localhost
# Edit config/prometheus/prometheus.yml:
# targets: ['host.docker.internal:9090']

# Reload Prometheus config
curl -X POST http://localhost:9091/-/reload
```

### Grafana datasource not working

```bash
# Check Prometheus is accessible from Grafana
docker exec agentic-grafana curl http://prometheus:9090/api/v1/status/config

# Reconfigure datasource in Grafana UI
# Configuration → Data Sources → Prometheus → Save & Test
```

---

## 📈 Next Steps

1. **Create Custom Dashboard**: Import or create Grafana dashboard
2. **Configure Alerts**: Update `config/alerts/agentic_alerts.yml`
3. **Set Up Notifications**: Configure Slack/email in `config/alertmanager/alertmanager.yml`
4. **Add Tracing**: Integrate OpenTelemetry for distributed tracing
5. **CI/CD Integration**: Add evals to GitHub Actions/GitLab CI

---

## 📚 Full Documentation

- **Complete Guide**: `EVALS_OBSERVABILITY_COMPLETE.md` (52 KB)
- **EVALS Explained**: `EVALS_EXPLAINED.md`
- **Architecture**: `ARCHITECTURE.md`


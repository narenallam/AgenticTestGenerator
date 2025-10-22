"""
Prometheus-compatible metrics collection.

Implements Counter, Gauge, Histogram, and Summary metrics
with TinyDB storage and Prometheus exposition format.
"""

import time
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Dict, List, Optional, Tuple

from tinydb import TinyDB, Query

from .config import get_config


# ═══════════════════════════════════════════════════════════════════════════
# Metric Types
# ═══════════════════════════════════════════════════════════════════════════


class MetricType(str, Enum):
    """Metric type enum."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class Metric:
    """Base metric class."""
    
    def __init__(self, name: str, help_text: str, labels: Optional[List[str]] = None):
        """
        Initialize metric.
        
        Args:
            name: Metric name (must follow Prometheus naming conventions)
            help_text: Metric description
            labels: Label names
        """
        self.name = name
        self.help_text = help_text
        self.labels = labels or []
        self._lock = Lock()
    
    def _validate_labels(self, labels: Dict[str, str]) -> None:
        """Validate provided labels match expected labels."""
        if set(labels.keys()) != set(self.labels):
            raise ValueError(
                f"Expected labels {self.labels}, got {list(labels.keys())}"
            )


class Counter(Metric):
    """Counter metric (monotonically increasing)."""
    
    def __init__(self, name: str, help_text: str, labels: Optional[List[str]] = None):
        """Initialize counter."""
        if not name.endswith('_total'):
            name = f"{name}_total"
        super().__init__(name, help_text, labels)
        self._values: Dict[tuple, float] = defaultdict(float)
    
    def inc(self, amount: float = 1.0, **labels) -> None:
        """Increment counter."""
        if amount < 0:
            raise ValueError("Counter can only increase")
        
        self._validate_labels(labels)
        label_key = tuple(sorted(labels.items()))
        
        with self._lock:
            self._values[label_key] += amount
    
    def get(self, **labels) -> float:
        """Get counter value."""
        self._validate_labels(labels)
        label_key = tuple(sorted(labels.items()))
        return self._values.get(label_key, 0.0)
    
    def get_all(self) -> Dict[tuple, float]:
        """Get all counter values."""
        with self._lock:
            return dict(self._values)


class Gauge(Metric):
    """Gauge metric (can go up and down)."""
    
    def __init__(self, name: str, help_text: str, labels: Optional[List[str]] = None):
        """Initialize gauge."""
        super().__init__(name, help_text, labels)
        self._values: Dict[tuple, float] = defaultdict(float)
    
    def set(self, value: float, **labels) -> None:
        """Set gauge value."""
        self._validate_labels(labels)
        label_key = tuple(sorted(labels.items()))
        
        with self._lock:
            self._values[label_key] = value
    
    def inc(self, amount: float = 1.0, **labels) -> None:
        """Increment gauge."""
        self._validate_labels(labels)
        label_key = tuple(sorted(labels.items()))
        
        with self._lock:
            self._values[label_key] += amount
    
    def dec(self, amount: float = 1.0, **labels) -> None:
        """Decrement gauge."""
        self.inc(-amount, **labels)
    
    def get(self, **labels) -> float:
        """Get gauge value."""
        self._validate_labels(labels)
        label_key = tuple(sorted(labels.items()))
        return self._values.get(label_key, 0.0)
    
    def get_all(self) -> Dict[tuple, float]:
        """Get all gauge values."""
        with self._lock:
            return dict(self._values)


class Histogram(Metric):
    """Histogram metric (distribution of observations)."""
    
    # Default buckets (Prometheus default)
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf')]
    
    def __init__(
        self,
        name: str,
        help_text: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None
    ):
        """Initialize histogram."""
        if not name.endswith('_seconds') and not name.endswith('_bytes'):
            name = f"{name}_seconds"
        super().__init__(name, help_text, labels)
        self.buckets = sorted(buckets) if buckets else self.DEFAULT_BUCKETS
        self._sum: Dict[tuple, float] = defaultdict(float)
        self._count: Dict[tuple, int] = defaultdict(int)
        self._buckets: Dict[tuple, Dict[float, int]] = defaultdict(lambda: {b: 0 for b in self.buckets})
    
    def observe(self, value: float, **labels) -> None:
        """Observe a value."""
        self._validate_labels(labels)
        label_key = tuple(sorted(labels.items()))
        
        with self._lock:
            self._sum[label_key] += value
            self._count[label_key] += 1
            
            # Update buckets
            for bucket in self.buckets:
                if value <= bucket:
                    self._buckets[label_key][bucket] += 1
    
    def get_sum(self, **labels) -> float:
        """Get sum of observations."""
        self._validate_labels(labels)
        label_key = tuple(sorted(labels.items()))
        return self._sum.get(label_key, 0.0)
    
    def get_count(self, **labels) -> int:
        """Get count of observations."""
        self._validate_labels(labels)
        label_key = tuple(sorted(labels.items()))
        return self._count.get(label_key, 0)
    
    def get_buckets(self, **labels) -> Dict[float, int]:
        """Get bucket counts."""
        self._validate_labels(labels)
        label_key = tuple(sorted(labels.items()))
        return self._buckets.get(label_key, {b: 0 for b in self.buckets})


# ═══════════════════════════════════════════════════════════════════════════
# Metrics Registry
# ═══════════════════════════════════════════════════════════════════════════


class MetricsRegistry:
    """Registry for all metrics."""
    
    def __init__(self):
        """Initialize registry."""
        self.metrics: Dict[str, Metric] = {}
        self._lock = Lock()
        self.config = get_config()
        self.db = TinyDB(self.config.get_metrics_db_path())
        self.table = self.db.table('metrics')
    
    def register(self, metric: Metric) -> Metric:
        """Register a metric."""
        with self._lock:
            if metric.name in self.metrics:
                raise ValueError(f"Metric {metric.name} already registered")
            self.metrics[metric.name] = metric
            return metric
    
    def get(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        return self.metrics.get(name)
    
    def counter(self, name: str, help_text: str, labels: Optional[List[str]] = None) -> Counter:
        """Get or create a counter."""
        metric = self.get(name)
        if metric is None:
            metric = Counter(name, help_text, labels)
            self.register(metric)
        return metric
    
    def gauge(self, name: str, help_text: str, labels: Optional[List[str]] = None) -> Gauge:
        """Get or create a gauge."""
        metric = self.get(name)
        if metric is None:
            metric = Gauge(name, help_text, labels)
            self.register(metric)
        return metric
    
    def histogram(
        self,
        name: str,
        help_text: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None
    ) -> Histogram:
        """Get or create a histogram."""
        metric = self.get(name)
        if metric is None:
            metric = Histogram(name, help_text, labels, buckets)
            self.register(metric)
        return metric
    
    def flush_to_db(self) -> None:
        """Flush current metrics to database."""
        timestamp = datetime.utcnow()
        
        for name, metric in self.metrics.items():
            if isinstance(metric, (Counter, Gauge)):
                for label_tuple, value in metric.get_all().items():
                    labels_dict = dict(label_tuple)
                    self.table.insert({
                        'timestamp': timestamp.isoformat(),
                        'metric_name': name,
                        'metric_type': MetricType.COUNTER.value if isinstance(metric, Counter) else MetricType.GAUGE.value,
                        'value': value,
                        'labels': labels_dict,
                    })
            
            elif isinstance(metric, Histogram):
                # Store histogram summary
                for label_tuple in metric._sum.keys():
                    labels_dict = dict(label_tuple)
                    self.table.insert({
                        'timestamp': timestamp.isoformat(),
                        'metric_name': name,
                        'metric_type': MetricType.HISTOGRAM.value,
                        'sum': metric.get_sum(**labels_dict),
                        'count': metric.get_count(**labels_dict),
                        'labels': labels_dict,
                    })
    
    def cleanup_old_metrics(self, days: int = 30) -> None:
        """Remove metrics older than specified days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        
        query = Query()
        self.table.remove(query.timestamp < cutoff_str)
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus exposition format."""
        lines = []
        
        for name, metric in self.metrics.items():
            # HELP line
            lines.append(f"# HELP {metric.name} {metric.help_text}")
            
            # TYPE line
            if isinstance(metric, Counter):
                lines.append(f"# TYPE {metric.name} counter")
                for label_tuple, value in metric.get_all().items():
                    label_str = self._format_labels(label_tuple)
                    lines.append(f"{metric.name}{label_str} {value}")
            
            elif isinstance(metric, Gauge):
                lines.append(f"# TYPE {metric.name} gauge")
                for label_tuple, value in metric.get_all().items():
                    label_str = self._format_labels(label_tuple)
                    lines.append(f"{metric.name}{label_str} {value}")
            
            elif isinstance(metric, Histogram):
                lines.append(f"# TYPE {metric.name} histogram")
                for label_tuple in metric._sum.keys():
                    labels_dict = dict(label_tuple)
                    
                    # Buckets
                    buckets = metric.get_buckets(**labels_dict)
                    for bucket, count in buckets.items():
                        bucket_label = f'le="{bucket}"'
                        label_str = self._format_labels(label_tuple, extra=bucket_label)
                        lines.append(f"{metric.name}_bucket{label_str} {count}")
                    
                    # Sum and count
                    label_str = self._format_labels(label_tuple)
                    lines.append(f"{metric.name}_sum{label_str} {metric.get_sum(**labels_dict)}")
                    lines.append(f"{metric.name}_count{label_str} {metric.get_count(**labels_dict)}")
            
            lines.append("")  # Empty line between metrics
        
        return "\n".join(lines)
    
    def _format_labels(self, label_tuple: tuple, extra: Optional[str] = None) -> str:
        """Format labels for Prometheus format."""
        if not label_tuple and not extra:
            return ""
        
        labels = [f'{k}="{v}"' for k, v in label_tuple]
        if extra:
            labels.append(extra)
        
        return "{" + ",".join(labels) + "}"


# ═══════════════════════════════════════════════════════════════════════════
# Global Registry
# ═══════════════════════════════════════════════════════════════════════════


_registry: Optional[MetricsRegistry] = None


def get_registry() -> MetricsRegistry:
    """Get global metrics registry."""
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry


def counter(name: str, help_text: str, labels: Optional[List[str]] = None) -> Counter:
    """Get or create a counter metric."""
    return get_registry().counter(name, help_text, labels)


def gauge(name: str, help_text: str, labels: Optional[List[str]] = None) -> Gauge:
    """Get or create a gauge metric."""
    return get_registry().gauge(name, help_text, labels)


def histogram(
    name: str,
    help_text: str,
    labels: Optional[List[str]] = None,
    buckets: Optional[List[float]] = None
) -> Histogram:
    """Get or create a histogram metric."""
    return get_registry().histogram(name, help_text, labels, buckets)


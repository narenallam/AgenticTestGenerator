"""
Observability configuration.

Centralized settings for logging, metrics, tracing, and alerting.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ObservabilityConfig(BaseModel):
    """Observability configuration."""
    
    # General
    enabled: bool = Field(default=True, description="Enable observability")
    workspace_dir: Path = Field(default=Path("observability"), description="Workspace directory")
    
    # Logging
    log_level: str = Field(default="INFO", description="Log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    log_to_console: bool = Field(default=True, description="Log to console")
    log_to_file: bool = Field(default=True, description="Log to file")
    log_to_db: bool = Field(default=True, description="Log to database")
    log_file_rotation: str = Field(default="1 day", description="Log file rotation")
    log_retention: str = Field(default="7 days", description="Log retention period")
    log_format_console: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        description="Console log format"
    )
    log_format_file: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        description="File log format"
    )
    
    # Metrics
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    metrics_flush_interval: int = Field(default=10, description="Metrics flush interval (seconds)")
    metrics_retention_days: int = Field(default=30, description="Metrics retention (days)")
    prometheus_port: int = Field(default=9090, description="Prometheus exporter port")
    
    # Tracing
    tracing_enabled: bool = Field(default=True, description="Enable tracing")
    trace_retention_days: int = Field(default=3, description="Trace retention (days)")
    trace_sampling_rate: float = Field(default=1.0, description="Trace sampling rate (0.0-1.0)")
    
    # Alerting
    alerting_enabled: bool = Field(default=True, description="Enable alerting")
    alert_error_rate_threshold: float = Field(default=0.05, description="Error rate threshold (5%)")
    alert_latency_p99_threshold: float = Field(default=60.0, description="p99 latency threshold (seconds)")
    alert_coverage_threshold: float = Field(default=0.80, description="Coverage threshold (80%)")
    alert_cpu_threshold: float = Field(default=0.90, description="CPU threshold (90%)")
    
    # Performance
    async_writes: bool = Field(default=True, description="Use async writes for performance")
    buffer_size: int = Field(default=1000, description="Buffer size for batch writes")
    
    # Privacy
    scrub_pii: bool = Field(default=True, description="Scrub PII from logs")
    sensitive_fields: list = Field(
        default_factory=lambda: ["password", "api_key", "secret", "token", "credential"],
        description="Sensitive field names to scrub"
    )
    
    def get_log_dir(self) -> Path:
        """Get log directory path."""
        log_dir = self.workspace_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    
    def get_metrics_db_path(self) -> Path:
        """Get metrics database path."""
        return self.workspace_dir / "metrics.json"
    
    def get_logs_db_path(self) -> Path:
        """Get logs database path."""
        return self.workspace_dir / "logs.json"
    
    def get_traces_db_path(self) -> Path:
        """Get traces database path."""
        return self.workspace_dir / "traces.json"


# Global configuration instance
_config: Optional[ObservabilityConfig] = None


def get_config() -> ObservabilityConfig:
    """Get or create observability configuration."""
    global _config
    if _config is None:
        _config = ObservabilityConfig()
        # Ensure workspace directory exists
        _config.workspace_dir.mkdir(parents=True, exist_ok=True)
    return _config


def set_config(config: ObservabilityConfig) -> None:
    """Set observability configuration."""
    global _config
    _config = config


def update_config(**kwargs) -> None:
    """Update observability configuration."""
    config = get_config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)


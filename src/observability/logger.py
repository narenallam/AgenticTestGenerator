"""
Structured logging with Loguru.

Provides console, file, and database logging with context binding,
PII scrubbing, and standard logging interception.
"""

import logging
import re
import sys
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger
from tinydb import TinyDB

from .config import get_config

# Context variables for request tracking
trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar("session_id", default=None)


# ═══════════════════════════════════════════════════════════════════════════
# PII Scrubbing
# ═══════════════════════════════════════════════════════════════════════════


class PIIScrubber:
    """Scrub PII from log messages."""
    
    # Common PII patterns
    PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        "api_key": r'\b[A-Za-z0-9]{32,}\b',
    }
    
    @classmethod
    def scrub(cls, text: str, sensitive_fields: list = None) -> str:
        """Scrub PII from text."""
        if not text:
            return text
        
        # Scrub known patterns
        for name, pattern in cls.PATTERNS.items():
            text = re.sub(pattern, f"[REDACTED_{name.upper()}]", text)
        
        # Scrub sensitive field values
        if sensitive_fields:
            for field in sensitive_fields:
                # Match field="value" or field: value patterns
                pattern = rf'{field}["\']?\s*[:=]\s*["\']?([^"\'\s,}}]+)'
                text = re.sub(pattern, f'{field}="[REDACTED]"', text, flags=re.IGNORECASE)
        
        return text


# ═══════════════════════════════════════════════════════════════════════════
# Custom Sinks
# ═══════════════════════════════════════════════════════════════════════════


class DatabaseSink:
    """TinyDB sink for structured log storage."""
    
    def __init__(self, db_path: Path):
        """Initialize database sink."""
        self.db = TinyDB(db_path)
        self.table = self.db.table('logs')
    
    def __call__(self, message):
        """Write log record to database."""
        record = message.record
        
        # Extract context
        extra = record.get("extra", {})
        
        log_entry = {
            'timestamp': record['time'].isoformat(),
            'level': record['level'].name,
            'message': record['message'],
            'module': record['name'],
            'function': record['function'],
            'line': record['line'],
            'file': record['file'].name,
            'process_id': record['process'].id,
            'thread_name': record['thread'].name,
            
            # Context
            'trace_id': extra.get('trace_id', trace_id_var.get()),
            'session_id': extra.get('session_id', session_id_var.get()),
            'agent': extra.get('agent'),
            'operation': extra.get('operation'),
            
            # Additional context
            'context': {
                k: v for k, v in extra.items()
                if k not in ['trace_id', 'session_id', 'agent', 'operation']
            },
            
            # Exception info
            'exception': str(record['exception']) if record['exception'] else None,
        }
        
        self.table.insert(log_entry)
    
    def close(self):
        """Close database connection."""
        self.db.close()


# ═══════════════════════════════════════════════════════════════════════════
# Standard Logging Interception
# ═══════════════════════════════════════════════════════════════════════════


class InterceptHandler(logging.Handler):
    """Intercept standard logging and redirect to Loguru."""
    
    def emit(self, record):
        """Emit a record."""
        # Get corresponding Loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # Find caller
        frame = logging.currentframe()
        depth = 0
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# ═══════════════════════════════════════════════════════════════════════════
# Logger Setup
# ═══════════════════════════════════════════════════════════════════════════


class ObservabilityLogger:
    """Enhanced logger with observability features."""
    
    def __init__(self):
        """Initialize observability logger."""
        self.config = get_config()
        self.db_sink = None
        self._initialized = False
    
    def setup(self) -> None:
        """Setup Loguru with all sinks."""
        if self._initialized:
            return
        
        # Remove default handler
        logger.remove()
        
        # Add console sink
        if self.config.log_to_console:
            logger.add(
                sys.stdout,
                format=self.config.log_format_console,
                level=self.config.log_level,
                colorize=True,
                filter=self._filter_with_scrubbing,
            )
        
        # Add file sink
        if self.config.log_to_file:
            log_dir = self.config.get_log_dir()
            logger.add(
                log_dir / "app_{time:YYYY-MM-DD}.log",
                format=self.config.log_format_file,
                level=self.config.log_level,
                rotation=self.config.log_file_rotation,
                retention=self.config.log_retention,
                compression="zip",
                filter=self._filter_with_scrubbing,
            )
        
        # Add database sink
        if self.config.log_to_db:
            self.db_sink = DatabaseSink(self.config.get_logs_db_path())
            logger.add(
                self.db_sink,
                level=self.config.log_level,
                filter=self._filter_with_scrubbing,
            )
        
        # Intercept standard logging
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
        
        self._initialized = True
        logger.info("Observability logger initialized")
    
    def _filter_with_scrubbing(self, record):
        """Filter and scrub PII from log records."""
        if self.config.scrub_pii:
            # Scrub message
            record["message"] = PIIScrubber.scrub(
                record["message"],
                self.config.sensitive_fields
            )
            
            # Scrub extra fields
            if "extra" in record:
                for key, value in record["extra"].items():
                    if isinstance(value, str):
                        record["extra"][key] = PIIScrubber.scrub(
                            value,
                            self.config.sensitive_fields
                        )
        
        return True
    
    def shutdown(self) -> None:
        """Shutdown logger and cleanup resources."""
        if self.db_sink:
            self.db_sink.close()
        logger.info("Observability logger shutdown")


# ═══════════════════════════════════════════════════════════════════════════
# Global Logger Instance
# ═══════════════════════════════════════════════════════════════════════════


_obs_logger: Optional[ObservabilityLogger] = None


def get_logger():
    """Get Loguru logger instance."""
    global _obs_logger
    if _obs_logger is None:
        _obs_logger = ObservabilityLogger()
        if get_config().enabled:
            _obs_logger.setup()
    return logger


def bind_context(**kwargs) -> logger:
    """Bind context to logger."""
    return logger.bind(**kwargs)


def set_trace_id(trace_id: str) -> None:
    """Set trace ID for current context."""
    trace_id_var.set(trace_id)


def get_trace_id() -> Optional[str]:
    """Get trace ID from current context."""
    return trace_id_var.get()


def set_session_id(session_id: str) -> None:
    """Set session ID for current context."""
    session_id_var.set(session_id)


def get_session_id() -> Optional[str]:
    """Get session ID from current context."""
    return session_id_var.get()


# Initialize logger on import
if get_config().enabled:
    get_logger()


"""
Guardrails module for comprehensive safety and compliance.

This module provides a complete set of guardrails for safe agentic execution.
Coverage: 95% (60% core + 35% advanced)

All guardrails are unified in one place:
- Policy & orchestration guardrails
- Input/output validation guardrails
- Security-focused guardrails (secrets, files, determinism)
"""

from src.guardrails.audit_logger import (
    AuditEvent,
    AuditLogger,
    EventType,
    Severity,
    create_audit_logger,
)
from src.guardrails.budget_tracker import (
    BudgetTracker,
    BudgetType,
    BudgetStatus,
    TokenCostEstimate,
    create_budget_tracker,
)
from src.guardrails.constitutional_ai import (
    ConstitutionalAI,
    ConstitutionalPrinciple,
    SelfVerificationResult,
    create_constitutional_ai,
)
from src.guardrails.determinism_checker import (
    DeterminismEnforcer,
    DeterminismViolation,
)
from src.guardrails.file_boundary import (
    FileBoundaryChecker,
    FileBoundaryViolation,
    FileBoundaryConfig,
)
from src.guardrails.guard_manager import (
    GuardManager,
    GuardResult,
    create_guard_manager,
)
from src.guardrails.hitl_manager import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
    HITLManager,
    RiskLevel,
    create_hitl_manager,
)
from src.guardrails.input_guardrails import (
    InputGuardrails,
    InputScanResult,
    PIIType,
    ThreatLevel,
    create_input_guardrails,
)
from src.guardrails.output_guardrails import (
    OutputGuardrails,
    OutputScanResult,
    CodeSafetyIssue,
    LicenseType,
    create_output_guardrails,
)
from src.guardrails.policy_engine import (
    PolicyConstraints,
    PolicyDecision,
    PolicyEngine,
    PolicyResult,
    PolicyRule,
    RiskTier,
    create_policy_engine,
)
from src.guardrails.schema_validator import (
    SchemaValidator,
    ToolSchema,
    ValidationError,
    ValidationResult,
    create_schema_validator,
)
from src.guardrails.secrets_scrubber import (
    SecretsScrubber,
    SecretsConfig,
)

__all__ = [
    # Guard Manager (Unified Interface - 95% Coverage)
    "GuardManager",
    "GuardResult",
    "create_guard_manager",
    # Policy Engine
    "PolicyEngine",
    "PolicyDecision",
    "PolicyResult",
    "PolicyRule",
    "PolicyConstraints",
    "RiskTier",
    "create_policy_engine",
    # Schema Validator
    "SchemaValidator",
    "ToolSchema",
    "ValidationResult",
    "ValidationError",
    "create_schema_validator",
    # Audit Logger
    "AuditLogger",
    "AuditEvent",
    "EventType",
    "Severity",
    "create_audit_logger",
    # HITL Manager
    "HITLManager",
    "ApprovalRequest",
    "ApprovalResponse",
    "ApprovalDecision",
    "RiskLevel",
    "create_hitl_manager",
    # Input Guardrails (Advanced)
    "InputGuardrails",
    "InputScanResult",
    "PIIType",
    "ThreatLevel",
    "create_input_guardrails",
    # Output Guardrails (Advanced)
    "OutputGuardrails",
    "OutputScanResult",
    "CodeSafetyIssue",
    "LicenseType",
    "create_output_guardrails",
    # Constitutional AI (Advanced)
    "ConstitutionalAI",
    "ConstitutionalPrinciple",
    "SelfVerificationResult",
    "create_constitutional_ai",
    # Budget Tracker (Advanced)
    "BudgetTracker",
    "BudgetType",
    "BudgetStatus",
    "TokenCostEstimate",
    "create_budget_tracker",
    # Security Guardrails (Specialized)
    "SecretsScrubber",
    "SecretsConfig",
    "FileBoundaryChecker",
    "FileBoundaryViolation",
    "FileBoundaryConfig",
    "DeterminismEnforcer",
    "DeterminismViolation",
]

"""
Policy Engine for centralized ALLOW/DENY/REVIEW decisions.

Implements risk-based access control for all agent actions.
"""

from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field
from rich.console import Console

console = Console()


class RiskTier(str, Enum):
    """Risk levels for tools and actions."""
    LOW = "low"              # Read-only, safe operations
    MEDIUM = "medium"        # Writes to tests/, reversible
    HIGH = "high"            # Writes to src/, repairs
    CRITICAL = "critical"    # External APIs, irreversible actions


class PolicyDecision(str, Enum):
    """Policy decision outcomes."""
    ALLOW = "allow"          # Proceed automatically
    DENY = "deny"            # Block action
    REVIEW = "review"        # Require human approval


class ToolConstraints(BaseModel):
    """Constraints for a specific tool."""
    
    risk_tier: RiskTier = Field(..., description="Risk level")
    max_calls_per_session: int = Field(default=100, description="Max calls per session")
    requires_justification: bool = Field(default=False, description="Needs Planner justification")
    parameter_bounds: Dict[str, Any] = Field(default_factory=dict, description="Parameter limits")
    allowed_contexts: List[str] = Field(default_factory=list, description="Allowed execution contexts")


class PolicyRule(BaseModel):
    """A policy rule for decision making."""
    
    id: str = Field(..., description="Rule ID")
    name: str = Field(..., description="Rule name")
    condition: str = Field(..., description="Condition to evaluate")
    decision: PolicyDecision = Field(..., description="Decision if condition matches")
    reason: str = Field(..., description="Reason for this rule")
    priority: int = Field(default=10, description="Rule priority (higher = checked first)")


class PolicyContext(BaseModel):
    """Context for policy evaluation."""
    
    user_role: str = Field(default="user", description="User role")
    session_id: str = Field(..., description="Session identifier")
    tool_usage_count: Dict[str, int] = Field(default_factory=dict, description="Tool call counts")
    time_elapsed: float = Field(default=0.0, description="Session time elapsed (seconds)")
    budget_remaining: Dict[str, float] = Field(default_factory=dict, description="Remaining budgets")


class PolicyEngine:
    """
    Central policy engine for all agent actions.
    
    Implements defense-in-depth with:
    - Risk-based access control
    - Parameter validation
    - Budget enforcement
    - Human-in-the-loop gating
    
    Example:
        >>> engine = PolicyEngine()
        >>> decision = engine.evaluate(
        ...     tool="generate_tests",
        ...     params={"max_iterations": 5},
        ...     context=PolicyContext(session_id="123")
        ... )
        >>> if decision.decision == PolicyDecision.DENY:
        ...     raise SecurityError(decision.reason)
    """
    
    def __init__(self):
        """Initialize policy engine with default rules and constraints."""
        self.tool_constraints = self._define_tool_constraints()
        self.rules = self._define_policy_rules()
        self._call_counts: Dict[str, int] = {}
        
        console.print("[bold green]🛡️  Policy Engine Initialized[/bold green]")
    
    def evaluate(
        self,
        tool: str,
        params: Dict[str, Any],
        context: PolicyContext
    ) -> "PolicyEvaluationResult":
        """
        Evaluate if action should be allowed.
        
        Args:
            tool: Tool name being called
            params: Tool parameters
            context: Execution context
            
        Returns:
            PolicyEvaluationResult with decision and reason
        """
        # Get tool constraints
        if tool not in self.tool_constraints:
            return PolicyEvaluationResult(
                decision=PolicyDecision.DENY,
                reason=f"Unknown tool: {tool}",
                risk_tier=RiskTier.CRITICAL
            )
        
        constraints = self.tool_constraints[tool]
        
        # Check call count limits
        self._call_counts[tool] = self._call_counts.get(tool, 0) + 1
        if self._call_counts[tool] > constraints.max_calls_per_session:
            return PolicyEvaluationResult(
                decision=PolicyDecision.DENY,
                reason=f"Tool {tool} exceeded max calls ({constraints.max_calls_per_session})",
                risk_tier=constraints.risk_tier
            )
        
        # Check parameter bounds
        param_check = self._check_parameter_bounds(params, constraints.parameter_bounds)
        if not param_check.valid:
            return PolicyEvaluationResult(
                decision=PolicyDecision.DENY,
                reason=f"Parameter violation: {param_check.reason}",
                risk_tier=constraints.risk_tier
            )
        
        # Evaluate policy rules
        for rule in sorted(self.rules, key=lambda r: r.priority, reverse=True):
            if self._evaluate_rule_condition(rule, tool, params, context, constraints):
                console.print(f"[blue]📋 Policy Rule Matched: {rule.name}[/blue]")
                return PolicyEvaluationResult(
                    decision=rule.decision,
                    reason=rule.reason,
                    risk_tier=constraints.risk_tier,
                    rule_id=rule.id
                )
        
        # Default: Allow if no rules matched
        return PolicyEvaluationResult(
            decision=PolicyDecision.ALLOW,
            reason="No restricting rules matched",
            risk_tier=constraints.risk_tier
        )
    
    def _define_tool_constraints(self) -> Dict[str, ToolConstraints]:
        """Define constraints for each tool."""
        return {
            # Git & Analysis Tools (LOW risk)
            "check_git_changes": ToolConstraints(
                risk_tier=RiskTier.LOW,
                max_calls_per_session=50,
                parameter_bounds={}
            ),
            "search_code": ToolConstraints(
                risk_tier=RiskTier.LOW,
                max_calls_per_session=100,
                parameter_bounds={
                    "query": {"max_length": 500},
                    "max_results": {"min": 1, "max": 50}
                }
            ),
            "get_code_context": ToolConstraints(
                risk_tier=RiskTier.LOW,
                max_calls_per_session=100,
                parameter_bounds={}
            ),
            "analyze_ast": ToolConstraints(
                risk_tier=RiskTier.LOW,
                max_calls_per_session=50,
                parameter_bounds={}
            ),
            
            # Test Generation (MEDIUM risk)
            "generate_tests": ToolConstraints(
                risk_tier=RiskTier.MEDIUM,
                max_calls_per_session=50,
                requires_justification=False,
                parameter_bounds={
                    "max_iterations": {"min": 1, "max": 10},
                    "target_coverage": {"min": 0, "max": 100}
                }
            ),
            "execute_tests": ToolConstraints(
                risk_tier=RiskTier.MEDIUM,
                max_calls_per_session=100,
                parameter_bounds={
                    "timeout": {"min": 1, "max": 60}
                }
            ),
            
            # Code Modification (HIGH risk)
            "repair_code": ToolConstraints(
                risk_tier=RiskTier.HIGH,
                max_calls_per_session=10,
                requires_justification=True,
                parameter_bounds={}
            ),
            "write_to_src": ToolConstraints(
                risk_tier=RiskTier.HIGH,
                max_calls_per_session=5,
                requires_justification=True,
                parameter_bounds={}
            ),
        }
    
    def _define_policy_rules(self) -> List[PolicyRule]:
        """Define policy rules for decision making."""
        return [
            # High-risk actions require review
            PolicyRule(
                id="rule_001",
                name="High Risk Requires Approval",
                condition="risk_tier == HIGH",
                decision=PolicyDecision.REVIEW,
                reason="High-risk action requires human approval",
                priority=100
            ),
            
            # Critical actions are denied by default
            PolicyRule(
                id="rule_002",
                name="Critical Actions Denied",
                condition="risk_tier == CRITICAL",
                decision=PolicyDecision.DENY,
                reason="Critical actions not allowed in this version",
                priority=200
            ),
            
            # Budget exceeded
            PolicyRule(
                id="rule_003",
                name="Time Budget Exceeded",
                condition="time_elapsed > 900",  # 15 minutes
                decision=PolicyDecision.DENY,
                reason="Session time budget exceeded (15 min)",
                priority=150
            ),
            
            # Too many iterations
            PolicyRule(
                id="rule_004",
                name="Max Iterations Exceeded",
                condition="max_iterations > 10",
                decision=PolicyDecision.DENY,
                reason="max_iterations must be ≤ 10",
                priority=90
            ),
            
            # Justification required but missing
            PolicyRule(
                id="rule_005",
                name="Justification Required",
                condition="requires_justification AND no_justification",
                decision=PolicyDecision.REVIEW,
                reason="Planner justification required for this action",
                priority=120
            ),
        ]
    
    def _evaluate_rule_condition(
        self,
        rule: PolicyRule,
        tool: str,
        params: Dict[str, Any],
        context: PolicyContext,
        constraints: ToolConstraints
    ) -> bool:
        """Evaluate if a rule condition matches."""
        condition = rule.condition
        
        # Simple condition evaluation
        if "risk_tier == HIGH" in condition:
            return constraints.risk_tier == RiskTier.HIGH
        
        if "risk_tier == CRITICAL" in condition:
            return constraints.risk_tier == RiskTier.CRITICAL
        
        if "time_elapsed >" in condition:
            threshold = float(condition.split(">")[1].strip())
            return context.time_elapsed > threshold
        
        if "max_iterations >" in condition:
            threshold = int(condition.split(">")[1].strip())
            return params.get("max_iterations", 0) > threshold
        
        if "requires_justification" in condition:
            return constraints.requires_justification and not context.budget_remaining.get("justification")
        
        return False
    
    def _check_parameter_bounds(
        self,
        params: Dict[str, Any],
        bounds: Dict[str, Dict[str, Any]]
    ) -> "ParameterCheckResult":
        """Check if parameters are within allowed bounds."""
        for param_name, value in params.items():
            if param_name not in bounds:
                continue
            
            param_bounds = bounds[param_name]
            
            # Check min/max for numbers
            if "min" in param_bounds:
                if isinstance(value, (int, float)) and value < param_bounds["min"]:
                    return ParameterCheckResult(
                        valid=False,
                        reason=f"{param_name} must be >= {param_bounds['min']}"
                    )
            
            if "max" in param_bounds:
                if isinstance(value, (int, float)) and value > param_bounds["max"]:
                    return ParameterCheckResult(
                        valid=False,
                        reason=f"{param_name} must be <= {param_bounds['max']}"
                    )
            
            # Check max_length for strings
            if "max_length" in param_bounds:
                if isinstance(value, str) and len(value) > param_bounds["max_length"]:
                    return ParameterCheckResult(
                        valid=False,
                        reason=f"{param_name} exceeds max length {param_bounds['max_length']}"
                    )
        
        return ParameterCheckResult(valid=True, reason="All parameters within bounds")
    
    def reset_session(self):
        """Reset session state."""
        self._call_counts.clear()
        console.print("[yellow]🔄 Policy Engine Session Reset[/yellow]")


class PolicyEvaluationResult(BaseModel):
    """Result of policy evaluation."""
    
    decision: PolicyDecision = Field(..., description="Allow/Deny/Review")
    reason: str = Field(..., description="Reason for decision")
    risk_tier: RiskTier = Field(..., description="Risk level")
    rule_id: Optional[str] = Field(default=None, description="Matched rule ID")


class ParameterCheckResult(BaseModel):
    """Result of parameter bounds checking."""
    
    valid: bool = Field(..., description="Parameters valid")
    reason: str = Field(..., description="Validation message")


def create_policy_engine() -> PolicyEngine:
    """Factory function to create policy engine."""
    return PolicyEngine()


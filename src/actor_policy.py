"""
Actor policy for intelligent tool selection.

Implements explicit policies for tool selection based on:
- Task state
- Tool success history
- Resource constraints
- Strategic planning
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from rich.console import Console

console = Console()


class ToolPriority(str, Enum):
    """Priority levels for tool selection."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ToolCategory(str, Enum):
    """Categories of available tools."""
    
    CODE_ANALYSIS = "code_analysis"
    GENERATION = "generation"
    EXECUTION = "execution"
    QUALITY = "quality"
    RETRIEVAL = "retrieval"


class ToolMetrics(BaseModel):
    """Metrics for a tool's performance."""
    
    uses: int = Field(default=0, description="Times used")
    successes: int = Field(default=0, description="Successful executions")
    failures: int = Field(default=0, description="Failed executions")
    avg_duration: float = Field(default=0.0, description="Average duration")
    last_success: bool = Field(default=True, description="Last execution succeeded")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.uses == 0:
            return 1.0
        return self.successes / self.uses
    
    @property
    def reliability_score(self) -> float:
        """Calculate reliability score (0-1)."""
        if self.uses == 0:
            return 0.8  # Initial assumption
        
        success_rate = self.success_rate
        recency_bonus = 0.2 if self.last_success else 0.0
        
        return min(success_rate + recency_bonus, 1.0)


class PolicyRule(BaseModel):
    """A policy rule for tool selection."""
    
    condition: str = Field(..., description="Condition for rule")
    tool_name: str = Field(..., description="Tool to select")
    priority: ToolPriority = Field(..., description="Priority level")
    reason: str = Field(..., description="Reason for selection")


class ActorPolicy:
    """
    Explicit policy for tool selection.
    
    Makes intelligent decisions about which tool to use based on:
    - Current state
    - Tool performance history
    - Strategic considerations
    """
    
    def __init__(self):
        self.tool_metrics: Dict[str, ToolMetrics] = {}
        self.rules: List[PolicyRule] = self._initialize_rules()
        self.execution_history: List[dict] = []
        
        console.print("[green]✓[/green] Actor policy initialized")
    
    def _initialize_rules(self) -> List[PolicyRule]:
        """Initialize policy rules."""
        return [
            PolicyRule(
                condition="no_code_analyzed",
                tool_name="check_git_changes",
                priority=ToolPriority.CRITICAL,
                reason="Must identify what code needs testing"
            ),
            PolicyRule(
                condition="code_identified_no_context",
                tool_name="get_code_context",
                priority=ToolPriority.HIGH,
                reason="Need context before generation"
            ),
            PolicyRule(
                condition="context_available_no_tests",
                tool_name="generate_tests",
                priority=ToolPriority.HIGH,
                reason="Ready to generate tests"
            ),
            PolicyRule(
                condition="tests_generated_not_executed",
                tool_name="execute_tests",
                priority=ToolPriority.HIGH,
                reason="Must validate generated tests"
            ),
            PolicyRule(
                condition="tests_failed",
                tool_name="generate_tests",
                priority=ToolPriority.MEDIUM,
                reason="Refine failed tests"
            ),
            PolicyRule(
                condition="low_coverage",
                tool_name="generate_tests",
                priority=ToolPriority.HIGH,
                reason="Improve coverage"
            ),
        ]
    
    def select_tool(
        self,
        state: Dict,
        available_tools: List[str]
    ) -> tuple[str, str]:
        """
        Select best tool for current state.
        
        Args:
            state: Current agent state
            available_tools: List of available tool names
            
        Returns:
            Tuple of (tool_name, reason)
            
        Example:
            >>> policy = ActorPolicy()
            >>> state = {"tests_generated": False, "context_retrieved": True}
            >>> tool, reason = policy.select_tool(state, ["generate_tests", "execute_tests"])
            >>> print(f"Selected: {tool}, Reason: {reason}")
        """
        # Evaluate rules
        applicable_rules = self._evaluate_rules(state)
        
        if applicable_rules:
            # Filter by available tools
            for rule in applicable_rules:
                if rule.tool_name in available_tools:
                    console.print(f"  → Policy: {rule.reason}")
                    return rule.tool_name, rule.reason
        
        # Fall back to heuristic selection
        return self._heuristic_selection(state, available_tools)
    
    def _evaluate_rules(self, state: Dict) -> List[PolicyRule]:
        """Evaluate which rules apply to current state."""
        applicable = []
        
        for rule in self.rules:
            if self._check_condition(rule.condition, state):
                applicable.append(rule)
        
        # Sort by priority
        priority_order = {
            ToolPriority.CRITICAL: 0,
            ToolPriority.HIGH: 1,
            ToolPriority.MEDIUM: 2,
            ToolPriority.LOW: 3
        }
        applicable.sort(key=lambda r: priority_order[r.priority])
        
        return applicable
    
    def _check_condition(self, condition: str, state: Dict) -> bool:
        """Check if a condition matches current state."""
        condition_map = {
            "no_code_analyzed": not state.get("code_analyzed", False),
            "code_identified_no_context": (
                state.get("code_analyzed", False) and
                not state.get("context_retrieved", False)
            ),
            "context_available_no_tests": (
                state.get("context_retrieved", False) and
                not state.get("tests_generated", False)
            ),
            "tests_generated_not_executed": (
                state.get("tests_generated", False) and
                not state.get("tests_executed", False)
            ),
            "tests_failed": (
                state.get("tests_executed", False) and
                not state.get("tests_passed", False)
            ),
            "low_coverage": (
                state.get("coverage", 0) < 80
            ),
        }
        
        return condition_map.get(condition, False)
    
    def _heuristic_selection(
        self,
        state: Dict,
        available_tools: List[str]
    ) -> tuple[str, str]:
        """Heuristic-based tool selection."""
        # Score each tool based on metrics and state
        scores = {}
        
        for tool_name in available_tools:
            metrics = self.tool_metrics.get(tool_name, ToolMetrics())
            
            # Base score from reliability
            score = metrics.reliability_score
            
            # Bonus for tools that haven't been used recently
            if tool_name not in [h.get("tool") for h in self.execution_history[-3:]]:
                score += 0.1
            
            scores[tool_name] = score
        
        # Select tool with highest score
        best_tool = max(scores, key=scores.get)
        reason = f"Heuristic selection (score: {scores[best_tool]:.2f})"
        
        return best_tool, reason
    
    def record_execution(
        self,
        tool_name: str,
        success: bool,
        duration: float
    ) -> None:
        """
        Record tool execution outcome.
        
        Args:
            tool_name: Tool that was executed
            success: Whether execution succeeded
            duration: Execution duration in seconds
        """
        # Update metrics
        if tool_name not in self.tool_metrics:
            self.tool_metrics[tool_name] = ToolMetrics()
        
        metrics = self.tool_metrics[tool_name]
        metrics.uses += 1
        
        if success:
            metrics.successes += 1
        else:
            metrics.failures += 1
        
        metrics.last_success = success
        
        # Update average duration
        metrics.avg_duration = (
            (metrics.avg_duration * (metrics.uses - 1) + duration) / metrics.uses
        )
        
        # Record in history
        self.execution_history.append({
            "tool": tool_name,
            "success": success,
            "duration": duration
        })
        
        console.print(
            f"  → Tool '{tool_name}': "
            f"{'✓' if success else '✗'} "
            f"(success rate: {metrics.success_rate:.1%})"
        )
    
    def get_tool_recommendations(
        self,
        state: Dict,
        top_n: int = 3
    ) -> List[tuple[str, float, str]]:
        """
        Get top N tool recommendations with scores.
        
        Args:
            state: Current state
            top_n: Number of recommendations
            
        Returns:
            List of (tool_name, score, reason) tuples
        """
        # Evaluate all rules
        applicable_rules = self._evaluate_rules(state)
        
        recommendations = []
        for rule in applicable_rules[:top_n]:
            metrics = self.tool_metrics.get(rule.tool_name, ToolMetrics())
            score = metrics.reliability_score
            recommendations.append((rule.tool_name, score, rule.reason))
        
        return recommendations
    
    def reset_metrics(self) -> None:
        """Reset all tool metrics."""
        self.tool_metrics.clear()
        self.execution_history.clear()
        console.print("[green]✓[/green] Policy metrics reset")


def create_actor_policy() -> ActorPolicy:
    """Factory function to create an actor policy."""
    return ActorPolicy()


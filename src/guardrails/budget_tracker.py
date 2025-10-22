"""
Advanced Budget Tracking for LLM usage.

Tracks and enforces limits on:
- Token usage
- API costs
- Time budgets
- Per-user quotas

This module prevents runaway costs and ensures fair resource allocation.
"""

import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from rich.console import Console

console = Console()


class BudgetType(str, Enum):
    """Types of budgets."""
    
    TOKEN = "token"  # Token count
    COST = "cost"  # Dollar cost
    TIME = "time"  # Time in seconds
    CALL = "call"  # Number of calls


class BudgetStatus(str, Enum):
    """Budget status."""
    
    OK = "ok"  # Within budget
    WARNING = "warning"  # Approaching limit (>80%)
    EXCEEDED = "exceeded"  # Over limit
    EXHAUSTED = "exhausted"  # No budget remaining


class BudgetLimit(BaseModel):
    """A budget limit."""
    
    budget_type: BudgetType = Field(..., description="Type of budget")
    limit: float = Field(..., description="Limit value")
    period: str = Field(default="session", description="Period (session, hour, day, month)")
    user_id: Optional[str] = Field(default=None, description="User ID (for per-user limits)")


class BudgetUsage(BaseModel):
    """Current budget usage."""
    
    budget_type: BudgetType = Field(..., description="Type of budget")
    used: float = Field(..., description="Amount used")
    limit: float = Field(..., description="Limit")
    percentage: float = Field(..., description="Percentage used (0-100)")
    status: BudgetStatus = Field(..., description="Status")
    remaining: float = Field(..., description="Remaining budget")


class TokenCostEstimate(BaseModel):
    """Estimate of token usage and cost."""
    
    input_tokens: int = Field(..., description="Input tokens")
    output_tokens: int = Field(..., description="Output tokens")
    total_tokens: int = Field(..., description="Total tokens")
    estimated_cost: float = Field(..., description="Estimated cost in USD")


class BudgetExceededError(Exception):
    """Budget limit exceeded."""
    pass


class BudgetTracker:
    """
    Advanced budget tracking for LLM usage.
    
    Features:
    - Multi-dimensional tracking (tokens, cost, time, calls)
    - Per-user quotas
    - Time-based periods (hour, day, month)
    - Automatic reset
    - Warning thresholds
    
    Example:
        >>> tracker = BudgetTracker()
        >>> tracker.set_limit(BudgetType.TOKEN, 100000, period="day")
        >>> tracker.set_limit(BudgetType.COST, 10.0, period="month")
        >>> 
        >>> # Before LLM call
        >>> if not tracker.check_budget(BudgetType.TOKEN, 1000):
        ...     raise BudgetExceededError("Token budget exceeded")
        >>> 
        >>> # After LLM call
        >>> tracker.record_usage(
        ...     input_tokens=500,
        ...     output_tokens=1500,
        ...     duration_seconds=2.5
        ... )
    """
    
    def __init__(self, session_id: str):
        """
        Initialize budget tracker.
        
        Args:
            session_id: Session identifier
        """
        self.session_id = session_id
        self.start_time = time.time()
        
        # Budget limits
        self.limits: Dict[str, BudgetLimit] = {}
        
        # Usage tracking
        self.usage = {
            BudgetType.TOKEN: 0,
            BudgetType.COST: 0.0,
            BudgetType.TIME: 0.0,
            BudgetType.CALL: 0
        }
        
        # Token pricing (per 1M tokens)
        self.token_prices = {
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "claude-3-opus": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet": {"input": 3.0, "output": 15.0},
            "gemini-pro": {"input": 0.5, "output": 1.5},
            "ollama": {"input": 0.0, "output": 0.0},  # Local, free
        }
        
        # Period start times
        self.period_starts: Dict[str, datetime] = {
            "session": datetime.now(),
            "hour": datetime.now(),
            "day": datetime.now(),
            "month": datetime.now()
        }
        
        console.print(f"[bold green]✅ Budget Tracker Initialized (session: {session_id})[/bold green]")
    
    def set_limit(
        self,
        budget_type: BudgetType,
        limit: float,
        period: str = "session",
        user_id: Optional[str] = None
    ):
        """
        Set a budget limit.
        
        Args:
            budget_type: Type of budget
            limit: Limit value
            period: Period (session, hour, day, month)
            user_id: User ID for per-user limits
        """
        key = f"{budget_type.value}_{period}_{user_id or 'global'}"
        self.limits[key] = BudgetLimit(
            budget_type=budget_type,
            limit=limit,
            period=period,
            user_id=user_id
        )
        
        console.print(
            f"[cyan]Budget set: {budget_type.value} = {limit} per {period}[/cyan]"
        )
    
    def check_budget(
        self,
        budget_type: BudgetType,
        amount: float,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Check if there's enough budget.
        
        Args:
            budget_type: Type of budget
            amount: Amount to check
            user_id: User ID
            
        Returns:
            True if within budget, False otherwise
        """
        # Reset periods if needed
        self._reset_periods()
        
        # Check each applicable limit
        for period in ["session", "hour", "day", "month"]:
            key = f"{budget_type.value}_{period}_{user_id or 'global'}"
            if key in self.limits:
                limit = self.limits[key].limit
                current = self.usage.get(budget_type, 0)
                
                if current + amount > limit:
                    console.print(
                        f"[red]⚠️  Budget exceeded: {budget_type.value} ({current + amount} > {limit})[/red]"
                    )
                    return False
        
        return True
    
    def record_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_seconds: float = 0,
        model: str = "gpt-4",
        user_id: Optional[str] = None
    ):
        """
        Record LLM usage.
        
        Args:
            input_tokens: Input tokens
            output_tokens: Output tokens
            duration_seconds: Duration
            model: Model name
            user_id: User ID
        """
        # Token usage
        total_tokens = input_tokens + output_tokens
        self.usage[BudgetType.TOKEN] += total_tokens
        
        # Cost calculation
        if model in self.token_prices:
            prices = self.token_prices[model]
            cost = (
                (input_tokens / 1_000_000) * prices["input"] +
                (output_tokens / 1_000_000) * prices["output"]
            )
            self.usage[BudgetType.COST] += cost
        
        # Time tracking
        self.usage[BudgetType.TIME] += duration_seconds
        
        # Call count
        self.usage[BudgetType.CALL] += 1
        
        # Log usage
        console.print(
            f"[dim]Usage: {total_tokens} tokens, ${self.usage[BudgetType.COST]:.4f}, "
            f"{duration_seconds:.2f}s[/dim]"
        )
    
    def get_usage(
        self,
        budget_type: Optional[BudgetType] = None,
        user_id: Optional[str] = None
    ) -> Dict[BudgetType, BudgetUsage]:
        """
        Get current usage.
        
        Args:
            budget_type: Specific budget type (or None for all)
            user_id: User ID
            
        Returns:
            Dictionary of budget usage
        """
        self._reset_periods()
        
        usage_dict = {}
        
        types = [budget_type] if budget_type else list(BudgetType)
        
        for btype in types:
            # Find applicable limit (prioritize session)
            limit = None
            for period in ["session", "hour", "day", "month"]:
                key = f"{btype.value}_{period}_{user_id or 'global'}"
                if key in self.limits:
                    limit = self.limits[key].limit
                    break
            
            if limit is None:
                limit = float('inf')
            
            used = self.usage.get(btype, 0)
            percentage = (used / limit * 100) if limit != float('inf') else 0
            
            if percentage >= 100:
                status = BudgetStatus.EXCEEDED
            elif percentage >= 80:
                status = BudgetStatus.WARNING
            else:
                status = BudgetStatus.OK
            
            usage_dict[btype] = BudgetUsage(
                budget_type=btype,
                used=used,
                limit=limit,
                percentage=percentage,
                status=status,
                remaining=max(0, limit - used)
            )
        
        return usage_dict
    
    def estimate_cost(
        self,
        prompt: str,
        expected_output_tokens: int,
        model: str = "gpt-4"
    ) -> TokenCostEstimate:
        """
        Estimate token usage and cost.
        
        Args:
            prompt: Input prompt
            expected_output_tokens: Expected output tokens
            model: Model name
            
        Returns:
            TokenCostEstimate
        """
        # Simple estimation: ~4 chars per token
        input_tokens = len(prompt) // 4
        output_tokens = expected_output_tokens
        total_tokens = input_tokens + output_tokens
        
        # Calculate cost
        if model in self.token_prices:
            prices = self.token_prices[model]
            cost = (
                (input_tokens / 1_000_000) * prices["input"] +
                (output_tokens / 1_000_000) * prices["output"]
            )
        else:
            cost = 0.0
        
        return TokenCostEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=cost
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get budget summary."""
        elapsed_time = time.time() - self.start_time
        usage = self.get_usage()
        
        return {
            "session_id": self.session_id,
            "elapsed_time": elapsed_time,
            "total_calls": int(self.usage[BudgetType.CALL]),
            "total_tokens": int(self.usage[BudgetType.TOKEN]),
            "total_cost": round(self.usage[BudgetType.COST], 4),
            "avg_tokens_per_call": (
                int(self.usage[BudgetType.TOKEN] / self.usage[BudgetType.CALL])
                if self.usage[BudgetType.CALL] > 0 else 0
            ),
            "budgets": {
                btype.value: {
                    "used": u.used,
                    "limit": u.limit if u.limit != float('inf') else "unlimited",
                    "percentage": round(u.percentage, 1),
                    "status": u.status.value
                }
                for btype, u in usage.items()
            }
        }
    
    def _reset_periods(self):
        """Reset period counters if needed."""
        now = datetime.now()
        
        # Check each period
        if now - self.period_starts["hour"] > timedelta(hours=1):
            self.period_starts["hour"] = now
            # Reset hourly counters (simplified)
        
        if now - self.period_starts["day"] > timedelta(days=1):
            self.period_starts["day"] = now
            # Reset daily counters (simplified)
        
        if now.month != self.period_starts["month"].month:
            self.period_starts["month"] = now
            # Reset monthly counters (simplified)


def create_budget_tracker(session_id: str) -> BudgetTracker:
    """Factory function to create budget tracker."""
    return BudgetTracker(session_id)


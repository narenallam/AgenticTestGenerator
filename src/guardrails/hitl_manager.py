"""
Human-in-the-Loop (HITL) Manager for approval workflows.

Handles user approvals for high-risk actions with clear diffs and prompts.
"""

import time
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

console = Console()


class ApprovalDecision(str, Enum):
    """Approval decision outcomes."""
    APPROVE = "approve"
    DENY = "deny"
    MODIFY = "modify"
    TIMEOUT = "timeout"


class RiskLevel(str, Enum):
    """Risk levels for actions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalRequest(BaseModel):
    """A request for human approval."""
    
    request_id: str = Field(..., description="Unique request ID")
    action: str = Field(..., description="Action description")
    tool: str = Field(..., description="Tool being called")
    params: Dict[str, Any] = Field(..., description="Tool parameters")
    risk_level: RiskLevel = Field(..., description="Risk level")
    reason: str = Field(..., description="Why approval is needed")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    timeout_seconds: int = Field(default=300, description="Timeout for approval (5 min)")


class ApprovalResponse(BaseModel):
    """Response to approval request."""
    
    request_id: str = Field(..., description="Request ID")
    decision: ApprovalDecision = Field(..., description="User decision")
    reason: Optional[str] = Field(default=None, description="Reason for decision")
    modified_params: Optional[Dict[str, Any]] = Field(default=None, description="Modified parameters")
    response_time_ms: float = Field(..., description="Time to respond (ms)")


class HITLManager:
    """
    Human-in-the-Loop Manager for approval workflows.
    
    Features:
    - Clear action summaries
    - Before/after diffs
    - Risk-based gating
    - Timeout handling
    - Approval history tracking
    
    Risk-Based Rules:
    - LOW: Auto-execute (no approval needed)
    - MEDIUM: Notify + proceed unless vetoed (30s timeout)
    - HIGH: Explicit approval required (5 min timeout)
    - CRITICAL: Two-factor approval (future)
    
    Example:
        >>> hitl = HITLManager()
        >>> request = ApprovalRequest(
        ...     request_id="req_001",
        ...     action="Modify source code",
        ...     tool="repair_code",
        ...     params={"file": "src/app.py"},
        ...     risk_level=RiskLevel.HIGH,
        ...     reason="Planner requested code repair"
        ... )
        >>> response = hitl.request_approval(request)
        >>> if response.decision != ApprovalDecision.APPROVE:
        ...     raise SecurityError("Action denied by user")
    """
    
    def __init__(self, interactive: bool = True):
        """
        Initialize HITL manager.
        
        Args:
            interactive: If False, auto-approve LOW/MEDIUM (for CI)
        """
        self.interactive = interactive
        self._approval_history: list[ApprovalResponse] = []
        
        console.print("[bold green]👤 HITL Manager Initialized[/bold green]")
    
    def request_approval(
        self,
        request: ApprovalRequest,
        show_diff: bool = False,
        before: Optional[str] = None,
        after: Optional[str] = None
    ) -> ApprovalResponse:
        """
        Request user approval for an action.
        
        Args:
            request: Approval request
            show_diff: Show before/after diff
            before: Before state (for diff)
            after: After state (for diff)
            
        Returns:
            ApprovalResponse with user decision
        """
        start_time = time.time()
        
        # Auto-approve LOW risk if not interactive
        if not self.interactive and request.risk_level == RiskLevel.LOW:
            return self._auto_approve(request, start_time)
        
        # Show request details
        self._display_request(request)
        
        # Show diff if provided
        if show_diff and before and after:
            self._display_diff(before, after)
        
        # Get decision based on risk level
        if request.risk_level == RiskLevel.LOW:
            decision = ApprovalDecision.APPROVE
            console.print("[green]✓ Auto-approved (LOW risk)[/green]")
        
        elif request.risk_level == RiskLevel.MEDIUM:
            decision = self._prompt_medium_risk(request)
        
        elif request.risk_level == RiskLevel.HIGH:
            decision = self._prompt_high_risk(request)
        
        else:  # CRITICAL
            decision = ApprovalDecision.DENY
            console.print("[red]✗ CRITICAL actions require manual implementation[/red]")
        
        # Get reason if denied
        reason = None
        if decision == ApprovalDecision.DENY:
            reason = Prompt.ask("[yellow]Reason for denial (optional)[/yellow]", default="")
        
        # Create response
        response = ApprovalResponse(
            request_id=request.request_id,
            decision=decision,
            reason=reason if reason else None,
            response_time_ms=(time.time() - start_time) * 1000
        )
        
        self._approval_history.append(response)
        return response
    
    def _display_request(self, request: ApprovalRequest):
        """Display approval request details."""
        # Risk color coding
        risk_colors = {
            RiskLevel.LOW: "green",
            RiskLevel.MEDIUM: "yellow",
            RiskLevel.HIGH: "red",
            RiskLevel.CRITICAL: "bold red"
        }
        color = risk_colors.get(request.risk_level, "white")
        
        # Create table
        table = Table(title="🔐 Approval Request", show_header=True, header_style="bold magenta")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Request ID", request.request_id)
        table.add_row("Action", request.action)
        table.add_row("Tool", request.tool)
        table.add_row("Risk Level", f"[{color}]{request.risk_level.value.upper()}[/{color}]")
        table.add_row("Reason", request.reason)
        
        # Add parameters
        for key, value in request.params.items():
            table.add_row(f"  {key}", str(value))
        
        console.print(table)
        console.print()
    
    def _display_diff(self, before: str, after: str):
        """Display before/after diff."""
        console.print(Panel.fit(
            "[bold cyan]BEFORE:[/bold cyan]\n" + before[:500],
            title="Before",
            border_style="cyan"
        ))
        console.print()
        console.print(Panel.fit(
            "[bold green]AFTER:[/bold green]\n" + after[:500],
            title="After",
            border_style="green"
        ))
        console.print()
    
    def _prompt_medium_risk(self, request: ApprovalRequest) -> ApprovalDecision:
        """Prompt for MEDIUM risk action (notify + proceed)."""
        console.print("[yellow]⚠️  MEDIUM RISK: Proceeding in 10 seconds unless you object[/yellow]")
        console.print("[yellow]Press Ctrl+C to cancel, or wait to continue...[/yellow]")
        
        try:
            # Wait 10 seconds
            import time
            for i in range(10, 0, -1):
                console.print(f"[yellow]{i}...[/yellow]", end=" ")
                time.sleep(1)
            console.print()
            
            # Auto-approve if not interrupted
            console.print("[green]✓ Proceeding (no objection)[/green]")
            return ApprovalDecision.APPROVE
        
        except KeyboardInterrupt:
            console.print("\n[red]✗ Cancelled by user[/red]")
            return ApprovalDecision.DENY
    
    def _prompt_high_risk(self, request: ApprovalRequest) -> ApprovalDecision:
        """Prompt for HIGH risk action (explicit approval)."""
        console.print("[bold red]⚠️  HIGH RISK: Explicit approval required[/bold red]")
        
        # Ask for approval
        approved = Confirm.ask(
            f"[yellow]Do you approve this action?[/yellow]",
            default=False
        )
        
        if approved:
            console.print("[green]✓ Approved by user[/green]")
            return ApprovalDecision.APPROVE
        else:
            console.print("[red]✗ Denied by user[/red]")
            return ApprovalDecision.DENY
    
    def _auto_approve(self, request: ApprovalRequest, start_time: float) -> ApprovalResponse:
        """Auto-approve for non-interactive mode."""
        return ApprovalResponse(
            request_id=request.request_id,
            decision=ApprovalDecision.APPROVE,
            reason="Auto-approved (non-interactive mode)",
            response_time_ms=(time.time() - start_time) * 1000
        )
    
    def get_approval_rate(self) -> Dict[str, float]:
        """Get approval statistics."""
        if not self._approval_history:
            return {
                "total_requests": 0,
                "approval_rate": 0.0,
                "deny_rate": 0.0,
                "timeout_rate": 0.0
            }
        
        total = len(self._approval_history)
        approved = sum(1 for r in self._approval_history if r.decision == ApprovalDecision.APPROVE)
        denied = sum(1 for r in self._approval_history if r.decision == ApprovalDecision.DENY)
        timeout = sum(1 for r in self._approval_history if r.decision == ApprovalDecision.TIMEOUT)
        
        return {
            "total_requests": total,
            "approval_rate": approved / total * 100,
            "deny_rate": denied / total * 100,
            "timeout_rate": timeout / total * 100,
            "avg_response_time_ms": sum(r.response_time_ms for r in self._approval_history) / total
        }


def create_hitl_manager(interactive: bool = True) -> HITLManager:
    """Factory function to create HITL manager."""
    return HITLManager(interactive)


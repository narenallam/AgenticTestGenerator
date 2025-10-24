"""
Enhanced Console Tracker for Detailed Component Visibility.

Provides real-time, detailed console output showing all architectural
components in action during test generation.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text

console = Console()


class ComponentTracker:
    """
    Tracks and displays detailed status for all architectural components.
    
    Components tracked:
    - Orchestrator (coordination)
    - LLM Provider (generation)
    - Tools (code retrieval, git, sandbox)
    - Guardrails (input/output validation, secrets, PII)
    - RAG/Embeddings (code search)
    - Critic (quality review)
    - Evaluators (coverage, quality, safety)
    - State Management
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize component tracker."""
        self.verbose = verbose
        self.events: list = []
        self.component_status: Dict[str, str] = {}
        self.metrics: Dict[str, Any] = {
            "tool_calls": 0,
            "guardrail_checks": 0,
            "llm_calls": 0,
            "rag_queries": 0,
            "critic_reviews": 0,
            "eval_runs": 0
        }
    
    def section_header(self, component: str, action: str, emoji: str = "🔧"):
        """Display a section header for a component."""
        if not self.verbose:
            return
        
        header = f"{emoji} [{component.upper()}] {action}"
        console.print()
        console.print(Panel(
            header,
            style="bold cyan",
            border_style="cyan"
        ))
    
    def component_start(self, component: str, action: str, details: Optional[str] = None):
        """Log component start."""
        if not self.verbose:
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        emoji_map = {
            "orchestrator": "🎭",
            "llm": "🤖",
            "tool": "🔧",
            "guardrail": "🛡️",
            "rag": "🔍",
            "critic": "👨‍⚖️",
            "evaluator": "📊",
            "sandbox": "🐳",
            "git": "📚",
            "indexer": "📇",
            "tracker": "📋"
        }
        
        emoji = emoji_map.get(component.lower(), "⚙️")
        
        console.print(
            f"[dim]{timestamp}[/dim] {emoji} [bold cyan]{component}[/bold cyan]: "
            f"[yellow]→[/yellow] {action}"
        )
        
        if details:
            console.print(f"  [dim]{details}[/dim]")
        
        self.events.append({
            "timestamp": timestamp,
            "component": component,
            "action": action,
            "type": "start"
        })
    
    def component_progress(self, component: str, message: str, status: str = "info"):
        """Log component progress."""
        if not self.verbose:
            return
        
        status_styles = {
            "info": "blue",
            "success": "green",
            "warning": "yellow",
            "error": "red"
        }
        
        style = status_styles.get(status, "blue")
        symbol_map = {
            "info": "ℹ",
            "success": "✓",
            "warning": "⚠",
            "error": "✗"
        }
        symbol = symbol_map.get(status, "•")
        
        console.print(f"  [{style}]{symbol}[/{style}] {message}")
    
    def component_complete(self, component: str, result: str, duration: Optional[float] = None):
        """Log component completion."""
        if not self.verbose:
            return
        
        duration_str = f" ({duration:.2f}s)" if duration else ""
        console.print(
            f"  [green]✓[/green] [bold]{component}[/bold] complete: "
            f"{result}{duration_str}"
        )
    
    def tool_call(self, tool_name: str, args: Dict[str, Any], result: Optional[str] = None):
        """Log tool invocation."""
        if not self.verbose:
            return
        
        self.metrics["tool_calls"] += 1
        
        console.print(
            f"  [bold magenta]🔧 TOOL CALL[/bold magenta]: "
            f"[cyan]{tool_name}[/cyan]"
        )
        
        if args:
            for key, value in args.items():
                # Truncate long values
                val_str = str(value)
                if len(val_str) > 60:
                    val_str = val_str[:57] + "..."
                console.print(f"    • {key}: [dim]{val_str}[/dim]")
        
        if result:
            console.print(f"    [green]→ Result: {result}[/green]")
    
    def guardrail_check(self, guardrail: str, passed: bool, details: Optional[str] = None):
        """Log guardrail check."""
        if not self.verbose:
            return
        
        self.metrics["guardrail_checks"] += 1
        
        status = "[green]✓ PASSED[/green]" if passed else "[red]✗ FAILED[/red]"
        console.print(
            f"  [bold yellow]🛡️  GUARDRAIL[/bold yellow]: "
            f"{guardrail} {status}"
        )
        
        if details:
            console.print(f"    [dim]{details}[/dim]")
    
    def llm_call(self, provider: str, model: str, tokens: Optional[int] = None):
        """Log LLM call."""
        if not self.verbose:
            return
        
        self.metrics["llm_calls"] += 1
        
        token_str = f" ({tokens} tokens)" if tokens else ""
        console.print(
            f"  [bold blue]🤖 LLM CALL[/bold blue]: "
            f"{provider}/{model}{token_str}"
        )
    
    def rag_query(self, query: str, results_count: int, duration: Optional[float] = None):
        """Log RAG query."""
        if not self.verbose:
            return
        
        self.metrics["rag_queries"] += 1
        
        duration_str = f" in {duration:.2f}s" if duration else ""
        query_preview = query[:50] + "..." if len(query) > 50 else query
        
        console.print(
            f"  [bold purple]🔍 RAG QUERY[/bold purple]: "
            f"'{query_preview}' → {results_count} results{duration_str}"
        )
    
    def critic_review(self, aspect: str, score: float, feedback: Optional[str] = None):
        """Log critic review."""
        if not self.verbose:
            return
        
        self.metrics["critic_reviews"] += 1
        
        score_color = "green" if score >= 0.8 else "yellow" if score >= 0.6 else "red"
        console.print(
            f"  [bold cyan]👨‍⚖️  CRITIC REVIEW[/bold cyan]: "
            f"{aspect} → [{score_color}]{score:.1%}[/{score_color}]"
        )
        
        if feedback:
            console.print(f"    [dim]{feedback}[/dim]")
    
    def evaluator_run(self, eval_type: str, result: Dict[str, Any]):
        """Log evaluator run."""
        if not self.verbose:
            return
        
        self.metrics["eval_runs"] += 1
        
        console.print(
            f"  [bold green]📊 EVALUATOR[/bold green]: {eval_type}"
        )
        
        for key, value in result.items():
            if isinstance(value, float):
                console.print(f"    • {key}: [cyan]{value:.2f}[/cyan]")
            else:
                console.print(f"    • {key}: [cyan]{value}[/cyan]")
    
    def state_update(self, state_key: str, value: Any):
        """Log state update."""
        if not self.verbose:
            return
        
        val_str = str(value)
        if len(val_str) > 60:
            val_str = val_str[:57] + "..."
        
        console.print(
            f"  [bold yellow]📝 STATE UPDATE[/bold yellow]: "
            f"{state_key} = [dim]{val_str}[/dim]"
        )
    
    def show_metrics_summary(self):
        """Display summary of all component activity."""
        if not self.verbose:
            return
        
        console.print()
        console.print(Panel(
            "[bold cyan]🎯 Component Activity Summary[/bold cyan]",
            border_style="cyan"
        ))
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan")
        table.add_column("Activity Count", justify="right", style="yellow")
        
        table.add_row("🤖 LLM Calls", str(self.metrics["llm_calls"]))
        table.add_row("🔧 Tool Calls", str(self.metrics["tool_calls"]))
        table.add_row("🛡️  Guardrail Checks", str(self.metrics["guardrail_checks"]))
        table.add_row("🔍 RAG Queries", str(self.metrics["rag_queries"]))
        table.add_row("👨‍⚖️  Critic Reviews", str(self.metrics["critic_reviews"]))
        table.add_row("📊 Evaluator Runs", str(self.metrics["eval_runs"]))
        
        console.print(table)
        console.print()
    
    @contextmanager
    def component_section(self, component: str, action: str):
        """Context manager for component sections."""
        start_time = datetime.now()
        
        self.section_header(component, action)
        self.component_start(component, action)
        
        try:
            yield self
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            self.component_complete(component, "finished", duration)


# Global tracker instance
_tracker = None


def get_tracker(verbose: bool = True) -> ComponentTracker:
    """Get or create global tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = ComponentTracker(verbose=verbose)
    return _tracker


def reset_tracker():
    """Reset global tracker."""
    global _tracker
    _tracker = None


"""
Console monitor for real-time observability.

Displays live metrics, logs, and system status.
"""

import sys
import time
from datetime import datetime
from typing import Dict

from .config import get_config
from .metrics import get_registry


class ConsoleMonitor:
    """Real-time console monitor."""
    
    def __init__(self, refresh_interval: int = 5):
        """
        Initialize monitor.
        
        Args:
            refresh_interval: Refresh interval in seconds
        """
        self.refresh_interval = refresh_interval
        self.registry = get_registry()
        self.config = get_config()
    
    def run(self):
        """Run the monitor."""
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║       Agentic Test Generator - Observability Monitor            ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print()
        print(f"Refresh Interval: {self.refresh_interval}s | Press Ctrl+C to exit")
        print()
        
        try:
            while True:
                self._clear_screen()
                self._display_dashboard()
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print("\n\nMonitor stopped.")
    
    def _clear_screen(self):
        """Clear the console screen."""
        print("\033[H\033[J", end="")
    
    def _display_dashboard(self):
        """Display the dashboard."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("┌────────────────────────────────────────────────────────────────┐")
        print(f"│ Observability Dashboard - {timestamp}                     │")
        print("├────────────────────────────────────────────────────────────────┤")
        
        # Get metrics summary
        summary = self._get_metrics_summary()
        
        # Test Generation
        print("│ 📊 Test Generation                                             │")
        print(f"│   Requests: {summary.get('test_gen_requests', 0):<10} "
              f"Success Rate: {summary.get('test_gen_success_rate', 0.0):.1%}              │")
        print(f"│   Coverage: {summary.get('coverage', 0.0):.1%}       "
              f"Pass Rate: {summary.get('pass_rate', 0.0):.1%}                 │")
        print("│                                                                │")
        
        # LLM Stats
        print("│ 🤖 LLM Performance                                             │")
        print(f"│   Calls: {summary.get('llm_calls', 0):<13} "
              f"Avg Latency: {summary.get('llm_avg_latency', 0.0):.2f}s        │")
        print(f"│   Tokens: {summary.get('llm_tokens', 0):<12} "
              f"Cost: ${summary.get('llm_cost', 0.0):.2f}                 │")
        print("│                                                                │")
        
        # Agents
        print("│ 🤖 Agent Activity                                              │")
        print(f"│   Planner: {summary.get('planner_iterations', 0):<10} "
              f"Coder: {summary.get('coder_iterations', 0):<10}               │")
        print(f"│   Critic: {summary.get('critic_iterations', 0):<11} "
              f"                                          │")
        print("│                                                                │")
        
        # Guardrails
        print("│ 🛡️  Guardrails                                                 │")
        print(f"│   Checks: {summary.get('guardrails_checks', 0):<11} "
              f"Violations: {summary.get('guardrails_violations', 0):<10}           │")
        print(f"│   Blocks: {summary.get('guardrails_blocks', 0):<11} "
              f"                                      │")
        print("│                                                                │")
        
        # System
        print("│ 💻 System                                                      │")
        print(f"│   Status: {'🟢 Healthy' if summary.get('errors', 0) == 0 else '🔴 Errors'}                                           │")
        print("└────────────────────────────────────────────────────────────────┘")
    
    def _get_metrics_summary(self) -> Dict:
        """Get metrics summary."""
        summary = {}
        
        # Extract metrics from registry
        for name, metric in self.registry.metrics.items():
            if name == "test_generation_calls_total":
                values = metric.get_all()
                summary['test_gen_requests'] = sum(values.values())
            elif name == "test_coverage_ratio":
                values = metric.get_all()
                if values:
                    summary['coverage'] = list(values.values())[0]
            elif name == "test_pass_rate_ratio":
                values = metric.get_all()
                if values:
                    summary['pass_rate'] = list(values.values())[0]
            elif name == "llm_calls_total":
                values = metric.get_all()
                summary['llm_calls'] = sum(values.values())
            elif name == "llm_tokens_total":
                values = metric.get_all()
                summary['llm_tokens'] = sum(values.values())
            elif name == "agent_iterations_total":
                values = metric.get_all()
                for label_tuple, value in values.items():
                    labels = dict(label_tuple)
                    agent = labels.get('agent', 'unknown')
                    summary[f'{agent}_iterations'] = value
            elif name == "guardrails_checks_total":
                values = metric.get_all()
                summary['guardrails_checks'] = sum(values.values())
            elif name == "guardrails_violations_total":
                values = metric.get_all()
                summary['guardrails_violations'] = sum(values.values())
            elif name == "guardrails_blocks_total":
                values = metric.get_all()
                summary['guardrails_blocks'] = sum(values.values())
        
        return summary


def run_monitor():
    """CLI entry point for monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Observability monitor")
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Refresh interval in seconds"
    )
    
    args = parser.parse_args()
    
    monitor = ConsoleMonitor(refresh_interval=args.interval)
    monitor.run()


if __name__ == "__main__":
    run_monitor()


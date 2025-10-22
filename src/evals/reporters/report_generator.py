"""
Report generation for evaluation results.

Generates:
- Console reports (terminal output)
- Markdown reports (for CI/CD)
- JSON reports (for programmatic access)
- HTML reports (for dashboards)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..base import BaseReporter, EvalResult, EvalSuite, QualityLevel


# ═══════════════════════════════════════════════════════════════════════════
# Console Reporter
# ═══════════════════════════════════════════════════════════════════════════


class ConsoleReporter(BaseReporter):
    """Generate colorful console reports."""
    
    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "magenta": "\033[95m",
    }
    
    def generate_report(self, suite: EvalSuite) -> str:
        """Generate console report."""
        lines = []
        
        # Header
        lines.append(self._format_header(suite))
        lines.append("")
        
        # Summary
        lines.append(self._format_summary(suite))
        lines.append("")
        
        # Individual results
        lines.append(self._format_results(suite))
        lines.append("")
        
        # Recommendations
        lines.append(self._format_recommendations(suite))
        
        return "\n".join(lines)
    
    def save_report(self, suite: EvalSuite, output_path: Path) -> None:
        """Save report to file (without color codes)."""
        report = self.generate_report(suite)
        
        # Remove color codes for file
        for code in self.COLORS.values():
            report = report.replace(code, "")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
    
    def _format_header(self, suite: EvalSuite) -> str:
        """Format header section."""
        c = self.COLORS
        
        header = f"""
{c['bold']}{c['cyan']}{'═' * 80}{c['reset']}
{c['bold']}{c['cyan']}║{' ' * 78}║{c['reset']}
{c['bold']}{c['cyan']}║{c['reset']}  {c['bold']}Evaluation Report: {suite.name}{' ' * (57 - len(suite.name))}║{c['reset']}
{c['bold']}{c['cyan']}║{' ' * 78}║{c['reset']}
{c['bold']}{c['cyan']}{'═' * 80}{c['reset']}
"""
        return header.strip()
    
    def _format_summary(self, suite: EvalSuite) -> str:
        """Format summary section."""
        c = self.COLORS
        
        # Quality level color
        if suite.quality_level == QualityLevel.EXCELLENT:
            level_color = c['green']
        elif suite.quality_level == QualityLevel.GOOD:
            level_color = c['cyan']
        elif suite.quality_level == QualityLevel.FAIR:
            level_color = c['yellow']
        else:
            level_color = c['red']
        
        score_display = f"{(suite.overall_score or 0.0) * 100:.1f}%"
        
        summary = f"""
{c['bold']}📊 Summary{c['reset']}
{'─' * 80}

  {c['bold']}Overall Score:{c['reset']}     {level_color}{score_display}{c['reset']} ({level_color}{suite.quality_level.value if suite.quality_level else 'N/A'}{c['reset']})
  {c['bold']}Total Evals:{c['reset']}      {suite.total_evals}
  {c['bold']}Passed:{c['reset']}          {c['green']}{suite.passed_evals}{c['reset']}
  {c['bold']}Failed:{c['reset']}          {c['red']}{suite.failed_evals}{c['reset']}
  {c['bold']}Duration:{c['reset']}        {suite.duration_seconds:.2f}s
  {c['bold']}Started:{c['reset']}         {suite.started_at.strftime('%Y-%m-%d %H:%M:%S')}
"""
        return summary.strip()
    
    def _format_results(self, suite: EvalSuite) -> str:
        """Format individual results."""
        c = self.COLORS
        
        lines = [f"{c['bold']}📋 Detailed Results{c['reset']}", "─" * 80, ""]
        
        for result in suite.eval_results:
            # Status indicator
            if result.score and result.score >= 0.90:
                indicator = f"{c['green']}✅{c['reset']}"
            elif result.score and result.score >= 0.70:
                indicator = f"{c['yellow']}⚠️{c['reset']}"
            else:
                indicator = f"{c['red']}❌{c['reset']}"
            
            score_str = f"{(result.score or 0.0) * 100:.1f}%" if result.score else "N/A"
            
            lines.append(f"{indicator} {c['bold']}{result.eval_name}{c['reset']}: {score_str}")
            
            # Metrics
            if result.metrics:
                for name, metric in result.metrics.items():
                    value_str = f"{metric.value * 100:.1f}%" if metric.unit == "percentage" or metric.unit == "ratio" else f"{metric.value:.2f}"
                    
                    if metric.passed:
                        status = f"{c['green']}✓{c['reset']}"
                    else:
                        status = f"{c['red']}✗{c['reset']}"
                    
                    lines.append(f"   {status} {name}: {value_str}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_recommendations(self, suite: EvalSuite) -> str:
        """Format recommendations."""
        c = self.COLORS
        
        # Collect all recommendations
        all_recommendations = []
        for result in suite.eval_results:
            all_recommendations.extend(result.recommendations)
        
        if not all_recommendations:
            return f"{c['bold']}{c['green']}✅ No recommendations - All evaluations passed!{c['reset']}"
        
        lines = [f"{c['bold']}💡 Recommendations{c['reset']}", "─" * 80, ""]
        
        for rec in all_recommendations:
            lines.append(f"  • {rec}")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Markdown Reporter
# ═══════════════════════════════════════════════════════════════════════════


class MarkdownReporter(BaseReporter):
    """Generate Markdown reports for CI/CD."""
    
    def generate_report(self, suite: EvalSuite) -> str:
        """Generate Markdown report."""
        lines = []
        
        # Header
        lines.append(f"# Evaluation Report: {suite.name}\n")
        lines.append(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
        
        # Summary
        lines.append("## 📊 Summary\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Overall Score | **{(suite.overall_score or 0.0) * 100:.1f}%** |")
        lines.append(f"| Quality Level | {self._quality_badge(suite.quality_level)} |")
        lines.append(f"| Total Evaluations | {suite.total_evals} |")
        lines.append(f"| Passed | ✅ {suite.passed_evals} |")
        lines.append(f"| Failed | ❌ {suite.failed_evals} |")
        lines.append(f"| Duration | {suite.duration_seconds:.2f}s |")
        lines.append("")
        
        # Detailed results
        lines.append("## 📋 Detailed Results\n")
        
        for result in suite.eval_results:
            status_badge = self._status_badge(result.score)
            lines.append(f"### {status_badge} {result.eval_name}\n")
            lines.append(f"**Score:** {(result.score or 0.0) * 100:.1f}%\n")
            
            if result.metrics:
                lines.append("| Metric | Value | Status |")
                lines.append("|--------|-------|--------|")
                
                for name, metric in result.metrics.items():
                    value_str = f"{metric.value * 100:.1f}%" if metric.unit in ["percentage", "ratio"] else f"{metric.value:.2f}"
                    status = "✅ Pass" if metric.passed else "❌ Fail"
                    lines.append(f"| {name} | {value_str} | {status} |")
                
                lines.append("")
            
            if result.recommendations:
                lines.append("**Recommendations:**\n")
                for rec in result.recommendations:
                    lines.append(f"- {rec}")
                lines.append("")
        
        # Overall recommendations
        all_recommendations = []
        for result in suite.eval_results:
            all_recommendations.extend(result.recommendations)
        
        if all_recommendations:
            lines.append("## 💡 Overall Recommendations\n")
            for rec in set(all_recommendations):  # Deduplicate
                lines.append(f"- {rec}")
        
        return "\n".join(lines)
    
    def save_report(self, suite: EvalSuite, output_path: Path) -> None:
        """Save Markdown report to file."""
        report = self.generate_report(suite)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
    
    def _quality_badge(self, quality: Optional[QualityLevel]) -> str:
        """Generate quality badge."""
        if quality == QualityLevel.EXCELLENT:
            return "![Excellent](https://img.shields.io/badge/Quality-Excellent-brightgreen)"
        elif quality == QualityLevel.GOOD:
            return "![Good](https://img.shields.io/badge/Quality-Good-green)"
        elif quality == QualityLevel.FAIR:
            return "![Fair](https://img.shields.io/badge/Quality-Fair-yellow)"
        else:
            return "![Poor](https://img.shields.io/badge/Quality-Poor-red)"
    
    def _status_badge(self, score: Optional[float]) -> str:
        """Generate status badge."""
        if score and score >= 0.90:
            return "✅"
        elif score and score >= 0.70:
            return "⚠️"
        else:
            return "❌"


# ═══════════════════════════════════════════════════════════════════════════
# JSON Reporter
# ═══════════════════════════════════════════════════════════════════════════


class JSONReporter(BaseReporter):
    """Generate JSON reports for programmatic access."""
    
    def generate_report(self, suite: EvalSuite) -> str:
        """Generate JSON report."""
        data = suite.model_dump()
        return json.dumps(data, indent=2, default=str)
    
    def save_report(self, suite: EvalSuite, output_path: Path) -> None:
        """Save JSON report to file."""
        report = self.generate_report(suite)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Reporter
# ═══════════════════════════════════════════════════════════════════════════


class MultiReporter:
    """Generate reports in multiple formats."""
    
    def __init__(self):
        """Initialize multi-reporter."""
        self.reporters = {
            "console": ConsoleReporter(),
            "markdown": MarkdownReporter(),
            "json": JSONReporter(),
        }
    
    def generate_all(
        self,
        suite: EvalSuite,
        output_dir: Optional[Path] = None
    ) -> Dict[str, str]:
        """
        Generate reports in all formats.
        
        Args:
            suite: Evaluation suite
            output_dir: Directory to save reports (if None, only returns strings)
        
        Returns:
            Dictionary mapping format to report content
        """
        reports = {}
        
        for format_name, reporter in self.reporters.items():
            report = reporter.generate_report(suite)
            reports[format_name] = report
            
            if output_dir:
                # Determine file extension
                ext_map = {
                    "console": "txt",
                    "markdown": "md",
                    "json": "json",
                }
                ext = ext_map.get(format_name, "txt")
                
                output_path = output_dir / f"{suite.suite_id}.{ext}"
                reporter.save_report(suite, output_path)
        
        return reports
    
    def print_console(self, suite: EvalSuite) -> None:
        """Print console report to stdout."""
        console_report = self.reporters["console"].generate_report(suite)
        print(console_report)


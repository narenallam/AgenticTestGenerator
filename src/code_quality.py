"""
Code quality tools integration for linting and formatting.

This module integrates:
- Black (code formatting)
- Flake8 (style linting)
- MyPy (type checking)
- Pylint (code quality)
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from rich.console import Console

console = Console()


class LintIssue(BaseModel):
    """A linting issue."""
    
    line: int = Field(..., description="Line number")
    column: int = Field(default=0, description="Column number")
    code: str = Field(..., description="Error code (e.g., 'E501')")
    message: str = Field(..., description="Error message")
    severity: str = Field(..., description="Severity (error, warning, info)")
    tool: str = Field(..., description="Tool that found the issue")


class QualityReport(BaseModel):
    """Code quality analysis report."""
    
    passed: bool = Field(..., description="All checks passed")
    issues: List[LintIssue] = Field(default_factory=list)
    formatted_code: Optional[str] = Field(default=None, description="Formatted code")
    error_count: int = Field(default=0)
    warning_count: int = Field(default=0)
    info_count: int = Field(default=0)


class CodeFormatter:
    """
    Code formatter using Black.
    
    Ensures generated code follows PEP 8 style guidelines.
    """
    
    def __init__(self, line_length: int = 100):
        self.line_length = line_length
    
    def format_code(self, code: str) -> str:
        """
        Format Python code using Black.
        
        Args:
            code: Python code to format
            
        Returns:
            Formatted code
            
        Example:
            >>> formatter = CodeFormatter()
            >>> code = "def foo(  ):  return 1"
            >>> formatted = formatter.format_code(code)
            >>> print(formatted)  # "def foo(): return 1"
        """
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(code)
                temp_path = f.name
            
            # Run Black
            result = subprocess.run(
                [
                    "black",
                    "--line-length", str(self.line_length),
                    "--quiet",
                    temp_path
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Read formatted code
            formatted_code = Path(temp_path).read_text(encoding='utf-8')
            
            # Clean up
            Path(temp_path).unlink()
            
            if result.returncode == 0 or result.returncode == 1:  # 1 means reformatted
                return formatted_code
            else:
                console.print(f"[yellow]Warning: Black formatting failed: {result.stderr}[/yellow]")
                return code
        
        except Exception as e:
            console.print(f"[yellow]Warning: Code formatting failed: {e}[/yellow]")
            return code
    
    def is_formatted(self, code: str) -> bool:
        """Check if code is already formatted."""
        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(code)
                temp_path = f.name
            
            result = subprocess.run(
                [
                    "black",
                    "--check",
                    "--line-length", str(self.line_length),
                    "--quiet",
                    temp_path
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            Path(temp_path).unlink()
            
            return result.returncode == 0
        
        except Exception:
            return False


class CodeLinter:
    """
    Code linter using Flake8.
    
    Checks code for style violations and common errors.
    """
    
    def __init__(
        self,
        max_line_length: int = 100,
        ignore: Optional[List[str]] = None
    ):
        self.max_line_length = max_line_length
        self.ignore = ignore or ["E203", "W503"]  # Black-compatible ignores
    
    def lint(self, code: str) -> List[LintIssue]:
        """
        Lint Python code using Flake8.
        
        Args:
            code: Python code to lint
            
        Returns:
            List of linting issues
            
        Example:
            >>> linter = CodeLinter()
            >>> issues = linter.lint("import os\\nimport sys\\nimport os")
            >>> print(len(issues))  # Duplicate import
        """
        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(code)
                temp_path = f.name
            
            # Run Flake8
            ignore_str = ",".join(self.ignore)
            result = subprocess.run(
                [
                    "flake8",
                    "--max-line-length", str(self.max_line_length),
                    "--ignore", ignore_str,
                    temp_path
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Clean up
            Path(temp_path).unlink()
            
            # Parse output
            issues = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                
                issue = self._parse_flake8_line(line)
                if issue:
                    issues.append(issue)
            
            return issues
        
        except Exception as e:
            console.print(f"[yellow]Warning: Linting failed: {e}[/yellow]")
            return []
    
    def _parse_flake8_line(self, line: str) -> Optional[LintIssue]:
        """Parse a Flake8 output line."""
        import re
        
        # Format: file.py:line:col: CODE message
        match = re.match(r'.*:(\d+):(\d+):\s+([A-Z]\d+)\s+(.*)', line)
        if match:
            line_num = int(match.group(1))
            col_num = int(match.group(2))
            code = match.group(3)
            message = match.group(4)
            
            # Determine severity
            severity = "error" if code.startswith('E') or code.startswith('F') else "warning"
            
            return LintIssue(
                line=line_num,
                column=col_num,
                code=code,
                message=message,
                severity=severity,
                tool="flake8"
            )
        
        return None


class TypeChecker:
    """
    Type checker using MyPy.
    
    Performs static type checking on Python code.
    """
    
    def __init__(self, strict: bool = False):
        self.strict = strict
    
    def check_types(self, code: str) -> List[LintIssue]:
        """
        Check types using MyPy.
        
        Args:
            code: Python code to check
            
        Returns:
            List of type issues
            
        Example:
            >>> checker = TypeChecker()
            >>> code = "def foo(x: int) -> str:\\n    return x"
            >>> issues = checker.check_types(code)
            >>> print(len(issues))  # Type mismatch
        """
        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(code)
                temp_path = f.name
            
            # Run MyPy
            cmd = ["mypy", "--no-error-summary"]
            if not self.strict:
                cmd.extend(["--ignore-missing-imports", "--allow-untyped-defs"])
            
            cmd.append(temp_path)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Clean up
            Path(temp_path).unlink()
            
            # Parse output
            issues = []
            for line in result.stdout.strip().split('\n'):
                if not line or ": note:" in line:
                    continue
                
                issue = self._parse_mypy_line(line)
                if issue:
                    issues.append(issue)
            
            return issues
        
        except Exception as e:
            console.print(f"[yellow]Warning: Type checking failed: {e}[/yellow]")
            return []
    
    def _parse_mypy_line(self, line: str) -> Optional[LintIssue]:
        """Parse a MyPy output line."""
        import re
        
        # Format: file.py:line: error: message
        match = re.match(r'.*:(\d+):\s+(error|warning):\s+(.*)', line)
        if match:
            line_num = int(match.group(1))
            severity = match.group(2)
            message = match.group(3)
            
            return LintIssue(
                line=line_num,
                column=0,
                code="TYPE",
                message=message,
                severity=severity,
                tool="mypy"
            )
        
        return None


class CodeQualityChecker:
    """
    Comprehensive code quality checker.
    
    Combines formatting, linting, and type checking.
    """
    
    def __init__(
        self,
        line_length: int = 100,
        format_code: bool = True,
        check_types: bool = False  # Optional, slower
    ):
        self.formatter = CodeFormatter(line_length=line_length)
        self.linter = CodeLinter(max_line_length=line_length)
        self.type_checker = TypeChecker(strict=False) if check_types else None
        self.format_code = format_code
    
    def check(self, code: str) -> QualityReport:
        """
        Run comprehensive quality checks.
        
        Args:
            code: Python code to check
            
        Returns:
            QualityReport with all findings
            
        Example:
            >>> checker = CodeQualityChecker()
            >>> report = checker.check("def foo(  ):return 1")
            >>> print(f"Issues: {len(report.issues)}")
            >>> print(report.formatted_code)
        """
        issues = []
        formatted_code = code
        
        # Format code
        if self.format_code:
            formatted_code = self.formatter.format_code(code)
        
        # Lint formatted code
        lint_issues = self.linter.lint(formatted_code)
        issues.extend(lint_issues)
        
        # Type check if enabled
        if self.type_checker:
            type_issues = self.type_checker.check_types(formatted_code)
            issues.extend(type_issues)
        
        # Count issues by severity
        error_count = sum(1 for issue in issues if issue.severity == "error")
        warning_count = sum(1 for issue in issues if issue.severity == "warning")
        info_count = sum(1 for issue in issues if issue.severity == "info")
        
        # Passed if no errors
        passed = error_count == 0
        
        return QualityReport(
            passed=passed,
            issues=issues,
            formatted_code=formatted_code,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count
        )
    
    def fix_and_check(self, code: str, max_iterations: int = 3) -> QualityReport:
        """
        Iteratively format and check code.
        
        Args:
            code: Python code
            max_iterations: Max formatting iterations
            
        Returns:
            Final QualityReport
        """
        current_code = code
        
        for i in range(max_iterations):
            report = self.check(current_code)
            
            if report.passed or report.formatted_code == current_code:
                return report
            
            current_code = report.formatted_code
        
        return report


def create_quality_checker(
    line_length: int = 100,
    format_code: bool = True,
    check_types: bool = False
) -> CodeQualityChecker:
    """
    Factory function to create a code quality checker.
    
    Args:
        line_length: Maximum line length
        format_code: Whether to format code
        check_types: Whether to check types (slower)
        
    Returns:
        Configured CodeQualityChecker
    """
    return CodeQualityChecker(
        line_length=line_length,
        format_code=format_code,
        check_types=check_types
    )


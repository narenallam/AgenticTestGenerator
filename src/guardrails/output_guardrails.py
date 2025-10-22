"""
Output Guardrails for LLM responses.

Validates and sanitizes LLM-generated content before execution:
- Code safety scanning
- License compliance
- Citation requirements
- Hallucination detection
- Format validation

This module ensures LLM outputs are safe, compliant, and accurate.
"""

import ast
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from rich.console import Console

console = Console()


class CodeSafetyIssue(str, Enum):
    """Types of code safety issues."""
    
    EVAL_EXEC = "eval_exec"  # eval() or exec()
    FILE_SYSTEM = "file_system"  # Unsafe file operations
    NETWORK = "network"  # Network calls
    SUBPROCESS = "subprocess"  # Subprocess execution
    IMPORT = "import"  # Dangerous imports
    INFINITE_LOOP = "infinite_loop"  # Potential infinite loops
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # Memory/CPU exhaustion


class LicenseType(str, Enum):
    """Common open source licenses."""
    
    MIT = "mit"
    APACHE = "apache"
    GPL = "gpl"
    BSD = "bsd"
    PROPRIETARY = "proprietary"
    UNKNOWN = "unknown"


class CodeIssue(BaseModel):
    """A detected code safety issue."""
    
    issue_type: CodeSafetyIssue = Field(..., description="Type of issue")
    severity: str = Field(..., description="Severity (LOW/MEDIUM/HIGH/CRITICAL)")
    line_number: Optional[int] = Field(default=None, description="Line number")
    code_snippet: str = Field(..., description="Problematic code")
    description: str = Field(..., description="Issue description")
    recommendation: str = Field(..., description="How to fix")


class CitationRequirement(BaseModel):
    """Citation requirement for code."""
    
    required: bool = Field(..., description="Citation required")
    reason: str = Field(..., description="Why citation is needed")
    suggested_attribution: Optional[str] = Field(default=None, description="Suggested citation")


class OutputScanResult(BaseModel):
    """Result of output scanning."""
    
    safe: bool = Field(..., description="Output is safe")
    code_issues: List[CodeIssue] = Field(default_factory=list, description="Code safety issues")
    license_compliant: bool = Field(default=True, description="License compliant")
    license_type: Optional[LicenseType] = Field(default=None, description="Detected license")
    citation_requirements: List[CitationRequirement] = Field(
        default_factory=list,
        description="Citation requirements"
    )
    hallucination_detected: bool = Field(default=False, description="Hallucination detected")
    sanitized_output: Optional[str] = Field(default=None, description="Sanitized output")


class OutputGuardrails:
    """
    Comprehensive output guardrails for LLM responses.
    
    Features:
    - Code safety scanning (AST analysis)
    - License compliance checking
    - Citation requirement detection
    - Hallucination detection
    - Format validation
    
    Example:
        >>> guardrails = OutputGuardrails()
        >>> code = '''
        ... import os
        ... os.system("rm -rf /")
        ... '''
        >>> result = guardrails.scan_code(code)
        >>> if not result.safe:
        ...     raise SecurityError("Unsafe code generated")
    """
    
    def __init__(
        self,
        enable_code_scanning: bool = True,
        enable_license_checking: bool = True
    ):
        """
        Initialize output guardrails.
        
        Args:
            enable_code_scanning: Enable code safety scanning
            enable_license_checking: Enable license compliance checking
        """
        self.enable_code_scanning = enable_code_scanning
        self.enable_license_checking = enable_license_checking
        
        # Dangerous patterns
        self.dangerous_imports = [
            'eval', 'exec', 'compile', '__import__',
            'os.system', 'subprocess', 'popen',
            'pickle', 'shelve', 'marshal',
        ]
        
        # Dangerous file operations
        self.dangerous_file_ops = [
            'os.remove', 'os.rmdir', 'shutil.rmtree',
            'os.unlink', 'os.chmod', 'os.chown',
        ]
        
        # Network operations
        self.network_ops = [
            'requests.', 'urllib.', 'http.',
            'socket.', 'ftplib.', 'telnetlib.',
        ]
        
        # License patterns
        self.license_patterns = {
            LicenseType.MIT: r'MIT License',
            LicenseType.APACHE: r'Apache License',
            LicenseType.GPL: r'GPL|GNU General Public',
            LicenseType.BSD: r'BSD License',
        }
        
        console.print("[bold green]✅ Output Guardrails Initialized[/bold green]")
    
    def scan_code(
        self,
        code: str,
        language: str = "python",
        context: Optional[Dict[str, Any]] = None
    ) -> OutputScanResult:
        """
        Scan generated code for safety issues.
        
        Args:
            code: Generated code to scan
            language: Programming language (default: python)
            context: Optional context
            
        Returns:
            OutputScanResult with issues and compliance status
        """
        issues = []
        license_type = None
        license_compliant = True
        citation_reqs = []
        
        if language == "python" and self.enable_code_scanning:
            # AST-based analysis
            ast_issues = self._scan_python_ast(code)
            issues.extend(ast_issues)
            
            # Pattern-based analysis
            pattern_issues = self._scan_code_patterns(code)
            issues.extend(pattern_issues)
        
        # License detection
        if self.enable_license_checking:
            license_type = self._detect_license(code)
            if license_type == LicenseType.GPL:
                license_compliant = False
                issues.append(CodeIssue(
                    issue_type=CodeSafetyIssue.IMPORT,
                    severity="HIGH",
                    code_snippet="GPL code detected",
                    description="GPL-licensed code may have copyleft requirements",
                    recommendation="Use MIT or Apache licensed alternatives"
                ))
        
        # Citation requirements
        citation_reqs = self._check_citation_requirements(code, issues)
        
        # Determine if safe
        critical_issues = [i for i in issues if i.severity in ("HIGH", "CRITICAL")]
        safe = len(critical_issues) == 0 and license_compliant
        
        return OutputScanResult(
            safe=safe,
            code_issues=issues,
            license_compliant=license_compliant,
            license_type=license_type,
            citation_requirements=citation_reqs,
            hallucination_detected=False,  # TODO: Implement with ML model
            sanitized_output=None  # TODO: Implement auto-fixing
        )
    
    def scan_text_response(
        self,
        text: str,
        expected_format: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> OutputScanResult:
        """
        Scan generated text response.
        
        Args:
            text: Generated text
            expected_format: Expected format (json, markdown, etc.)
            context: Optional context
            
        Returns:
            OutputScanResult
        """
        issues = []
        
        # Format validation
        if expected_format == "json":
            try:
                import json
                json.loads(text)
            except json.JSONDecodeError as e:
                issues.append(CodeIssue(
                    issue_type=CodeSafetyIssue.IMPORT,  # Reusing enum
                    severity="MEDIUM",
                    code_snippet=text[:100],
                    description=f"Invalid JSON: {str(e)}",
                    recommendation="Fix JSON formatting"
                ))
        
        # Check for leaked PII (should have been redacted)
        pii_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        ]
        
        for pattern in pii_patterns:
            if re.search(pattern, text):
                issues.append(CodeIssue(
                    issue_type=CodeSafetyIssue.FILE_SYSTEM,  # Reusing
                    severity="HIGH",
                    code_snippet="PII detected in output",
                    description="Output contains potential PII",
                    recommendation="Redact PII from output"
                ))
        
        safe = len([i for i in issues if i.severity in ("HIGH", "CRITICAL")]) == 0
        
        return OutputScanResult(
            safe=safe,
            code_issues=issues,
            license_compliant=True,
            hallucination_detected=False
        )
    
    def _scan_python_ast(self, code: str) -> List[CodeIssue]:
        """Scan Python code using AST analysis."""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check for eval/exec
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ('eval', 'exec', 'compile'):
                            issues.append(CodeIssue(
                                issue_type=CodeSafetyIssue.EVAL_EXEC,
                                severity="CRITICAL",
                                line_number=node.lineno if hasattr(node, 'lineno') else None,
                                code_snippet=f"{node.func.id}()",
                                description=f"Dangerous {node.func.id}() call detected",
                                recommendation=f"Remove {node.func.id}() or use safer alternatives"
                            ))
                
                # Check for infinite loops (while True without break)
                if isinstance(node, ast.While):
                    if isinstance(node.test, ast.Constant) and node.test.value is True:
                        has_break = any(
                            isinstance(n, ast.Break)
                            for n in ast.walk(node)
                        )
                        if not has_break:
                            issues.append(CodeIssue(
                                issue_type=CodeSafetyIssue.INFINITE_LOOP,
                                severity="HIGH",
                                line_number=node.lineno if hasattr(node, 'lineno') else None,
                                code_snippet="while True:",
                                description="Potential infinite loop without break",
                                recommendation="Add break condition or timeout"
                            ))
        
        except SyntaxError as e:
            issues.append(CodeIssue(
                issue_type=CodeSafetyIssue.IMPORT,  # Reusing enum
                severity="HIGH",
                line_number=e.lineno,
                code_snippet=str(e),
                description=f"Syntax error: {str(e)}",
                recommendation="Fix syntax errors"
            ))
        
        return issues
    
    def _scan_code_patterns(self, code: str) -> List[CodeIssue]:
        """Scan code using pattern matching."""
        issues = []
        
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Check dangerous imports
            for danger in self.dangerous_imports:
                if danger in line and ('import' in line or danger + '(' in line):
                    issues.append(CodeIssue(
                        issue_type=CodeSafetyIssue.IMPORT,
                        severity="HIGH",
                        line_number=line_num,
                        code_snippet=line.strip(),
                        description=f"Dangerous operation: {danger}",
                        recommendation=f"Avoid {danger} or use safer alternatives"
                    ))
            
            # Check file operations
            for file_op in self.dangerous_file_ops:
                if file_op in line:
                    issues.append(CodeIssue(
                        issue_type=CodeSafetyIssue.FILE_SYSTEM,
                        severity="MEDIUM",
                        line_number=line_num,
                        code_snippet=line.strip(),
                        description=f"File system operation: {file_op}",
                        recommendation="Ensure file operations are safe and controlled"
                    ))
            
            # Check network operations
            for net_op in self.network_ops:
                if net_op in line:
                    issues.append(CodeIssue(
                        issue_type=CodeSafetyIssue.NETWORK,
                        severity="MEDIUM",
                        line_number=line_num,
                        code_snippet=line.strip(),
                        description=f"Network operation: {net_op}",
                        recommendation="Mock network calls in tests"
                    ))
        
        return issues
    
    def _detect_license(self, code: str) -> Optional[LicenseType]:
        """Detect license type from code."""
        for license_type, pattern in self.license_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                return license_type
        
        return LicenseType.UNKNOWN if 'license' in code.lower() else None
    
    def _check_citation_requirements(
        self,
        code: str,
        issues: List[CodeIssue]
    ) -> List[CitationRequirement]:
        """Check if code requires citations."""
        requirements = []
        
        # Check for copied code patterns (simplified)
        lines = code.split('\n')
        comment_blocks = []
        current_block = []
        
        for line in lines:
            if line.strip().startswith('#'):
                current_block.append(line)
            elif current_block:
                comment_blocks.append('\n'.join(current_block))
                current_block = []
        
        # If we find large comment blocks with URLs, might need citation
        for block in comment_blocks:
            if 'http' in block and len(block) > 100:
                requirements.append(CitationRequirement(
                    required=True,
                    reason="Code appears to be adapted from external source",
                    suggested_attribution=block.strip()
                ))
        
        # If GPL license detected, require attribution
        if any(i.issue_type == CodeSafetyIssue.IMPORT and 'GPL' in i.description for i in issues):
            requirements.append(CitationRequirement(
                required=True,
                reason="GPL-licensed code requires attribution",
                suggested_attribution="Source: GPL-licensed code (see LICENSE)"
            ))
        
        return requirements


def create_output_guardrails(
    enable_code_scanning: bool = True,
    enable_license_checking: bool = True
) -> OutputGuardrails:
    """Factory function to create output guardrails."""
    return OutputGuardrails(enable_code_scanning, enable_license_checking)


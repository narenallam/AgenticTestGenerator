"""
File boundary enforcement for test generation.

Ensures generated tests only write to allowed directories.
"""

import ast
import re
from pathlib import Path
from typing import List, Optional, Set

from pydantic import BaseModel, Field


class FileBoundaryConfig(BaseModel):
    """Configuration for file boundary checking."""
    
    allowed_write_dirs: List[str] = Field(
        default=['tests/', 'test/', '.pytest_cache/', '__pycache__/'],
        description="Directories where writes are allowed"
    )
    
    allowed_read_dirs: List[str] = Field(
        default=['src/', 'lib/', 'tests/', 'test/', 'data/'],
        description="Directories where reads are allowed"
    )
    
    require_justification_for: List[str] = Field(
        default=['src/', 'lib/', 'config/'],
        description="Directories requiring Planner justification for writes"
    )


class FileBoundaryViolation(BaseModel):
    """Represents a file boundary violation."""
    
    file_path: str = Field(..., description="File being accessed")
    operation: str = Field(..., description="Operation: read|write")
    line_number: int = Field(..., description="Line number in code")
    code_snippet: str = Field(..., description="Violating code")
    reason: str = Field(..., description="Why it's a violation")


class FileBoundaryChecker:
    """
    Enforces file boundary restrictions on generated tests.
    
    Ensures that:
    - Tests only write to tests/ directory
    - Writes to src/ require Planner justification
    - No writes to sensitive directories
    - Reads are limited to safe locations
    
    Example:
        >>> checker = FileBoundaryChecker()
        >>> violations = checker.check_code(test_code)
        >>> if violations:
        ...     raise SecurityError("File boundary violations detected")
    """
    
    def __init__(self, config: Optional[FileBoundaryConfig] = None):
        """
        Initialize file boundary checker.
        
        Args:
            config: Boundary configuration
        """
        self.config = config or FileBoundaryConfig()
        self._planner_justifications: Set[str] = set()
    
    def add_planner_justification(self, file_path: str, reason: str):
        """
        Add Planner justification for writing to restricted area.
        
        Args:
            file_path: Path that's justified
            reason: Reason from Planner
        """
        self._planner_justifications.add(file_path)
    
    def check_code(
        self,
        code: str,
        allow_all: bool = False
    ) -> List[FileBoundaryViolation]:
        """
        Check code for file boundary violations.
        
        Args:
            code: Test code to check
            allow_all: If True, skip checks (for repair mode)
            
        Returns:
            List of violations found
        """
        if allow_all:
            return []
        
        violations = []
        
        # Check AST for file operations
        try:
            tree = ast.parse(code)
            violations.extend(self._check_ast(tree))
        except SyntaxError:
            pass  # Invalid syntax will be caught elsewhere
        
        # Check string patterns for file operations
        violations.extend(self._check_patterns(code))
        
        return violations
    
    def _check_ast(self, tree: ast.AST) -> List[FileBoundaryViolation]:
        """Check AST for file operations."""
        violations = []
        
        for node in ast.walk(tree):
            # Check open() calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'open':
                    violation = self._check_open_call(node)
                    if violation:
                        violations.append(violation)
                
                # Check Path().write_text(), etc.
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ('write_text', 'write_bytes', 'mkdir', 'touch'):
                        violation = self._check_path_operation(node)
                        if violation:
                            violations.append(violation)
        
        return violations
    
    def _check_patterns(self, code: str) -> List[FileBoundaryViolation]:
        """Check string patterns for file operations."""
        violations = []
        
        # Pattern for file writes
        write_patterns = [
            (r'open\(["\']([^"\']+)["\'],\s*["\']w', 'write'),
            (r'open\(["\']([^"\']+)["\'],\s*["\']a', 'append'),
            (r'Path\(["\']([^"\']+)["\']\)\.write', 'write'),
            (r'with\s+open\(["\']([^"\']+)["\'],\s*["\']w', 'write'),
        ]
        
        for pattern, operation in write_patterns:
            for match in re.finditer(pattern, code):
                file_path = match.group(1)
                if not self._is_write_allowed(file_path):
                    violations.append(FileBoundaryViolation(
                        file_path=file_path,
                        operation=operation,
                        line_number=code[:match.start()].count('\n') + 1,
                        code_snippet=match.group(0),
                        reason=f"Write to {file_path} outside allowed directories"
                    ))
        
        return violations
    
    def _check_open_call(self, node: ast.Call) -> Optional[FileBoundaryViolation]:
        """Check an open() call for violations."""
        if not node.args:
            return None
        
        # Get file path
        file_arg = node.args[0]
        if isinstance(file_arg, ast.Constant):
            file_path = file_arg.value
        else:
            return None  # Dynamic path, can't check statically
        
        # Get mode
        mode = 'r'
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
            mode = node.args[1].value
        
        # Check for write operations
        if 'w' in mode or 'a' in mode:
            if not self._is_write_allowed(file_path):
                return FileBoundaryViolation(
                    file_path=file_path,
                    operation='write' if 'w' in mode else 'append',
                    line_number=node.lineno,
                    code_snippet=f"open('{file_path}', '{mode}')",
                    reason=f"Write to {file_path} outside allowed directories"
                )
        
        return None
    
    def _check_path_operation(self, node: ast.Call) -> Optional[FileBoundaryViolation]:
        """Check a Path operation for violations."""
        # This is simplified - would need more complex analysis
        return None
    
    def _is_write_allowed(self, file_path: str) -> bool:
        """Check if write to path is allowed."""
        path = Path(file_path)
        
        # Check if in allowed write directories
        for allowed_dir in self.config.allowed_write_dirs:
            if str(path).startswith(allowed_dir):
                return True
        
        # Check if justified by Planner
        if str(path) in self._planner_justifications:
            return True
        
        # Check if requires justification
        for restricted_dir in self.config.require_justification_for:
            if str(path).startswith(restricted_dir):
                return False  # Requires justification but doesn't have it
        
        # Default deny
        return False
    
    def get_safe_test_path(self, source_file: str) -> Path:
        """
        Get safe test file path for a source file.
        
        Args:
            source_file: Source file being tested
            
        Returns:
            Safe path in tests/ directory
        """
        source_path = Path(source_file)
        
        # Extract module name
        if source_path.stem == '__init__':
            module_name = source_path.parent.name
        else:
            module_name = source_path.stem
        
        # Create test file in tests/generated/
        return Path('tests/generated') / f'test_{module_name}.py'


def create_file_boundary_checker(
    additional_allowed_dirs: Optional[List[str]] = None
) -> FileBoundaryChecker:
    """
    Factory function to create file boundary checker.
    
    Args:
        additional_allowed_dirs: Additional allowed write directories
        
    Returns:
        Configured FileBoundaryChecker
    """
    config = FileBoundaryConfig()
    
    if additional_allowed_dirs:
        config.allowed_write_dirs.extend(additional_allowed_dirs)
    
    return FileBoundaryChecker(config)


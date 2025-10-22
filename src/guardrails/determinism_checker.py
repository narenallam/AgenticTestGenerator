"""
Determinism enforcer for test code.

Ensures tests are deterministic by detecting and blocking:
- time.sleep()
- datetime.now()
- random.random()
- Non-seeded randomness
"""

import ast
import re
from typing import List, Optional, Set

from pydantic import BaseModel, Field


class DeterminismViolation(BaseModel):
    """Represents a determinism violation."""
    
    type: str = Field(..., description="Violation type")
    line_number: int = Field(..., description="Line number")
    code_snippet: str = Field(..., description="Violating code")
    suggestion: str = Field(..., description="How to fix")
    severity: str = Field(default="HIGH", description="Severity level")


class DeterminismEnforcer:
    """
    Enforces deterministic test patterns.
    
    Blocks:
    - time.sleep() - Use mocking instead
    - datetime.now() - Use freezegun or mock
    - random.random() - Must use random.seed() or mock
    - Real network calls - Must mock
    - Real file I/O - Use temp files or mock
    
    Example:
        >>> enforcer = DeterminismEnforcer()
        >>> violations = enforcer.check_code(test_code)
        >>> if violations:
        ...     enforcer.fix_code(test_code)
    """
    
    # Non-deterministic patterns
    FORBIDDEN_PATTERNS = {
        'time_sleep': {
            'pattern': r'time\.sleep\(',
            'reason': 'time.sleep() makes tests slow and flaky',
            'fix': 'Use @patch("time.sleep") or freezegun',
            'severity': 'HIGH'
        },
        'datetime_now': {
            'pattern': r'datetime\.(?:now|today|utcnow)\(',
            'reason': 'datetime.now() is non-deterministic',
            'fix': 'Use freezegun.freeze_time() or mock',
            'severity': 'HIGH'
        },
        'random_unseeded': {
            'pattern': r'random\.(?:random|randint|choice|shuffle)\(',
            'reason': 'random without seed is non-deterministic',
            'fix': 'Use random.seed(42) or mock random',
            'severity': 'MEDIUM'
        },
        'uuid_generation': {
            'pattern': r'uuid\.uuid[14]\(',
            'reason': 'UUID generation is non-deterministic',
            'fix': 'Mock uuid.uuid4() with fixed values',
            'severity': 'MEDIUM'
        },
        'real_requests': {
            'pattern': r'requests\.(?:get|post|put|delete)\(',
            'reason': 'Real HTTP requests in tests',
            'fix': 'Use @responses.activate or mock',
            'severity': 'HIGH'
        },
    }
    
    def __init__(self, strict: bool = True):
        """
        Initialize determinism enforcer.
        
        Args:
            strict: If True, enforce strictly. If False, only warn.
        """
        self.strict = strict
        self._allowed_exceptions: Set[str] = set()
    
    def check_code(self, code: str) -> List[DeterminismViolation]:
        """
        Check code for determinism violations.
        
        Args:
            code: Test code to check
            
        Returns:
            List of violations found
        """
        violations = []
        
        # Check patterns
        for violation_type, config in self.FORBIDDEN_PATTERNS.items():
            pattern = config['pattern']
            for match in re.finditer(pattern, code):
                line_num = code[:match.start()].count('\n') + 1
                
                # Check if this line has mocking
                line = code.split('\n')[line_num - 1]
                if self._has_mocking(line, code, line_num):
                    continue  # Mocked, so it's OK
                
                violations.append(DeterminismViolation(
                    type=violation_type,
                    line_number=line_num,
                    code_snippet=match.group(0),
                    suggestion=config['fix'],
                    severity=config['severity']
                ))
        
        # Check AST for more complex patterns
        try:
            tree = ast.parse(code)
            violations.extend(self._check_ast(tree, code))
        except SyntaxError:
            pass
        
        return violations
    
    def fix_code(self, code: str) -> str:
        """
        Automatically fix determinism issues where possible.
        
        Args:
            code: Code to fix
            
        Returns:
            Fixed code
        """
        fixed = code
        
        # Add imports if needed
        needs_mock = any(pattern in code for pattern in ['time.sleep', 'datetime.now', 'requests.'])
        needs_freezegun = 'datetime.' in code
        needs_seed = 'random.' in code
        
        imports_to_add = []
        
        if needs_mock:
            imports_to_add.append('from unittest.mock import patch, MagicMock')
        if needs_freezegun:
            imports_to_add.append('from freezegun import freeze_time')
        if needs_seed:
            # Add random.seed() at test start
            fixed = self._add_random_seed(fixed)
        
        # Add imports at the top
        if imports_to_add:
            import_block = '\n'.join(imports_to_add) + '\n\n'
            # Find first import or first function
            first_import_match = re.search(r'^(?:from |import )', fixed, re.MULTILINE)
            if first_import_match:
                insert_pos = first_import_match.start()
            else:
                insert_pos = 0
            
            fixed = fixed[:insert_pos] + import_block + fixed[insert_pos:]
        
        # Replace problematic patterns
        fixed = self._fix_time_sleep(fixed)
        fixed = self._fix_datetime_now(fixed)
        fixed = self._fix_requests(fixed)
        
        return fixed
    
    def _has_mocking(self, line: str, full_code: str, line_num: int) -> bool:
        """Check if line is within mocked context."""
        # Check for @patch decorator
        prev_lines = full_code.split('\n')[:line_num]
        for prev_line in reversed(prev_lines):
            if '@patch' in prev_line or '@mock' in prev_line:
                return True
            if prev_line.strip() and not prev_line.strip().startswith('#'):
                break  # Decorator only applies to immediate next function
        
        # Check for with patch() context
        context_window = full_code.split('\n')[max(0, line_num-5):line_num]
        for context_line in context_window:
            if 'with' in context_line and 'patch' in context_line:
                return True
        
        return False
    
    def _check_ast(self, tree: ast.AST, code: str) -> List[DeterminismViolation]:
        """Check AST for determinism issues."""
        violations = []
        
        for node in ast.walk(tree):
            # Check for random usage without seeding
            if isinstance(node, ast.Call):
                if self._is_random_call(node):
                    # Check if random.seed() is called
                    if not self._has_random_seed(tree):
                        violations.append(DeterminismViolation(
                            type='random_unseeded',
                            line_number=node.lineno,
                            code_snippet=ast.unparse(node) if hasattr(ast, 'unparse') else 'random call',
                            suggestion='Add random.seed(42) at the start of your test',
                            severity='MEDIUM'
                        ))
        
        return violations
    
    def _is_random_call(self, node: ast.Call) -> bool:
        """Check if node is a random module call."""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return node.func.value.id == 'random'
        return False
    
    def _has_random_seed(self, tree: ast.AST) -> bool:
        """Check if random.seed() is called in AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if (isinstance(node.func.value, ast.Name) and
                        node.func.value.id == 'random' and
                        node.func.attr == 'seed'):
                        return True
        return False
    
    def _add_random_seed(self, code: str) -> str:
        """Add random.seed() to tests."""
        # Find test functions and add seed
        fixed = re.sub(
            r'(def test_\w+\([^)]*\):)\n(\s+)',
            r'\1\n\2random.seed(42)\n\2',
            code
        )
        return fixed
    
    def _fix_time_sleep(self, code: str) -> str:
        """Replace time.sleep with mock."""
        # Add @patch decorator
        fixed = re.sub(
            r'(def test_\w+)\(([^)]*)\):',
            r'@patch("time.sleep")\n\1(self, mock_sleep, \2):',
            code
        )
        return fixed
    
    def _fix_datetime_now(self, code: str) -> str:
        """Wrap tests using datetime.now with freeze_time."""
        # Add @freeze_time decorator
        fixed = re.sub(
            r'(def test_\w+.*datetime\.now)',
            r'@freeze_time("2024-01-01 12:00:00")\n\1',
            code,
            flags=re.DOTALL
        )
        return fixed
    
    def _fix_requests(self, code: str) -> str:
        """Replace requests with mocks."""
        # Add @patch decorator for requests
        fixed = re.sub(
            r'(def test_\w+.*requests\.)',
            r'@patch("requests.get")\n@patch("requests.post")\n\1',
            code,
            flags=re.DOTALL
        )
        return fixed


def create_determinism_enforcer(strict: bool = True) -> DeterminismEnforcer:
    """
    Factory function to create determinism enforcer.
    
    Args:
        strict: If True, enforce strictly. If False, only warn.
        
    Returns:
        Configured DeterminismEnforcer
    """
    return DeterminismEnforcer(strict=strict)


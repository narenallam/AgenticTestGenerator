"""
AST and Control Flow Graph (CFG) analyzer for deep code understanding.

This module provides:
- Abstract Syntax Tree parsing
- Control Flow Graph generation
- Data flow analysis
- Complexity metrics
- External call detection (for mocking)
"""

import ast
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field
from rich.console import Console

console = Console()


class NodeType(str, Enum):
    """Types of CFG nodes."""
    
    ENTRY = "entry"
    EXIT = "exit"
    STATEMENT = "statement"
    CONDITION = "condition"
    LOOP = "loop"
    FUNCTION_CALL = "function_call"
    RETURN = "return"
    EXCEPTION = "exception"


@dataclass
class CFGNode:
    """Node in Control Flow Graph."""
    
    id: int
    type: NodeType
    ast_node: Optional[ast.AST] = None
    source_line: Optional[int] = None
    code: Optional[str] = None
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return f"CFGNode(id={self.id}, type={self.type}, line={self.source_line})"


class ControlFlowGraph:
    """
    Control Flow Graph representation.
    
    Represents the control flow structure of a function or code block.
    """
    
    def __init__(self, function_name: str):
        self.function_name = function_name
        self.nodes: Dict[int, CFGNode] = {}
        self.entry_node_id: Optional[int] = None
        self.exit_node_id: Optional[int] = None
        self.next_id = 0
    
    def add_node(
        self,
        node_type: NodeType,
        ast_node: Optional[ast.AST] = None,
        code: Optional[str] = None
    ) -> CFGNode:
        """Add a node to the CFG."""
        node_id = self.next_id
        self.next_id += 1
        
        line_num = None
        if ast_node and hasattr(ast_node, 'lineno'):
            line_num = ast_node.lineno
        
        node = CFGNode(
            id=node_id,
            type=node_type,
            ast_node=ast_node,
            source_line=line_num,
            code=code
        )
        
        self.nodes[node_id] = node
        return node
    
    def add_edge(self, from_id: int, to_id: int) -> None:
        """Add an edge between two nodes."""
        if from_id not in self.nodes or to_id not in self.nodes:
            return
        
        if to_id not in self.nodes[from_id].successors:
            self.nodes[from_id].successors.append(to_id)
        
        if from_id not in self.nodes[to_id].predecessors:
            self.nodes[to_id].predecessors.append(from_id)
    
    def get_paths(self) -> List[List[int]]:
        """Get all paths from entry to exit."""
        if not self.entry_node_id or not self.exit_node_id:
            return []
        
        paths = []
        self._dfs_paths(self.entry_node_id, [self.entry_node_id], paths)
        return paths
    
    def _dfs_paths(
        self,
        current_id: int,
        current_path: List[int],
        all_paths: List[List[int]]
    ) -> None:
        """DFS to find all paths."""
        if current_id == self.exit_node_id:
            all_paths.append(current_path.copy())
            return
        
        node = self.nodes[current_id]
        for successor_id in node.successors:
            if successor_id not in current_path:  # Avoid cycles
                current_path.append(successor_id)
                self._dfs_paths(successor_id, current_path, all_paths)
                current_path.pop()
    
    def get_branch_nodes(self) -> List[CFGNode]:
        """Get all branching nodes (conditions, loops)."""
        return [
            node for node in self.nodes.values()
            if node.type in [NodeType.CONDITION, NodeType.LOOP] and len(node.successors) > 1
        ]
    
    def calculate_cyclomatic_complexity(self) -> int:
        """
        Calculate cyclomatic complexity: M = E - N + 2P
        where E = edges, N = nodes, P = connected components (usually 1)
        """
        edges = sum(len(node.successors) for node in self.nodes.values())
        nodes = len(self.nodes)
        return edges - nodes + 2  # Assuming 1 connected component


class FunctionInfo(BaseModel):
    """Information about a function."""
    
    name: str = Field(..., description="Function name")
    lineno: int = Field(..., description="Starting line number")
    end_lineno: int = Field(..., description="Ending line number")
    args: List[str] = Field(default_factory=list, description="Argument names")
    returns: Optional[str] = Field(default=None, description="Return type annotation")
    decorators: List[str] = Field(default_factory=list, description="Decorator names")
    docstring: Optional[str] = Field(default=None, description="Docstring")
    is_async: bool = Field(default=False, description="Is async function")
    complexity: int = Field(default=1, description="Cyclomatic complexity")
    external_calls: List[str] = Field(default_factory=list, description="External function calls")
    imports_used: Set[str] = Field(default_factory=set, description="Imports used")


class ClassInfo(BaseModel):
    """Information about a class."""
    
    name: str = Field(..., description="Class name")
    lineno: int = Field(..., description="Starting line number")
    end_lineno: int = Field(..., description="Ending line number")
    bases: List[str] = Field(default_factory=list, description="Base classes")
    methods: List[FunctionInfo] = Field(default_factory=list, description="Methods")
    decorators: List[str] = Field(default_factory=list, description="Decorators")
    docstring: Optional[str] = Field(default=None, description="Docstring")


class ASTAnalysisResult(BaseModel):
    """Result of AST analysis."""
    
    functions: List[FunctionInfo] = Field(default_factory=list)
    classes: List[ClassInfo] = Field(default_factory=list)
    imports: List[str] = Field(default_factory=list)
    global_variables: List[str] = Field(default_factory=list)
    total_lines: int = Field(default=0)
    total_complexity: int = Field(default=0)


class ASTAnalyzer:
    """
    Comprehensive AST analyzer for Python code.
    
    Provides deep code analysis including CFG generation,
    complexity metrics, and dependency analysis.
    """
    
    def __init__(self):
        self.tree: Optional[ast.AST] = None
        self.source_lines: List[str] = []
    
    def parse(self, code: str) -> ast.AST:
        """
        Parse Python code into AST.
        
        Args:
            code: Python source code
            
        Returns:
            AST root node
            
        Raises:
            SyntaxError: If code has syntax errors
        """
        self.source_lines = code.splitlines()
        self.tree = ast.parse(code)
        return self.tree
    
    def parse_file(self, file_path: Path) -> ast.AST:
        """Parse a Python file."""
        code = file_path.read_text(encoding='utf-8')
        return self.parse(code)
    
    def analyze(self, code: str) -> ASTAnalysisResult:
        """
        Perform comprehensive AST analysis.
        
        Args:
            code: Python source code
            
        Returns:
            ASTAnalysisResult with detailed information
            
        Example:
            >>> analyzer = ASTAnalyzer()
            >>> result = analyzer.analyze("def foo(x): return x + 1")
            >>> print(f"Functions: {len(result.functions)}")
        """
        tree = self.parse(code)
        
        # Extract functions
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                func_info = self._extract_function_info(node)
                functions.append(func_info)
        
        # Extract classes
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = self._extract_class_info(node)
                classes.append(class_info)
        
        # Extract imports
        imports = self._extract_imports(tree)
        
        # Extract global variables
        global_vars = self._extract_global_variables(tree)
        
        # Calculate total complexity
        total_complexity = sum(f.complexity for f in functions)
        for cls in classes:
            total_complexity += sum(m.complexity for m in cls.methods)
        
        return ASTAnalysisResult(
            functions=functions,
            classes=classes,
            imports=imports,
            global_variables=global_vars,
            total_lines=len(self.source_lines),
            total_complexity=total_complexity
        )
    
    def _extract_function_info(self, node: ast.FunctionDef) -> FunctionInfo:
        """Extract detailed information about a function."""
        # Get argument names
        args = [arg.arg for arg in node.args.args]
        
        # Get return type if annotated
        returns = None
        if node.returns:
            returns = ast.unparse(node.returns) if hasattr(ast, 'unparse') else None
        
        # Get decorators
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                decorators.append(dec.func.id)
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Calculate complexity
        complexity = self._calculate_complexity(node)
        
        # Find external calls
        external_calls = self._find_external_calls(node)
        
        # Find imports used
        imports_used = self._find_imports_used(node)
        
        return FunctionInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            args=args,
            returns=returns,
            decorators=decorators,
            docstring=docstring,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            complexity=complexity,
            external_calls=external_calls,
            imports_used=imports_used
        )
    
    def _extract_class_info(self, node: ast.ClassDef) -> ClassInfo:
        """Extract detailed information about a class."""
        # Get base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
        
        # Get decorators
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = self._extract_function_info(item)
                methods.append(method_info)
        
        return ClassInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            bases=bases,
            methods=methods,
            decorators=decorators,
            docstring=docstring
        )
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all imports from the AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
        
        return imports
    
    def _extract_global_variables(self, tree: ast.AST) -> List[str]:
        """Extract global variable names."""
        variables = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append(target.id)
        
        return variables
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Decision points
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            # Boolean operators
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            # Ternary expressions
            elif isinstance(child, ast.IfExp):
                complexity += 1
        
        return complexity
    
    def _find_external_calls(self, node: ast.FunctionDef) -> List[str]:
        """Find all external function/method calls."""
        calls = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    # Get full attribute chain
                    call_str = self._get_attribute_chain(child.func)
                    calls.append(call_str)
        
        return list(set(calls))  # Remove duplicates
    
    def _find_imports_used(self, node: ast.FunctionDef) -> Set[str]:
        """Find which imports are used in a function."""
        imports_used = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                imports_used.add(child.id)
            elif isinstance(child, ast.Attribute):
                # Get the root name
                root = child
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name):
                    imports_used.add(root.id)
        
        return imports_used
    
    def _get_attribute_chain(self, node: ast.Attribute) -> str:
        """Get full attribute chain (e.g., 'obj.method.call')."""
        parts = []
        
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            parts.append(current.id)
        
        return '.'.join(reversed(parts))
    
    def build_cfg(self, function_node: ast.FunctionDef) -> ControlFlowGraph:
        """
        Build Control Flow Graph for a function.
        
        Args:
            function_node: AST node for function
            
        Returns:
            ControlFlowGraph
            
        Example:
            >>> analyzer = ASTAnalyzer()
            >>> tree = analyzer.parse("def foo(x):\\n if x > 0:\\n return x\\n return 0")
            >>> func = tree.body[0]
            >>> cfg = analyzer.build_cfg(func)
            >>> print(f"Nodes: {len(cfg.nodes)}, Paths: {len(cfg.get_paths())}")
        """
        cfg = ControlFlowGraph(function_node.name)
        
        # Create entry and exit nodes
        entry = cfg.add_node(NodeType.ENTRY, code=f"def {function_node.name}(...)")
        cfg.entry_node_id = entry.id
        
        exit_node = cfg.add_node(NodeType.EXIT, code="return")
        cfg.exit_node_id = exit_node.id
        
        # Build CFG from function body
        last_nodes = [entry.id]
        last_nodes = self._build_cfg_from_statements(
            cfg,
            function_node.body,
            last_nodes,
            exit_node.id
        )
        
        # Connect last nodes to exit
        for node_id in last_nodes:
            cfg.add_edge(node_id, exit_node.id)
        
        return cfg
    
    def _build_cfg_from_statements(
        self,
        cfg: ControlFlowGraph,
        statements: List[ast.stmt],
        entry_nodes: List[int],
        exit_node_id: int
    ) -> List[int]:
        """Build CFG from a list of statements."""
        current_nodes = entry_nodes
        
        for stmt in statements:
            current_nodes = self._build_cfg_from_statement(
                cfg, stmt, current_nodes, exit_node_id
            )
        
        return current_nodes
    
    def _build_cfg_from_statement(
        self,
        cfg: ControlFlowGraph,
        stmt: ast.stmt,
        entry_nodes: List[int],
        exit_node_id: int
    ) -> List[int]:
        """Build CFG from a single statement."""
        if isinstance(stmt, ast.If):
            return self._build_if_cfg(cfg, stmt, entry_nodes, exit_node_id)
        elif isinstance(stmt, (ast.While, ast.For)):
            return self._build_loop_cfg(cfg, stmt, entry_nodes, exit_node_id)
        elif isinstance(stmt, ast.Return):
            return self._build_return_cfg(cfg, stmt, entry_nodes, exit_node_id)
        else:
            # Simple statement
            node = cfg.add_node(NodeType.STATEMENT, ast_node=stmt)
            for entry_id in entry_nodes:
                cfg.add_edge(entry_id, node.id)
            return [node.id]
    
    def _build_if_cfg(
        self,
        cfg: ControlFlowGraph,
        stmt: ast.If,
        entry_nodes: List[int],
        exit_node_id: int
    ) -> List[int]:
        """Build CFG for if statement."""
        # Create condition node
        cond_node = cfg.add_node(NodeType.CONDITION, ast_node=stmt, code="if ...")
        for entry_id in entry_nodes:
            cfg.add_edge(entry_id, cond_node.id)
        
        # Process then branch
        then_exits = self._build_cfg_from_statements(
            cfg, stmt.body, [cond_node.id], exit_node_id
        )
        
        # Process else branch
        if stmt.orelse:
            else_exits = self._build_cfg_from_statements(
                cfg, stmt.orelse, [cond_node.id], exit_node_id
            )
        else:
            else_exits = [cond_node.id]  # Fall through
        
        return then_exits + else_exits
    
    def _build_loop_cfg(
        self,
        cfg: ControlFlowGraph,
        stmt: ast.stmt,
        entry_nodes: List[int],
        exit_node_id: int
    ) -> List[int]:
        """Build CFG for loop statement."""
        # Create loop node
        loop_node = cfg.add_node(NodeType.LOOP, ast_node=stmt, code="loop ...")
        for entry_id in entry_nodes:
            cfg.add_edge(entry_id, loop_node.id)
        
        # Process loop body
        body_exits = self._build_cfg_from_statements(
            cfg,
            stmt.body if hasattr(stmt, 'body') else [],
            [loop_node.id],
            exit_node_id
        )
        
        # Loop back
        for body_exit in body_exits:
            cfg.add_edge(body_exit, loop_node.id)
        
        # Exit loop
        return [loop_node.id]
    
    def _build_return_cfg(
        self,
        cfg: ControlFlowGraph,
        stmt: ast.Return,
        entry_nodes: List[int],
        exit_node_id: int
    ) -> List[int]:
        """Build CFG for return statement."""
        return_node = cfg.add_node(NodeType.RETURN, ast_node=stmt, code="return")
        for entry_id in entry_nodes:
            cfg.add_edge(entry_id, return_node.id)
        
        # Connect to exit
        cfg.add_edge(return_node.id, exit_node_id)
        
        return []  # No successors (terminates)


def create_ast_analyzer() -> ASTAnalyzer:
    """Create an AST analyzer instance."""
    return ASTAnalyzer()


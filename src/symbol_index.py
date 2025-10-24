"""
Symbol Index - Fast exact lookups and call graph analysis.

This module provides traditional symbol-based code search (like ctags/LSP)
to complement the semantic vector search.

Features:
- O(1) function/class lookups by name
- Call graph analysis (who calls what)
- Import graph analysis (who imports what)
- Reference tracking (find all usages)
"""

import ast
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@dataclass
class Location:
    """Code location."""
    file_path: str
    line_number: int
    end_line: int
    column: int = 0


@dataclass
class FunctionSymbol:
    """Function symbol information."""
    name: str
    location: Location
    signature: str
    docstring: Optional[str]
    is_async: bool
    is_method: bool
    class_name: Optional[str]
    complexity: int
    calls: List[str]  # Functions this calls
    

@dataclass
class ClassSymbol:
    """Class symbol information."""
    name: str
    location: Location
    docstring: Optional[str]
    bases: List[str]
    methods: List[str]


@dataclass
class ImportSymbol:
    """Import statement information."""
    module: str
    names: List[str]
    location: Location
    is_from_import: bool


@dataclass
class CallSite:
    """Function call site."""
    caller: str
    callee: str
    location: Location


class SymbolIndex:
    """
    Fast symbol index for exact lookups and call graph analysis.
    
    This provides traditional IDE-like functionality:
    - Jump to definition
    - Find all references
    - Call hierarchy
    - Import analysis
    """
    
    def __init__(self, index_file: Optional[Path] = None):
        """
        Initialize symbol index.
        
        Args:
            index_file: Path to persist index (JSON)
        """
        self.index_file = index_file or Path(".symbol_index.json")
        
        # Symbol tables
        self.functions: Dict[str, List[FunctionSymbol]] = defaultdict(list)
        self.classes: Dict[str, List[ClassSymbol]] = defaultdict(list)
        self.imports: Dict[str, List[ImportSymbol]] = defaultdict(list)
        
        # Call graph
        self.calls: Dict[str, List[CallSite]] = defaultdict(list)
        self.callers: Dict[str, List[CallSite]] = defaultdict(list)
        
        # File index (for efficient updates)
        self.files_indexed: Set[str] = set()
        
        # Try to load existing index
        self._load_index()
        
        console.print("[green]✓[/green] Symbol Index initialized")
    
    def index_file(self, file_path: Path) -> int:
        """
        Index a single file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Number of symbols indexed
        """
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            file_str = str(file_path)
            symbols_count = 0
            
            # Clear existing symbols for this file
            self._clear_file_symbols(file_str)
            
            # Index functions and methods
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbol = self._extract_function(node, file_str)
                    self.functions[symbol.name].append(symbol)
                    symbols_count += 1
                    
                    # Index calls made by this function
                    self._index_calls(node, symbol.name, file_str)
                
                elif isinstance(node, ast.ClassDef):
                    symbol = self._extract_class(node, file_str)
                    self.classes[symbol.name].append(symbol)
                    symbols_count += 1
            
            # Index imports
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_sym = self._extract_import(node, file_str)
                    if import_sym:
                        self.imports[import_sym.module].append(import_sym)
                        symbols_count += 1
            
            self.files_indexed.add(file_str)
            return symbols_count
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Failed to index {file_path}: {e}[/yellow]")
            return 0
    
    def index_directory(
        self,
        directory: Path,
        file_extensions: Optional[Set[str]] = None
    ) -> Dict[str, int]:
        """
        Index all files in a directory.
        
        Args:
            directory: Root directory to index
            file_extensions: File extensions to index (default: {'.py'})
            
        Returns:
            Statistics dict
        """
        if file_extensions is None:
            file_extensions = {'.py'}
        
        files_to_index = []
        for ext in file_extensions:
            files_to_index.extend(directory.rglob(f"*{ext}"))
        
        stats = {
            'files_indexed': 0,
            'symbols_found': 0,
            'functions': 0,
            'classes': 0,
            'imports': 0
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(
                f"Indexing {len(files_to_index)} files...",
                total=len(files_to_index)
            )
            
            for file_path in files_to_index:
                symbols = self.index_file(file_path)
                stats['symbols_found'] += symbols
                stats['files_indexed'] += 1
                progress.update(task, advance=1)
        
        stats['functions'] = sum(len(syms) for syms in self.functions.values())
        stats['classes'] = sum(len(syms) for syms in self.classes.values())
        stats['imports'] = sum(len(syms) for syms in self.imports.values())
        
        console.print(f"[green]✓[/green] Indexed {stats['files_indexed']} files")
        console.print(f"  → Functions: {stats['functions']}")
        console.print(f"  → Classes: {stats['classes']}")
        console.print(f"  → Imports: {stats['imports']}")
        
        # Save index
        self._save_index()
        
        return stats
    
    def find_function(self, name: str) -> List[FunctionSymbol]:
        """
        Find function by exact name.
        
        Args:
            name: Function name to find
            
        Returns:
            List of matching function symbols
        """
        return self.functions.get(name, [])
    
    def find_class(self, name: str) -> List[ClassSymbol]:
        """Find class by exact name."""
        return self.classes.get(name, [])
    
    def find_callers(self, function_name: str) -> List[CallSite]:
        """
        Find all functions that call the given function.
        
        Args:
            function_name: Name of function to find callers for
            
        Returns:
            List of call sites
        """
        return self.callers.get(function_name, [])
    
    def find_callees(self, function_name: str) -> List[CallSite]:
        """
        Find all functions called by the given function.
        
        Args:
            function_name: Name of function
            
        Returns:
            List of call sites
        """
        return self.calls.get(function_name, [])
    
    def find_imports_of(self, module_name: str) -> List[ImportSymbol]:
        """
        Find all places that import a module.
        
        Args:
            module_name: Module name (e.g., 'os', 'pathlib')
            
        Returns:
            List of import symbols
        """
        return self.imports.get(module_name, [])
    
    def get_call_graph(self, function_name: str, depth: int = 2) -> Dict:
        """
        Get call graph for a function (both up and down).
        
        Args:
            function_name: Root function name
            depth: How many levels to traverse
            
        Returns:
            Dict with 'callers' and 'callees'
        """
        result = {
            'callers': self._get_callers_recursive(function_name, depth),
            'callees': self._get_callees_recursive(function_name, depth)
        }
        return result
    
    def _get_callers_recursive(
        self,
        function_name: str,
        depth: int,
        visited: Optional[Set[str]] = None
    ) -> List[Dict]:
        """Recursively get callers."""
        if visited is None:
            visited = set()
        
        if depth == 0 or function_name in visited:
            return []
        
        visited.add(function_name)
        result = []
        
        for call_site in self.find_callers(function_name):
            caller_info = {
                'name': call_site.caller,
                'location': {
                    'file': call_site.location.file_path,
                    'line': call_site.location.line_number
                },
                'callers': self._get_callers_recursive(
                    call_site.caller,
                    depth - 1,
                    visited
                )
            }
            result.append(caller_info)
        
        return result
    
    def _get_callees_recursive(
        self,
        function_name: str,
        depth: int,
        visited: Optional[Set[str]] = None
    ) -> List[Dict]:
        """Recursively get callees."""
        if visited is None:
            visited = set()
        
        if depth == 0 or function_name in visited:
            return []
        
        visited.add(function_name)
        result = []
        
        for call_site in self.find_callees(function_name):
            callee_info = {
                'name': call_site.callee,
                'location': {
                    'file': call_site.location.file_path,
                    'line': call_site.location.line_number
                },
                'callees': self._get_callees_recursive(
                    call_site.callee,
                    depth - 1,
                    visited
                )
            }
            result.append(callee_info)
        
        return result
    
    def _extract_function(
        self,
        node: ast.FunctionDef,
        file_path: str
    ) -> FunctionSymbol:
        """Extract function symbol from AST node."""
        # Get signature
        args = [arg.arg for arg in node.args.args]
        signature = f"{node.name}({', '.join(args)})"
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Check if method
        is_method = False
        class_name = None
        # (Would need parent tracking for accurate detection)
        
        # Calculate complexity
        complexity = self._calculate_complexity(node)
        
        # Extract calls
        calls = self._extract_calls(node)
        
        return FunctionSymbol(
            name=node.name,
            location=Location(
                file_path=file_path,
                line_number=node.lineno,
                end_line=node.end_lineno or node.lineno,
                column=node.col_offset
            ),
            signature=signature,
            docstring=docstring,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=is_method,
            class_name=class_name,
            complexity=complexity,
            calls=calls
        )
    
    def _extract_class(
        self,
        node: ast.ClassDef,
        file_path: str
    ) -> ClassSymbol:
        """Extract class symbol from AST node."""
        docstring = ast.get_docstring(node)
        
        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
        
        # Extract method names
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)
        
        return ClassSymbol(
            name=node.name,
            location=Location(
                file_path=file_path,
                line_number=node.lineno,
                end_line=node.end_lineno or node.lineno,
                column=node.col_offset
            ),
            docstring=docstring,
            bases=bases,
            methods=methods
        )
    
    def _extract_import(
        self,
        node: ast.AST,
        file_path: str
    ) -> Optional[ImportSymbol]:
        """Extract import symbol from AST node."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                return ImportSymbol(
                    module=alias.name,
                    names=[alias.name],
                    location=Location(
                        file_path=file_path,
                        line_number=node.lineno,
                        end_line=node.lineno,
                        column=node.col_offset
                    ),
                    is_from_import=False
                )
        
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            names = [alias.name for alias in node.names]
            return ImportSymbol(
                module=module,
                names=names,
                location=Location(
                    file_path=file_path,
                    line_number=node.lineno,
                    end_line=node.lineno,
                    column=node.col_offset
                ),
                is_from_import=True
            )
        
        return None
    
    def _extract_calls(self, node: ast.AST) -> List[str]:
        """Extract function calls from AST node."""
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        return list(set(calls))  # Deduplicate
    
    def _index_calls(
        self,
        node: ast.FunctionDef,
        caller_name: str,
        file_path: str
    ) -> None:
        """Index all calls made by a function."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee_name = None
                
                if isinstance(child.func, ast.Name):
                    callee_name = child.func.id
                elif isinstance(child.func, ast.Attribute):
                    callee_name = child.func.attr
                
                if callee_name:
                    call_site = CallSite(
                        caller=caller_name,
                        callee=callee_name,
                        location=Location(
                            file_path=file_path,
                            line_number=child.lineno,
                            end_line=child.end_lineno or child.lineno,
                            column=child.col_offset
                        )
                    )
                    
                    # Add to both directions
                    self.calls[caller_name].append(call_site)
                    self.callers[callee_name].append(call_site)
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def _clear_file_symbols(self, file_path: str) -> None:
        """Clear all symbols for a file (for re-indexing)."""
        # Remove functions
        for name in list(self.functions.keys()):
            self.functions[name] = [
                s for s in self.functions[name]
                if s.location.file_path != file_path
            ]
            if not self.functions[name]:
                del self.functions[name]
        
        # Remove classes
        for name in list(self.classes.keys()):
            self.classes[name] = [
                s for s in self.classes[name]
                if s.location.file_path != file_path
            ]
            if not self.classes[name]:
                del self.classes[name]
        
        # Remove imports
        for module in list(self.imports.keys()):
            self.imports[module] = [
                s for s in self.imports[module]
                if s.location.file_path != file_path
            ]
            if not self.imports[module]:
                del self.imports[module]
        
        # Remove call sites
        for func in list(self.calls.keys()):
            self.calls[func] = [
                c for c in self.calls[func]
                if c.location.file_path != file_path
            ]
            if not self.calls[func]:
                del self.calls[func]
        
        for func in list(self.callers.keys()):
            self.callers[func] = [
                c for c in self.callers[func]
                if c.location.file_path != file_path
            ]
            if not self.callers[func]:
                del self.callers[func]
    
    def _save_index(self) -> None:
        """Save index to disk."""
        try:
            data = {
                'functions': {
                    name: [self._symbol_to_dict(s) for s in syms]
                    for name, syms in self.functions.items()
                },
                'classes': {
                    name: [self._symbol_to_dict(s) for s in syms]
                    for name, syms in self.classes.items()
                },
                'imports': {
                    module: [self._symbol_to_dict(s) for s in syms]
                    for module, syms in self.imports.items()
                },
                'calls': {
                    func: [self._callsite_to_dict(c) for c in sites]
                    for func, sites in self.calls.items()
                },
                'files_indexed': list(self.files_indexed)
            }
            
            self.index_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            console.print(f"[yellow]⚠️  Failed to save index: {e}[/yellow]")
    
    def _load_index(self) -> None:
        """Load index from disk."""
        if not self.index_file.exists():
            return
        
        try:
            data = json.loads(self.index_file.read_text())
            
            # Load functions
            for name, syms in data.get('functions', {}).items():
                self.functions[name] = [
                    self._dict_to_function_symbol(s) for s in syms
                ]
            
            # Load classes
            for name, syms in data.get('classes', {}).items():
                self.classes[name] = [
                    self._dict_to_class_symbol(s) for s in syms
                ]
            
            # Load imports
            for module, syms in data.get('imports', {}).items():
                self.imports[module] = [
                    self._dict_to_import_symbol(s) for s in syms
                ]
            
            # Load calls
            for func, sites in data.get('calls', {}).items():
                self.calls[func] = [
                    self._dict_to_callsite(c) for c in sites
                ]
            
            # Rebuild callers from calls
            for func, sites in self.calls.items():
                for site in sites:
                    self.callers[site.callee].append(site)
            
            self.files_indexed = set(data.get('files_indexed', []))
            
            console.print(f"[dim]Loaded {len(self.files_indexed)} indexed files[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Failed to load index: {e}[/yellow]")
    
    def _symbol_to_dict(self, symbol) -> Dict:
        """Convert symbol to dict for JSON."""
        result = asdict(symbol)
        # Convert Location to dict
        if 'location' in result:
            result['location'] = asdict(result['location'])
        return result
    
    def _callsite_to_dict(self, callsite: CallSite) -> Dict:
        """Convert callsite to dict for JSON."""
        return {
            'caller': callsite.caller,
            'callee': callsite.callee,
            'location': asdict(callsite.location)
        }
    
    def _dict_to_function_symbol(self, data: Dict) -> FunctionSymbol:
        """Convert dict to FunctionSymbol."""
        data['location'] = Location(**data['location'])
        return FunctionSymbol(**data)
    
    def _dict_to_class_symbol(self, data: Dict) -> ClassSymbol:
        """Convert dict to ClassSymbol."""
        data['location'] = Location(**data['location'])
        return ClassSymbol(**data)
    
    def _dict_to_import_symbol(self, data: Dict) -> ImportSymbol:
        """Convert dict to ImportSymbol."""
        data['location'] = Location(**data['location'])
        return ImportSymbol(**data)
    
    def _dict_to_callsite(self, data: Dict) -> CallSite:
        """Convert dict to CallSite."""
        data['location'] = Location(**data['location'])
        return CallSite(**data)


def create_symbol_index(index_file: Optional[Path] = None) -> SymbolIndex:
    """Factory function to create symbol index."""
    return SymbolIndex(index_file)


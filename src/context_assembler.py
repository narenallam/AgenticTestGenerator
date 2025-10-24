"""
Context Assembler - Pre-gather comprehensive context for test generation.

This module ensures that ALL necessary context is gathered BEFORE sending to LLM,
rather than relying on the LLM to call tools.

Key principle: FORCE context gathering, don't rely on LLM behavior.
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
from rich.console import Console

from src.code_embeddings import CodeEmbeddingStore
from src.git_integration import GitIntegration
from src.test_tracking_db import TestTrackingDB
from src.symbol_index import SymbolIndex
from src.hybrid_search import HybridSearchEngine

console = Console()


@dataclass
class AssembledContext:
    """Complete context bundle for test generation."""
    
    # Core
    target_code: str
    file_path: str
    function_name: Optional[str]
    
    # Related code
    related_functions: List[Dict[str, str]]  # Similar functions from vector search
    dependencies: List[str]  # Import statements
    
    # Usage analysis
    callers: List[Dict[str, str]]  # Functions that call this one
    callees: List[str]  # Functions this one calls
    usage_examples: List[str]  # Example usages
    
    # Historical
    git_history: List[Dict[str, str]]  # Recent changes
    existing_tests: Optional[str]  # Current test code if exists
    
    # Metadata
    complexity_score: int
    total_lines: int
    docstring: Optional[str]
    
    def to_llm_prompt_section(self) -> str:
        """Format context for LLM prompt."""
        sections = []
        
        # Target code
        sections.append("=== TARGET CODE ===")
        sections.append(f"File: {self.file_path}")
        if self.function_name:
            sections.append(f"Function: {self.function_name}")
        sections.append(f"Lines: {self.total_lines}")
        sections.append(f"Complexity: {self.complexity_score}")
        sections.append(f"\n{self.target_code}\n")
        
        # Docstring
        if self.docstring:
            sections.append("=== DOCSTRING ===")
            sections.append(self.docstring)
            sections.append("")
        
        # Dependencies
        if self.dependencies:
            sections.append("=== DEPENDENCIES ===")
            for dep in self.dependencies[:10]:  # Limit to top 10
                sections.append(f"  - {dep}")
            sections.append("")
        
        # Related code
        if self.related_functions:
            sections.append("=== RELATED CODE (for context) ===")
            for i, related in enumerate(self.related_functions[:3], 1):  # Top 3
                sections.append(f"\nRelated {i}: {related.get('name', 'unknown')}")
                sections.append(f"File: {related.get('file_path', 'unknown')}")
                sections.append(f"Relevance: {related.get('score', 0):.2f}")
                code = related.get('code', '')
                sections.append(f"{code[:300]}..." if len(code) > 300 else code)
        
        # Callers (who uses this function)
        if self.callers:
            sections.append("\n=== USAGE - Called By ===")
            for caller in self.callers[:5]:  # Top 5 callers
                sections.append(f"  - {caller.get('caller_name')} in {caller.get('file')}")
        
        # Callees (what this function uses)
        if self.callees:
            sections.append("\n=== DEPENDENCIES - Calls ===")
            for callee in self.callees[:10]:  # Top 10
                sections.append(f"  - {callee}")
        
        # Usage examples
        if self.usage_examples:
            sections.append("\n=== USAGE EXAMPLES ===")
            for i, example in enumerate(self.usage_examples[:2], 1):
                sections.append(f"\nExample {i}:")
                sections.append(example[:200] + "..." if len(example) > 200 else example)
        
        # Existing tests
        if self.existing_tests:
            sections.append("\n=== EXISTING TESTS (for reference) ===")
            sections.append(self.existing_tests[:500] + "..." if len(self.existing_tests) > 500 else self.existing_tests)
        
        # Git history
        if self.git_history:
            sections.append("\n=== RECENT CHANGES ===")
            for change in self.git_history[:3]:  # Last 3 changes
                sections.append(f"  - {change.get('date', 'unknown')}: {change.get('message', 'no message')}")
        
        return "\n".join(sections)
    
    def get_quality_score(self) -> float:
        """Calculate context quality score (0-1)."""
        score = 0.0
        max_score = 10.0
        
        # Core (mandatory)
        if self.target_code:
            score += 2.0
        if self.dependencies:
            score += 1.0
        
        # Related code
        if self.related_functions:
            score += 2.0
        
        # Usage analysis
        if self.callers:
            score += 1.5
        if self.callees:
            score += 0.5
        if self.usage_examples:
            score += 1.0
        
        # Historical
        if self.existing_tests:
            score += 1.0
        if self.git_history:
            score += 0.5
        
        # Documentation
        if self.docstring:
            score += 0.5
        
        return min(score / max_score, 1.0)


class ContextAssembler:
    """
    Assembles comprehensive context for test generation.
    
    This class FORCES context gathering rather than relying on LLM tools.
    It ensures consistent, high-quality context for every test generation request.
    """
    
    def __init__(
        self,
        embedding_store: Optional[CodeEmbeddingStore] = None,
        git_integration: Optional[GitIntegration] = None,
        tracking_db: Optional[TestTrackingDB] = None,
        symbol_index: Optional[SymbolIndex] = None,
        hybrid_search: Optional[HybridSearchEngine] = None
    ):
        """
        Initialize context assembler.
        
        Args:
            embedding_store: For semantic code search
            git_integration: For git history
            tracking_db: For test tracking
            symbol_index: For exact symbol lookups
            hybrid_search: For combined search (NEW)
        """
        self.embedding_store = embedding_store or CodeEmbeddingStore()
        self.git = git_integration
        self.tracking_db = tracking_db
        self.symbol_index = symbol_index
        self.hybrid_search = hybrid_search
        
        console.print("[green]✓[/green] Context Assembler initialized")
    
    def assemble(
        self,
        source_code: str,
        file_path: str,
        function_name: Optional[str] = None,
        max_related: int = 5
    ) -> AssembledContext:
        """
        Assemble comprehensive context for test generation.
        
        This method ALWAYS gathers context, regardless of LLM behavior.
        
        Args:
            source_code: The code to test
            file_path: Path to source file
            function_name: Specific function to test (optional)
            max_related: Maximum number of related code examples
            
        Returns:
            AssembledContext with all available information
        """
        console.print(f"\n[cyan]🔍 Assembling context for {Path(file_path).name}[/cyan]")
        
        # Extract function name if not provided
        if not function_name:
            function_name = self._extract_function_name(source_code)
        
        # 1. Parse target code
        metadata = self._extract_code_metadata(source_code, function_name)
        console.print(f"  [dim]→ Target: {function_name or 'module'}[/dim]")
        
        # 2. Get related code (semantic search)
        related = self._get_related_code(source_code, function_name, max_related)
        console.print(f"  [dim]→ Found {len(related)} related functions[/dim]")
        
        # 3. Extract dependencies
        dependencies = self._extract_dependencies(source_code)
        console.print(f"  [dim]→ Found {len(dependencies)} dependencies[/dim]")
        
        # 4. Analyze call graph
        callers = self._find_callers(function_name, file_path) if function_name else []
        callees = self._find_callees(source_code)
        console.print(f"  [dim]→ Callers: {len(callers)}, Callees: {len(callees)}[/dim]")
        
        # 5. Find usage examples
        usage_examples = self._find_usage_examples(function_name) if function_name else []
        console.print(f"  [dim]→ Found {len(usage_examples)} usage examples[/dim]")
        
        # 6. Get git history
        git_history = self._get_git_history(file_path)
        console.print(f"  [dim]→ Found {len(git_history)} recent changes[/dim]")
        
        # 7. Find existing tests
        existing_tests = self._find_existing_tests(file_path, function_name)
        if existing_tests:
            console.print(f"  [dim]→ Found existing tests ({len(existing_tests)} chars)[/dim]")
        
        # Assemble context
        context = AssembledContext(
            target_code=source_code,
            file_path=file_path,
            function_name=function_name,
            related_functions=related,
            dependencies=dependencies,
            callers=callers,
            callees=callees,
            usage_examples=usage_examples,
            git_history=git_history,
            existing_tests=existing_tests,
            complexity_score=metadata['complexity'],
            total_lines=metadata['lines'],
            docstring=metadata['docstring']
        )
        
        # Report quality
        quality = context.get_quality_score()
        quality_label = "excellent" if quality > 0.8 else "good" if quality > 0.6 else "fair"
        console.print(f"  [green]✓[/green] Context quality: {quality:.2f} ({quality_label})")
        
        return context
    
    def _extract_function_name(self, code: str) -> Optional[str]:
        """Extract first function name from code."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except:
            pass
        return None
    
    def _extract_code_metadata(self, code: str, function_name: Optional[str]) -> Dict:
        """Extract metadata from code."""
        metadata = {
            'complexity': 1,
            'lines': len(code.split('\n')),
            'docstring': None
        }
        
        try:
            tree = ast.parse(code)
            
            # Find target function
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not function_name or node.name == function_name:
                        # Get docstring
                        docstring = ast.get_docstring(node)
                        if docstring:
                            metadata['docstring'] = docstring
                        
                        # Calculate cyclomatic complexity
                        metadata['complexity'] = self._calculate_complexity(node)
                        break
        except:
            pass
        
        return metadata
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _get_related_code(
        self,
        source_code: str,
        function_name: Optional[str],
        max_results: int
    ) -> List[Dict[str, str]]:
        """Get semantically related code."""
        related = []
        
        # Use hybrid search if available (BEST - combines all methods)
        if self.hybrid_search and function_name:
            try:
                search_results = self.hybrid_search.search(
                    query=function_name,
                    n_results=max_results * 2,
                    search_types=['exact', 'semantic', 'keyword']
                )
                
                for result in search_results:
                    # Skip self
                    if result.name == function_name:
                        continue
                    
                    related.append({
                        'name': result.name,
                        'file_path': result.file_path,
                        'code': result.code_snippet,
                        'score': result.relevance_score
                    })
                    
                    if len(related) >= max_results:
                        break
                
                return related
            except Exception as e:
                console.print(f"[yellow]⚠️  Hybrid search failed: {e}[/yellow]")
        
        # Fallback: semantic search only
        try:
            results = self.embedding_store.search_similar_code(
                query=source_code,
                n_results=max_results * 2,
                use_reranking=True
            )
            
            for result in results:
                name = result.get('metadata', {}).get('name', 'unknown')
                # Skip self
                if function_name and name == function_name:
                    continue
                
                related.append({
                    'name': name,
                    'file_path': result.get('metadata', {}).get('file_path', 'unknown'),
                    'code': result.get('content', ''),
                    'score': result.get('rerank_score', result.get('distance', 0))
                })
                
                if len(related) >= max_results:
                    break
            
            return related
        except Exception as e:
            console.print(f"[yellow]⚠️  Related code search failed: {e}[/yellow]")
            return []
    
    def _extract_dependencies(self, code: str) -> List[str]:
        """Extract import statements."""
        dependencies = []
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        dependencies.append(f"from {module} import {alias.name}")
        except:
            pass
        
        return dependencies
    
    def _find_callers(self, function_name: str, file_path: str) -> List[Dict[str, str]]:
        """Find functions that call the target function."""
        callers = []
        
        # Use symbol index if available (FAST O(1) lookup)
        if self.symbol_index:
            try:
                call_sites = self.symbol_index.find_callers(function_name)
                for site in call_sites[:10]:  # Top 10 callers
                    callers.append({
                        'caller_name': site.caller,
                        'file': site.location.file_path,
                        'line': site.location.line_number,
                        'code_snippet': f"{site.caller} calls {function_name}"
                    })
                return callers
            except Exception as e:
                console.print(f"[yellow]⚠️  Symbol index search failed: {e}[/yellow]")
        
        # Fallback: search in embedding store
        try:
            query = f"{function_name}("
            results = self.embedding_store.search_similar_code(
                query=query,
                n_results=10,
                use_reranking=False
            )
            
            for result in results:
                content = result.get('content', '')
                if f"{function_name}(" in content:
                    metadata = result.get('metadata', {})
                    callers.append({
                        'caller_name': metadata.get('name', 'unknown'),
                        'file': metadata.get('file_path', 'unknown'),
                        'code_snippet': content[:100]
                    })
        except Exception as e:
            console.print(f"[yellow]⚠️  Embedding search failed: {e}[/yellow]")
        
        return callers
    
    def _find_callees(self, code: str) -> List[str]:
        """Find functions called by the target code."""
        callees = []
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Simple function call
                    if isinstance(node.func, ast.Name):
                        callees.append(node.func.id)
                    # Method call
                    elif isinstance(node.func, ast.Attribute):
                        callees.append(node.func.attr)
        except:
            pass
        
        return list(set(callees))  # Deduplicate
    
    def _find_usage_examples(self, function_name: str) -> List[str]:
        """Find example usages of the function."""
        examples = []
        
        try:
            # Search for code that uses this function
            query = f"example using {function_name}"
            results = self.embedding_store.search_similar_code(
                query=query,
                n_results=5,
                use_reranking=True
            )
            
            for result in results:
                content = result.get('content', '')
                if function_name in content:
                    examples.append(content)
        except:
            pass
        
        return examples
    
    def _get_git_history(self, file_path: str, limit: int = 5) -> List[Dict[str, str]]:
        """Get recent git history for the file."""
        if not self.git:
            return []
        
        try:
            # Get file history
            history = []
            repo = self.git.repo
            
            for commit in repo.iter_commits(paths=file_path, max_count=limit):
                history.append({
                    'hash': commit.hexsha[:8],
                    'date': commit.committed_datetime.strftime('%Y-%m-%d'),
                    'author': commit.author.name,
                    'message': commit.message.split('\n')[0][:100]
                })
            
            return history
        except Exception as e:
            console.print(f"[yellow]⚠️  Git history failed: {e}[/yellow]")
            return []
    
    def _find_existing_tests(
        self,
        file_path: str,
        function_name: Optional[str]
    ) -> Optional[str]:
        """Find existing test code."""
        try:
            # Convert source path to test path
            source_path = Path(file_path)
            
            # Try common test path patterns
            test_patterns = [
                source_path.parent.parent / "tests" / f"test_{source_path.name}",
                source_path.parent / "tests" / f"test_{source_path.name}",
                source_path.parent / f"test_{source_path.name}"
            ]
            
            for test_path in test_patterns:
                if test_path.exists():
                    test_content = test_path.read_text(encoding='utf-8')
                    
                    # If looking for specific function, try to extract relevant tests
                    if function_name:
                        # Look for test functions that match
                        relevant_tests = []
                        tree = ast.parse(test_content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                if f"test_{function_name}" in node.name:
                                    # Extract this test function
                                    start = node.lineno - 1
                                    end = node.end_lineno
                                    lines = test_content.split('\n')
                                    relevant_tests.append('\n'.join(lines[start:end]))
                        
                        if relevant_tests:
                            return '\n\n'.join(relevant_tests)
                    
                    return test_content
        except:
            pass
        
        return None


def create_context_assembler(
    embedding_store: Optional[CodeEmbeddingStore] = None,
    git_integration: Optional[GitIntegration] = None,
    tracking_db: Optional[TestTrackingDB] = None,
    symbol_index: Optional[SymbolIndex] = None,
    hybrid_search: Optional[HybridSearchEngine] = None
) -> ContextAssembler:
    """Factory function to create context assembler."""
    return ContextAssembler(
        embedding_store,
        git_integration,
        tracking_db,
        symbol_index,
        hybrid_search
    )


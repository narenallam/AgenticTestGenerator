"""
RAG (Retrieval-Augmented Generation) system for code context retrieval.

This module combines code embeddings, git changes, and semantic search
to provide relevant context for test generation.
"""

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from rich.console import Console

from config.settings import settings
from src.code_embeddings import CodeEmbeddingStore
from src.git_integration import GitIntegration

console = Console()


class CodeContext(BaseModel):
    """
    Represents retrieved code context for test generation.
    
    Attributes:
        target_code: The main code to test
        related_code: Related code chunks for context
        dependencies: External dependencies
        existing_tests: Existing test code if available
        metadata: Additional metadata
    """
    
    target_code: str = Field(..., description="Main code to generate tests for")
    related_code: List[str] = Field(
        default_factory=list,
        description="Related code chunks"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Dependencies"
    )
    existing_tests: Optional[str] = Field(
        default=None,
        description="Existing tests"
    )
    metadata: Dict = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class RAGRetriever:
    """
    RAG retrieval system for intelligent code context gathering.
    
    This class orchestrates:
    - Git change detection
    - Semantic code search
    - Context aggregation
    - Relevance ranking
    """
    
    def __init__(
        self,
        embedding_store: Optional[CodeEmbeddingStore] = None,
        git_integration: Optional[GitIntegration] = None,
        source_dir: Optional[Path] = None
    ) -> None:
        """
        Initialize RAG retriever.
        
        Args:
            embedding_store: Code embedding store instance
            git_integration: Git integration instance
            source_dir: Source code directory
        """
        self.embedding_store = embedding_store or CodeEmbeddingStore()
        # Always use settings.source_code_dir if not explicitly provided
        self.source_dir = source_dir if source_dir is not None else settings.source_code_dir
        self.git_integration = git_integration or GitIntegration(repo_path=self.source_dir)
        
        console.print("[green]✓[/green] RAG Retriever initialized")
    
    def get_context_for_changed_code(
        self,
        max_context_size: int = 3000
    ) -> List[CodeContext]:
        """
        Get comprehensive context for all changed code since last commit.
        
        Args:
            max_context_size: Maximum size of context in characters
            
        Returns:
            List of CodeContext objects for each changed function
            
        Example:
            >>> retriever = RAGRetriever()
            >>> contexts = retriever.get_context_for_changed_code()
            >>> for ctx in contexts:
            ...     print(f"Testing: {ctx.metadata['function_name']}")
        """
        contexts = []
        
        # Get changed files from git
        changed_files = self.git_integration.get_changed_files_since_last_commit()
        
        console.print(f"[cyan]Found {len(changed_files)} changed files[/cyan]")
        
        for file_change in changed_files:
            file_path = file_change.file_path
            
            # Get new functions in this file
            new_functions = self.git_integration.get_new_functions_since_commit(
                file_path,
                base_commit="HEAD"
            )
            
            for func_info in new_functions:
                context = self._build_context_for_function(
                    file_path,
                    func_info,
                    max_context_size
                )
                if context:
                    contexts.append(context)
        
        return contexts
    
    def get_context_for_function(
        self,
        file_path: str,
        function_name: str,
        max_context_size: int = 3000
    ) -> Optional[CodeContext]:
        """
        Get context for a specific function.
        
        Args:
            file_path: Path to the file containing the function
            function_name: Name of the function
            max_context_size: Maximum context size
            
        Returns:
            CodeContext object or None if not found
        """
        # Search for the function in embeddings
        results = self.embedding_store.search_similar_code(
            query=function_name,
            n_results=1,
            filter_dict={"file_path": file_path, "name": function_name}
        )
        
        if not results:
            return None
        
        func_info = {
            'name': function_name,
            'source': results[0]['content']
        }
        
        return self._build_context_for_function(
            file_path,
            func_info,
            max_context_size
        )
    
    def _build_context_for_function(
        self,
        file_path: str,
        func_info: Dict,
        max_context_size: int
    ) -> Optional[CodeContext]:
        """
        Build comprehensive context for a function.
        
        Args:
            file_path: File containing the function
            func_info: Dictionary with function information
            max_context_size: Maximum context size
            
        Returns:
            CodeContext object
        """
        function_name = func_info['name']
        target_code = func_info['source']
        
        # Search for related code using semantic similarity with reranking
        # Fetch more results (10) and let reranker select best 5
        related_results = self.embedding_store.search_similar_code(
            query=target_code,
            n_results=5,
            use_reranking=True  # Enable reranking for better context
        )
        
        # Filter out the target function itself and limit size
        related_code = []
        current_size = len(target_code)
        
        for result in related_results:
            if result['metadata'].get('name') != function_name:
                code = result['content']
                if current_size + len(code) < max_context_size:
                    related_code.append(code)
                    current_size += len(code)
                    
                    # Log rerank score if available
                    if 'rerank_score' in result:
                        console.print(
                            f"  → Added context: {result['metadata'].get('name', 'unknown')} "
                            f"(score: {result['rerank_score']:.2f})"
                        )
                else:
                    break
        
        # Extract dependencies
        dependencies = self._extract_dependencies(target_code)
        
        # Look for existing tests
        existing_tests = self._find_existing_tests(file_path, function_name)
        
        return CodeContext(
            target_code=target_code,
            related_code=related_code,
            dependencies=dependencies,
            existing_tests=existing_tests,
            metadata={
                'file_path': file_path,
                'function_name': function_name,
                'has_existing_tests': existing_tests is not None
            }
        )
    
    def _extract_dependencies(self, code: str) -> List[str]:
        """Extract dependencies from code."""
        import ast
        
        dependencies = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append(node.module)
        except SyntaxError:
            pass
        
        return list(set(dependencies))
    
    def _find_existing_tests(
        self,
        file_path: str,
        function_name: str
    ) -> Optional[str]:
        """
        Look for existing test files for the given function.
        
        Args:
            file_path: Source file path
            function_name: Function name
            
        Returns:
            Existing test code or None
        """
        # Convert source path to test path
        test_file_patterns = [
            f"test_{Path(file_path).name}",
            f"{Path(file_path).stem}_test.py"
        ]
        
        test_dirs = [
            self.source_dir.parent / "tests",
            self.source_dir / "tests"
        ]
        
        for test_dir in test_dirs:
            if not test_dir.exists():
                continue
            
            for pattern in test_file_patterns:
                test_file = test_dir / pattern
                if test_file.exists():
                    try:
                        with open(test_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Check if function tests exist
                        if f"test_{function_name}" in content:
                            return content
                    except Exception:
                        pass
        
        return None
    
    def update_embeddings_for_changes(self) -> int:
        """
        Update embeddings for all changed files.
        
        Returns:
            Number of chunks updated
        """
        changed_files = self.git_integration.get_changed_files_since_last_commit()
        total_updated = 0
        
        for file_change in changed_files:
            if file_change.change_type == 'deleted':
                continue
            
            file_path = self.git_integration.repo_root / file_change.file_path
            if file_path.exists():
                count = self.embedding_store.update_file_embeddings(
                    file_path,
                    self.source_dir
                )
                total_updated += count
        
        console.print(f"[green]✓[/green] Updated {total_updated} embeddings")
        return total_updated


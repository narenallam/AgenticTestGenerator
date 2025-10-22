"""
Code embedding system for semantic code search and retrieval.

This module handles code indexing, embedding generation, and storage
using ChromaDB as the vector database.
"""

import ast
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from config.settings import settings
from src.embedding_providers import get_chroma_embedding_function

console = Console()


class CodeChunk(BaseModel):
    """
    Represents a chunk of code with metadata.
    
    Attributes:
        id: Unique identifier for the chunk
        file_path: Path to the source file
        chunk_type: Type of code (function, class, module)
        name: Name of the code entity
        content: Actual code content
        start_line: Starting line number
        end_line: Ending line number
        docstring: Docstring if available
        dependencies: List of imports/dependencies
    """
    
    id: str = Field(..., description="Unique chunk identifier")
    file_path: str = Field(..., description="Source file path")
    chunk_type: str = Field(..., description="Type of code chunk")
    name: str = Field(..., description="Code entity name")
    content: str = Field(..., description="Code content")
    start_line: int = Field(..., description="Start line number")
    end_line: int = Field(..., description="End line number")
    docstring: Optional[str] = Field(default=None, description="Docstring")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies")


class CodeEmbeddingStore:
    """
    Manages code embeddings using ChromaDB for semantic code search.
    
    This class handles:
    - Code chunking and parsing
    - Embedding generation
    - Vector storage and retrieval
    - Semantic search
    """
    
    def __init__(
        self,
        collection_name: str = "code_embeddings",
        persist_directory: Optional[Path] = None
    ) -> None:
        """
        Initialize the code embedding store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistence (uses settings if None)
        """
        self.collection_name = collection_name
        self.persist_dir = persist_directory or settings.chroma_persist_dir
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding function based on provider
        self.embedding_function = get_chroma_embedding_function(
            provider=settings.llm_provider
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "Code embeddings for semantic search"}
        )
        
        console.print(f"[green]✓[/green] Initialized embedding store: {collection_name}")
    
    def index_codebase(
        self,
        source_dir: Path,
        file_extensions: Optional[Set[str]] = None,
        force_reindex: bool = False
    ) -> int:
        """
        Index entire codebase by creating embeddings for all code chunks.
        
        Args:
            source_dir: Root directory of source code
            file_extensions: File extensions to index (default: {'.py'})
            force_reindex: Whether to clear existing index
            
        Returns:
            Number of code chunks indexed
            
        Example:
            >>> store = CodeEmbeddingStore()
            >>> count = store.index_codebase(Path("./src"))
            >>> print(f"Indexed {count} code chunks")
        """
        if file_extensions is None:
            file_extensions = {'.py'}
        
        if force_reindex:
            console.print("[yellow]Clearing existing index...[/yellow]")
            self.clear_index()
        
        # Find all source files
        source_files = []
        for ext in file_extensions:
            source_files.extend(source_dir.rglob(f"*{ext}"))
        
        total_chunks = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[cyan]Indexing {len(source_files)} files...",
                total=len(source_files)
            )
            
            for file_path in source_files:
                try:
                    chunks = self._parse_file(file_path, source_dir)
                    if chunks:
                        self._add_chunks(chunks)
                        total_chunks += len(chunks)
                except Exception as e:
                    console.print(f"[red]Error parsing {file_path}: {e}[/red]")
                
                progress.advance(task)
        
        console.print(f"[green]✓[/green] Indexed {total_chunks} code chunks from {len(source_files)} files")
        return total_chunks
    
    def search_similar_code(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None,
        use_reranking: bool = True
    ) -> List[Dict]:
        """
        Search for semantically similar code chunks.
        
        Args:
            query: Search query (code snippet or description)
            n_results: Number of results to return
            filter_dict: Metadata filter for search
            use_reranking: Whether to use reranker for better results
            
        Returns:
            List of similar code chunks with metadata
            
        Example:
            >>> store = CodeEmbeddingStore()
            >>> results = store.search_similar_code("function to calculate factorial")
            >>> for result in results:
            ...     print(f"{result['name']}: {result['distance']}")
        """
        # Fetch more results for reranking
        fetch_count = n_results * 4 if use_reranking else n_results
        
        results = self.collection.query(
            query_texts=[query],
            n_results=fetch_count,
            where=filter_dict
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        # Apply reranking if enabled
        if use_reranking and len(formatted_results) > n_results:
            formatted_results = self._rerank_results(query, formatted_results, n_results)
        
        return formatted_results[:n_results]
    
    def _rerank_results(
        self,
        query: str,
        results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Rerank search results using Qwen3-Reranker.
        
        Args:
            query: Original search query
            results: Initial search results
            top_k: Number of top results to return
            
        Returns:
            Reranked results
        """
        try:
            from src.reranker import create_reranker
            
            reranker = create_reranker(provider=settings.llm_provider, top_k=top_k)
            
            # Extract documents and metadata
            documents = [r['content'] for r in results]
            metadata = [r['metadata'] for r in results]
            
            # Rerank
            reranked = reranker.rerank(query, documents, metadata, top_k=top_k)
            
            # Reconstruct result format with reranked order
            reranked_results = []
            for rr in reranked:
                original = results[rr.index]
                # Add reranker score
                original['rerank_score'] = rr.score
                reranked_results.append(original)
            
            return reranked_results
            
        except Exception as e:
            console.print(f"[yellow]Warning: Reranking failed, using original order: {e}[/yellow]")
            return results
    
    def get_code_context(
        self,
        file_path: str,
        function_names: List[str],
        include_dependencies: bool = True
    ) -> str:
        """
        Get comprehensive code context for specific functions.
        
        Args:
            file_path: Path to the source file
            function_names: List of function names to retrieve
            include_dependencies: Whether to include imported dependencies
            
        Returns:
            Concatenated code context as string
        """
        context_parts = []
        
        for func_name in function_names:
            # Search for the specific function
            results = self.collection.query(
                query_texts=[func_name],
                n_results=1,
                where={
                    "file_path": file_path,
                    "name": func_name
                }
            )
            
            if results['documents']:
                context_parts.append(results['documents'][0][0])
                
                if include_dependencies:
                    # Get related code
                    metadata = results['metadatas'][0][0]
                    if 'dependencies' in metadata:
                        deps = metadata['dependencies']
                        dep_results = self.search_similar_code(
                            " ".join(deps),
                            n_results=3
                        )
                        context_parts.extend([r['content'] for r in dep_results])
        
        return "\n\n".join(context_parts)
    
    def update_file_embeddings(
        self,
        file_path: Path,
        source_dir: Path
    ) -> int:
        """
        Update embeddings for a specific file.
        
        Args:
            file_path: Path to the file to update
            source_dir: Root source directory
            
        Returns:
            Number of chunks updated
        """
        # Remove existing chunks for this file
        rel_path = str(file_path.relative_to(source_dir))
        try:
            self.collection.delete(where={"file_path": rel_path})
        except Exception:
            pass
        
        # Re-parse and add chunks
        chunks = self._parse_file(file_path, source_dir)
        if chunks:
            self._add_chunks(chunks)
            return len(chunks)
        return 0
    
    def clear_index(self) -> None:
        """Clear all embeddings from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "Code embeddings for semantic search"}
        )
    
    def _parse_file(
        self,
        file_path: Path,
        source_dir: Path
    ) -> List[CodeChunk]:
        """
        Parse a Python file into code chunks.
        
        Args:
            file_path: Path to the file
            source_dir: Root source directory
            
        Returns:
            List of CodeChunk objects
        """
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            lines = content.split('\n')
            rel_path = str(file_path.relative_to(source_dir))
            
            # Extract imports
            imports = self._extract_imports(tree)
            
            # Parse functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    chunk = self._create_function_chunk(
                        node, lines, rel_path, imports
                    )
                    if chunk:
                        chunks.append(chunk)
                
                elif isinstance(node, ast.ClassDef):
                    chunk = self._create_class_chunk(
                        node, lines, rel_path, imports
                    )
                    if chunk:
                        chunks.append(chunk)
        
        except (SyntaxError, FileNotFoundError) as e:
            console.print(f"[yellow]Warning: Could not parse {file_path}: {e}[/yellow]")
        
        return chunks
    
    def _create_function_chunk(
        self,
        node: ast.FunctionDef,
        lines: List[str],
        file_path: str,
        imports: List[str]
    ) -> Optional[CodeChunk]:
        """Create a CodeChunk from a function AST node."""
        try:
            # Get function source
            start_line = node.lineno - 1
            end_line = node.end_lineno if node.end_lineno else start_line + 1
            content = '\n'.join(lines[start_line:end_line])
            
            # Extract docstring
            docstring = ast.get_docstring(node)
            
            # Generate unique ID
            chunk_id = self._generate_chunk_id(file_path, node.name, start_line)
            
            return CodeChunk(
                id=chunk_id,
                file_path=file_path,
                chunk_type="function",
                name=node.name,
                content=content,
                start_line=start_line + 1,
                end_line=end_line,
                docstring=docstring,
                dependencies=imports
            )
        except Exception:
            return None
    
    def _create_class_chunk(
        self,
        node: ast.ClassDef,
        lines: List[str],
        file_path: str,
        imports: List[str]
    ) -> Optional[CodeChunk]:
        """Create a CodeChunk from a class AST node."""
        try:
            start_line = node.lineno - 1
            end_line = node.end_lineno if node.end_lineno else start_line + 1
            content = '\n'.join(lines[start_line:end_line])
            
            docstring = ast.get_docstring(node)
            chunk_id = self._generate_chunk_id(file_path, node.name, start_line)
            
            return CodeChunk(
                id=chunk_id,
                file_path=file_path,
                chunk_type="class",
                name=node.name,
                content=content,
                start_line=start_line + 1,
                end_line=end_line,
                docstring=docstring,
                dependencies=imports
            )
        except Exception:
            return None
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports
    
    def _add_chunks(self, chunks: List[CodeChunk]) -> None:
        """Add code chunks to the collection."""
        if not chunks:
            return
        
        self.collection.add(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[{
                "file_path": chunk.file_path,
                "chunk_type": chunk.chunk_type,
                "name": chunk.name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "docstring": chunk.docstring or "",
                "dependencies": ",".join(chunk.dependencies)
            } for chunk in chunks]
        )
    
    def _generate_chunk_id(
        self,
        file_path: str,
        name: str,
        line: int
    ) -> str:
        """Generate a unique ID for a code chunk."""
        content = f"{file_path}:{name}:{line}"
        return hashlib.md5(content.encode()).hexdigest()


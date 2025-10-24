"""
Hybrid Search - Combines exact, semantic, and keyword search.

This module provides a unified search interface that combines:
1. Exact matching (symbol index)
2. Semantic similarity (vector embeddings)
3. Keyword matching (BM25/TF-IDF)

Results are fused and reranked for optimal relevance.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
from rich.console import Console

from src.symbol_index import SymbolIndex, create_symbol_index
from src.code_embeddings import CodeEmbeddingStore
from src.reranker import BaseReranker, create_reranker

console = Console()


@dataclass
class SearchResult:
    """Unified search result."""
    name: str
    file_path: str
    line_number: int
    code_snippet: str
    relevance_score: float
    source: str  # 'exact', 'semantic', 'keyword', or 'hybrid'
    metadata: Dict


class HybridSearchEngine:
    """
    Hybrid search engine combining multiple search strategies.
    
    Uses Reciprocal Rank Fusion (RRF) to merge results from:
    - Exact symbol matching (O(1) lookup)
    - Semantic vector search (embedding similarity)
    - Keyword search (BM25-style scoring)
    """
    
    def __init__(
        self,
        symbol_index: Optional[SymbolIndex] = None,
        embedding_store: Optional[CodeEmbeddingStore] = None,
        use_reranking: bool = True
    ):
        """
        Initialize hybrid search engine.
        
        Args:
            symbol_index: Symbol index for exact lookups
            embedding_store: Vector store for semantic search
            use_reranking: Whether to use cross-encoder reranking
        """
        self.symbol_index = symbol_index or create_symbol_index()
        self.embedding_store = embedding_store or CodeEmbeddingStore()
        self.use_reranking = use_reranking
        self.reranker = create_reranker() if use_reranking else None
        
        console.print("[green]✓[/green] Hybrid Search Engine initialized")
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        search_types: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query (can be natural language or function name)
            n_results: Number of results to return
            search_types: Which search types to use (default: all)
                          Options: ['exact', 'semantic', 'keyword']
            
        Returns:
            List of search results, sorted by relevance
        """
        if search_types is None:
            search_types = ['exact', 'semantic', 'keyword']
        
        console.print(f"\n[cyan]🔍 Hybrid Search: {query}[/cyan]")
        
        all_results = []
        
        # 1. Exact matching (if query looks like a symbol name)
        if 'exact' in search_types:
            exact_results = self._exact_search(query)
            console.print(
                f"  [dim]→ Exact: {len(exact_results)} results[/dim]"
            )
            all_results.extend(exact_results)
        
        # 2. Semantic search
        if 'semantic' in search_types:
            semantic_results = self._semantic_search(query, n_results * 2)
            console.print(
                f"  [dim]→ Semantic: {len(semantic_results)} results[/dim]"
            )
            all_results.extend(semantic_results)
        
        # 3. Keyword search (simple TF-IDF style)
        if 'keyword' in search_types:
            keyword_results = self._keyword_search(query, n_results * 2)
            console.print(
                f"  [dim]→ Keyword: {len(keyword_results)} results[/dim]"
            )
            all_results.extend(keyword_results)
        
        # 4. Fuse results using Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(all_results)
        
        # 5. Optional reranking
        if self.use_reranking and self.reranker and len(fused_results) > n_results:
            console.print(f"  [dim]→ Reranking top {len(fused_results)}...[/dim]")
            fused_results = self._rerank_results(query, fused_results)
        
        # Return top-n
        final_results = fused_results[:n_results]
        console.print(
            f"  [green]✓[/green] Returned {len(final_results)} results"
        )
        
        return final_results
    
    def _exact_search(self, query: str) -> List[SearchResult]:
        """Search using exact symbol matching."""
        results = []
        
        # Try function lookup
        functions = self.symbol_index.find_function(query)
        for func in functions:
            results.append(SearchResult(
                name=func.name,
                file_path=func.location.file_path,
                line_number=func.location.line_number,
                code_snippet=func.signature,
                relevance_score=1.0,  # Exact match gets highest score
                source='exact',
                metadata={
                    'type': 'function',
                    'docstring': func.docstring,
                    'complexity': func.complexity,
                    'is_async': func.is_async
                }
            ))
        
        # Try class lookup
        classes = self.symbol_index.find_class(query)
        for cls in classes:
            results.append(SearchResult(
                name=cls.name,
                file_path=cls.location.file_path,
                line_number=cls.location.line_number,
                code_snippet=f"class {cls.name}({', '.join(cls.bases)})",
                relevance_score=1.0,
                source='exact',
                metadata={
                    'type': 'class',
                    'docstring': cls.docstring,
                    'methods': cls.methods,
                    'bases': cls.bases
                }
            ))
        
        return results
    
    def _semantic_search(
        self,
        query: str,
        n_results: int
    ) -> List[SearchResult]:
        """Search using semantic similarity (embeddings)."""
        results = []
        
        try:
            # Use embedding store's semantic search
            embedding_results = self.embedding_store.search_similar_code(
                query=query,
                n_results=n_results,
                use_reranking=False  # We'll rerank at the end
            )
            
            for result in embedding_results:
                metadata = result.get('metadata', {})
                
                # Calculate relevance from distance
                # ChromaDB returns cosine distance (0=identical, 2=opposite)
                # Convert to similarity score (0-1)
                distance = result.get('distance', 1.0)
                relevance = max(0.0, 1.0 - (distance / 2.0))
                
                results.append(SearchResult(
                    name=metadata.get('name', 'unknown'),
                    file_path=metadata.get('file_path', 'unknown'),
                    line_number=metadata.get('start_line', 0),
                    code_snippet=result.get('content', '')[:200],
                    relevance_score=relevance,
                    source='semantic',
                    metadata={
                        'type': metadata.get('chunk_type', 'unknown'),
                        'docstring': metadata.get('docstring', ''),
                        'distance': distance
                    }
                ))
        except Exception as e:
            console.print(f"[yellow]⚠️  Semantic search failed: {e}[/yellow]")
        
        return results
    
    def _keyword_search(
        self,
        query: str,
        n_results: int
    ) -> List[SearchResult]:
        """Search using keyword matching (simple TF-IDF style)."""
        results = []
        
        # Simple implementation: search for query terms in symbol names
        query_terms = set(query.lower().split())
        
        # Search in function names
        for name, symbols in self.symbol_index.functions.items():
            name_lower = name.lower()
            # Calculate match score
            matches = sum(1 for term in query_terms if term in name_lower)
            if matches > 0:
                score = matches / len(query_terms)  # Simple scoring
                
                for symbol in symbols:
                    results.append(SearchResult(
                        name=symbol.name,
                        file_path=symbol.location.file_path,
                        line_number=symbol.location.line_number,
                        code_snippet=symbol.signature,
                        relevance_score=score,
                        source='keyword',
                        metadata={
                            'type': 'function',
                            'docstring': symbol.docstring,
                            'match_score': score
                        }
                    ))
        
        # Search in class names
        for name, symbols in self.symbol_index.classes.items():
            name_lower = name.lower()
            matches = sum(1 for term in query_terms if term in name_lower)
            if matches > 0:
                score = matches / len(query_terms)
                
                for symbol in symbols:
                    results.append(SearchResult(
                        name=symbol.name,
                        file_path=symbol.location.file_path,
                        line_number=symbol.location.line_number,
                        code_snippet=f"class {symbol.name}",
                        relevance_score=score,
                        source='keyword',
                        metadata={
                            'type': 'class',
                            'docstring': symbol.docstring,
                            'match_score': score
                        }
                    ))
        
        # Sort by score and return top-n
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:n_results]
    
    def _reciprocal_rank_fusion(
        self,
        results: List[SearchResult],
        k: int = 60
    ) -> List[SearchResult]:
        """
        Fuse results using Reciprocal Rank Fusion (RRF).
        
        RRF formula: RRF(d) = Σ 1 / (k + rank(d))
        
        Args:
            results: All results from different search methods
            k: Constant for RRF (default 60, as in literature)
            
        Returns:
            Fused and re-ranked results
        """
        # Group by (name, file_path) to merge duplicates
        result_map: Dict[tuple, Dict] = {}
        
        # Track ranks from each source
        source_ranks: Dict[str, Dict[tuple, int]] = {
            'exact': {},
            'semantic': {},
            'keyword': {}
        }
        
        # Assign ranks within each source
        for source in ['exact', 'semantic', 'keyword']:
            source_results = [r for r in results if r.source == source]
            source_results.sort(
                key=lambda x: x.relevance_score,
                reverse=True
            )
            for rank, result in enumerate(source_results, 1):
                key = (result.name, result.file_path)
                source_ranks[source][key] = rank
        
        # Calculate RRF scores
        for result in results:
            key = (result.name, result.file_path)
            
            if key not in result_map:
                result_map[key] = {
                    'result': result,
                    'rrf_score': 0.0,
                    'sources': []
                }
            
            # Add RRF contribution from this source
            rank = source_ranks[result.source].get(key, 1000)
            rrf_contribution = 1.0 / (k + rank)
            result_map[key]['rrf_score'] += rrf_contribution
            result_map[key]['sources'].append(result.source)
        
        # Create final results with RRF scores
        fused = []
        for key, data in result_map.items():
            result = data['result']
            # Update relevance score to RRF score
            result.relevance_score = data['rrf_score']
            result.source = 'hybrid' if len(data['sources']) > 1 else result.source
            result.metadata['sources'] = data['sources']
            fused.append(result)
        
        # Sort by RRF score
        fused.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return fused
    
    def _rerank_results(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """Rerank results using cross-encoder model."""
        if not self.reranker:
            return results
        
        try:
            # Prepare documents for reranking
            documents = [r.code_snippet for r in results]
            
            # Rerank
            reranked_indices = self.reranker.rerank(
                query=query,
                documents=documents,
                top_n=len(results)
            )
            
            # Reorder results
            reranked_results = []
            for idx_info in reranked_indices:
                original_idx = idx_info.get('index', 0)
                if original_idx < len(results):
                    result = results[original_idx]
                    result.relevance_score = idx_info.get('score', 0.0)
                    result.metadata['rerank_score'] = idx_info.get('score', 0.0)
                    reranked_results.append(result)
            
            return reranked_results
        except Exception as e:
            console.print(f"[yellow]⚠️  Reranking failed: {e}[/yellow]")
            return results
    
    def find_with_context(
        self,
        query: str,
        include_callers: bool = True,
        include_callees: bool = True,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Find code with full context (including call graph).
        
        Args:
            query: Search query
            include_callers: Include calling functions
            include_callees: Include called functions
            n_results: Number of results
            
        Returns:
            List of results with context
        """
        # First do hybrid search
        results = self.search(query, n_results)
        
        # Enhance with call graph context
        enhanced = []
        for result in results:
            context = {
                'result': result,
                'callers': [],
                'callees': []
            }
            
            # Add callers
            if include_callers:
                callers = self.symbol_index.find_callers(result.name)
                context['callers'] = [
                    {
                        'name': c.caller,
                        'file': c.location.file_path,
                        'line': c.location.line_number
                    }
                    for c in callers[:5]  # Top 5 callers
                ]
            
            # Add callees
            if include_callees:
                callees = self.symbol_index.find_callees(result.name)
                context['callees'] = [
                    {
                        'name': c.callee,
                        'file': c.location.file_path,
                        'line': c.location.line_number
                    }
                    for c in callees[:10]  # Top 10 callees
                ]
            
            enhanced.append(context)
        
        return enhanced


def create_hybrid_search(
    symbol_index: Optional[SymbolIndex] = None,
    embedding_store: Optional[CodeEmbeddingStore] = None,
    use_reranking: bool = True
) -> HybridSearchEngine:
    """Factory function to create hybrid search engine."""
    return HybridSearchEngine(symbol_index, embedding_store, use_reranking)


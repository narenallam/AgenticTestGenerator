"""
Code reranker with multi-provider support.

This module provides reranking functionality to improve the relevance
of retrieved code chunks from semantic search.
Supports Ollama native reranker and LLM-based reranking for other providers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field
from rich.console import Console

from config.settings import settings

console = Console()


class RerankResult(BaseModel):
    """
    Result from reranking operation.
    
    Attributes:
        index: Original index in the input list
        document: The document text
        score: Relevance score from reranker
        metadata: Optional metadata from original search
    """
    
    index: int = Field(..., description="Original index")
    document: str = Field(..., description="Document text")
    score: float = Field(..., description="Relevance score")
    metadata: dict = Field(default_factory=dict, description="Metadata")


class BaseReranker(ABC):
    """Base class for reranking implementations."""
    
    def __init__(self, top_k: int = 5) -> None:
        """Initialize reranker."""
        self.top_k = top_k
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        metadata: List[dict] = None,
        top_k: int = None
    ) -> List[RerankResult]:
        """Rerank documents based on query relevance."""
        pass
    
    def _fallback_score(self, query: str, document: str) -> float:
        """
        Fallback scoring using simple text similarity.
        
        Args:
            query: Query text
            document: Document text
            
        Returns:
            Simple similarity score
        """
        query_lower = query.lower()
        doc_lower = document.lower()
        
        query_words = set(query_lower.split())
        doc_words = set(doc_lower.split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words & doc_words
        union = query_words | doc_words
        
        return len(intersection) / len(union) if union else 0.0


class OllamaReranker(BaseReranker):
    """Ollama-based reranker using Qwen3-Reranker."""
    
    def __init__(self, model_name: str = None, top_k: int = 5) -> None:
        """Initialize Ollama reranker."""
        super().__init__(top_k)
        self.model_name = model_name or settings.ollama_reranker_model
        
        import ollama
        if settings.ollama_api_key:
            ollama.api_key = settings.ollama_api_key
        if settings.ollama_base_url:
            ollama.base_url = settings.ollama_base_url
        
        console.print(f"[green]✓[/green] Ollama reranker: {self.model_name}")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        metadata: List[dict] = None,
        top_k: int = None
    ) -> List[RerankResult]:
        """Rerank documents using Ollama reranker."""
        if not documents:
            return []
        
        k = top_k or self.top_k
        metadata = metadata or [{} for _ in documents]
        
        scores = self._calculate_scores(query, documents)
        
        results = [
            RerankResult(
                index=i,
                document=doc,
                score=score,
                metadata=meta
            )
            for i, (doc, score, meta) in enumerate(zip(documents, scores, metadata))
        ]
        
        results.sort(key=lambda x: x.score, reverse=True)
        
        console.print(f"[cyan]Reranked {len(documents)} → {min(k, len(results))} results[/cyan]")
        
        return results[:k]
    
    def _calculate_scores(self, query: str, documents: List[str]) -> List[float]:
        """Calculate relevance scores."""
        scores = []
        
        for doc in documents:
            try:
                score = self._score_pair(query, doc)
                scores.append(score)
            except Exception as e:
                console.print(f"[yellow]Warning: Reranking error: {e}[/yellow]")
                scores.append(self._fallback_score(query, doc))
        
        return scores
    
    def _score_pair(self, query: str, document: str) -> float:
        """Score a query-document pair using Ollama reranker."""
        import ollama
        
        prompt = f"""Given the query and document below, provide a relevance score between 0 and 1, where:
- 0 means completely irrelevant
- 1 means highly relevant

Query: {query}

Document: {document[:500]}

Relevance score (0.0-1.0):"""
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.0,
                    "num_predict": 10
                }
            )
            
            score_text = response['response'].strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))
            
        except (ValueError, KeyError) as e:
            console.print(f"[yellow]Score parsing error: {e}, using fallback[/yellow]")
            return self._fallback_score(query, document)


class LLMReranker(BaseReranker):
    """
    LLM-based reranker for providers without native reranking.
    
    Uses the generation model to score relevance.
    """
    
    def __init__(self, llm_provider=None, top_k: int = 5) -> None:
        """Initialize LLM reranker."""
        super().__init__(top_k)
        
        if llm_provider is None:
            from src.llm_providers import get_default_provider
            llm_provider = get_default_provider()
        
        self.llm_provider = llm_provider
        
        console.print(
            f"[green]✓[/green] LLM reranker: "
            f"{self.llm_provider.provider_name}/{self.llm_provider.model_name}"
        )
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        metadata: List[dict] = None,
        top_k: int = None
    ) -> List[RerankResult]:
        """Rerank documents using LLM."""
        if not documents:
            return []
        
        k = top_k or self.top_k
        metadata = metadata or [{} for _ in documents]
        
        scores = []
        for doc in documents:
            try:
                score = self._score_with_llm(query, doc)
                scores.append(score)
            except Exception as e:
                console.print(f"[yellow]Reranking error: {e}[/yellow]")
                scores.append(self._fallback_score(query, doc))
        
        results = [
            RerankResult(
                index=i,
                document=doc,
                score=score,
                metadata=meta
            )
            for i, (doc, score, meta) in enumerate(zip(documents, scores, metadata))
        ]
        
        results.sort(key=lambda x: x.score, reverse=True)
        
        console.print(f"[cyan]Reranked {len(documents)} → {min(k, len(results))} results[/cyan]")
        
        return results[:k]
    
    def _score_with_llm(self, query: str, document: str) -> float:
        """Score using LLM generation."""
        prompt = f"""Rate the relevance of this document to the query on a scale from 0.0 to 1.0:

Query: {query}

Document: {document[:500]}

Provide ONLY a number between 0.0 and 1.0 representing relevance:"""
        
        response = self.llm_provider.generate(
            prompt=prompt,
            system="You are a relevance scoring expert. Return ONLY a number.",
            temperature=0.0,
            max_tokens=10
        )
        
        try:
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return self._fallback_score(query, document)


class RerankerFactory:
    """Factory for creating reranker instances."""
    
    @staticmethod
    def create(
        provider: Optional[str] = None,
        llm_provider=None,
        top_k: int = 5
    ) -> BaseReranker:
        """
        Create a reranker instance.
        
        Args:
            provider: Provider name ('ollama', 'openai', 'gemini')
            llm_provider: LLM provider instance (for LLM-based reranking)
            top_k: Number of top results
            
        Returns:
            Reranker instance
        """
        provider = provider or settings.llm_provider
        provider = provider.lower()
        
        # Ollama has native reranker
        if provider == 'ollama':
            return OllamaReranker(top_k=top_k)
        
        # Others use LLM-based reranking
        else:
            if llm_provider is None:
                from src.llm_providers import get_llm_provider
                llm_provider = get_llm_provider(provider)
            
            return LLMReranker(llm_provider=llm_provider, top_k=top_k)


def create_reranker(
    provider: Optional[str] = None,
    top_k: int = 5
) -> BaseReranker:
    """
    Factory function to create a reranker instance.
    
    Args:
        provider: Provider name
        top_k: Number of top results to return
        
    Returns:
        Initialized reranker
    """
    return RerankerFactory.create(provider=provider, top_k=top_k)

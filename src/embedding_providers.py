"""
Embedding provider abstraction for different LLM services.

This module provides unified embedding functionality across
Ollama, OpenAI, and Google providers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from rich.console import Console

from config.settings import settings

console = Console()


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Initialize embedding provider.
        
        Args:
            model_name: Model name (uses provider default if None)
        """
        self.model_name = model_name
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embedding provider."""
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize Ollama embeddings."""
        super().__init__(model_name or settings.ollama_embedding_model)
        
        import ollama
        if settings.ollama_api_key:
            ollama.api_key = settings.ollama_api_key
        if settings.ollama_base_url:
            ollama.base_url = settings.ollama_base_url
        
        console.print(f"[green]✓[/green] Ollama embeddings: {self.model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using Ollama."""
        import ollama
        
        embeddings = []
        for text in texts:
            try:
                response = ollama.embeddings(
                    model=self.model_name,
                    prompt=text
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                console.print(f"[yellow]Warning: Embedding error: {e}[/yellow]")
                embeddings.append([0.0] * self.dimension)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query using Ollama."""
        import ollama
        
        try:
            response = ollama.embeddings(
                model=self.model_name,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            console.print(f"[yellow]Warning: Query embedding error: {e}[/yellow]")
            return [0.0] * self.dimension
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        # qwen3-embedding:8b typically returns 768-dim vectors
        return 768


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize OpenAI embeddings."""
        super().__init__(model_name or settings.openai_embedding_model)
        
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        console.print(f"[green]✓[/green] OpenAI embeddings: {self.model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using OpenAI."""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )
        
        try:
            response = client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            console.print(f"[red]OpenAI embedding error: {e}[/red]")
            return [[0.0] * self.dimension for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query using OpenAI."""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )
        
        try:
            response = client.embeddings.create(
                model=self.model_name,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            console.print(f"[red]OpenAI query embedding error: {e}[/red]")
            return [0.0] * self.dimension
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        # text-embedding-3-large: 3072, text-embedding-3-small: 1536
        if "large" in self.model_name:
            return 3072
        elif "small" in self.model_name or "3" in self.model_name:
            return 1536
        else:
            return 1536  # Default


class GoogleEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gemini embedding provider."""
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize Google embeddings."""
        super().__init__(model_name or settings.google_embedding_model)
        
        if not settings.google_api_key:
            raise ValueError("Google API key not configured")
        
        import google.generativeai as genai
        genai.configure(api_key=settings.google_api_key)
        
        console.print(f"[green]✓[/green] Google embeddings: {self.model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using Google."""
        import google.generativeai as genai
        
        embeddings = []
        for text in texts:
            try:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                console.print(f"[yellow]Warning: Google embedding error: {e}[/yellow]")
                embeddings.append([0.0] * self.dimension)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query using Google."""
        import google.generativeai as genai
        
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            console.print(f"[red]Google query embedding error: {e}[/red]")
            return [0.0] * self.dimension
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        # text-embedding-004: 768 dimensions
        return 768


class ChromaEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    ChromaDB-compatible embedding function wrapper.
    
    This wraps our embedding providers for use with ChromaDB.
    """
    
    def __init__(self, provider: BaseEmbeddingProvider) -> None:
        """
        Initialize ChromaDB embedding function.
        
        Args:
            provider: Embedding provider instance
        """
        self.provider = provider
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for ChromaDB.
        
        Args:
            input: List of documents
            
        Returns:
            List of embeddings
        """
        return self.provider.embed_documents(input)


class EmbeddingProviderFactory:
    """Factory for creating embedding provider instances."""
    
    @staticmethod
    def create(
        provider: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> BaseEmbeddingProvider:
        """
        Create an embedding provider.
        
        Args:
            provider: Provider name ('ollama', 'openai', 'gemini')
            model_name: Model name (uses provider default if None)
            
        Returns:
            Embedding provider instance
        """
        provider = provider or settings.llm_provider
        provider = provider.lower()
        
        providers = {
            'ollama': OllamaEmbeddingProvider,
            'openai': OpenAIEmbeddingProvider,
            'gemini': GoogleEmbeddingProvider,
        }
        
        if provider not in providers:
            console.print(f"[yellow]Unknown provider {provider}, using Ollama[/yellow]")
            return OllamaEmbeddingProvider(model_name)
        
        try:
            return providers[provider](model_name)
        except Exception as e:
            console.print(f"[red]Failed to initialize {provider} embeddings: {e}[/red]")
            console.print("[yellow]Falling back to Ollama embeddings[/yellow]")
            return OllamaEmbeddingProvider(model_name)


def get_embedding_provider(
    provider: Optional[str] = None,
    model_name: Optional[str] = None
) -> BaseEmbeddingProvider:
    """
    Get an embedding provider instance.
    
    Args:
        provider: Provider name
        model_name: Model name
        
    Returns:
        Embedding provider
    """
    return EmbeddingProviderFactory.create(provider, model_name)


def get_chroma_embedding_function(
    provider: Optional[str] = None
) -> ChromaEmbeddingFunction:
    """
    Get ChromaDB-compatible embedding function.
    
    Args:
        provider: Provider name
        
    Returns:
        ChromaDB embedding function
    """
    embedding_provider = get_embedding_provider(provider)
    return ChromaEmbeddingFunction(embedding_provider)


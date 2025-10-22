"""
Ollama-based embedding functions for ChromaDB.

This module provides custom embedding functions using Ollama models
for code semantic search.
"""

from typing import List

import ollama
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from config.settings import settings


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Custom embedding function using Ollama models.
    
    This class integrates Ollama embedding models with ChromaDB
    for semantic code search.
    """
    
    def __init__(
        self,
        model_name: str = "qwen3-embedding:8b",
        api_key: str = None,
        base_url: str = None
    ) -> None:
        """
        Initialize Ollama embedding function.
        
        Args:
            model_name: Ollama embedding model name
            api_key: Ollama API key
            base_url: Ollama API base URL
        """
        self.model_name = model_name
        self.api_key = api_key or settings.ollama_api_key
        self.base_url = base_url or settings.ollama_base_url
        
        # Configure ollama client
        if self.api_key:
            ollama.api_key = self.api_key
        if self.base_url:
            ollama.base_url = self.base_url
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for documents.
        
        Args:
            input: List of documents to embed
            
        Returns:
            List of embeddings
        """
        embeddings = []
        
        for text in input:
            try:
                # Generate embedding using Ollama
                response = ollama.embeddings(
                    model=self.model_name,
                    prompt=text
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                # Fallback to zero vector on error
                print(f"Error generating embedding: {e}")
                embeddings.append([0.0] * 768)  # Default embedding size
        
        return embeddings


"""
LLM Provider abstraction layer.

This module provides a unified interface for different LLM providers
(Ollama, OpenAI, Gemini) with automatic provider selection based on configuration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, Field
from rich.console import Console

from config.settings import settings

console = Console()


class LLMResponse(BaseModel):
    """
    Standardized LLM response format.
    
    Attributes:
        content: Generated text content
        model: Model name used
        usage: Token usage information
        metadata: Additional metadata
    """
    
    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model used")
    usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All providers must implement these methods to ensure consistent interface.
    """
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Initialize the provider.
        
        Args:
            model_name: Model name to use (uses default if None)
        """
        self.model_name = model_name
        self.client: Optional[BaseChatModel] = None
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """
        Generate text from prompt.
        
        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific arguments
            
        Returns:
            LLMResponse with generated content
        """
        pass
    
    @abstractmethod
    def get_langchain_model(self) -> BaseChatModel:
        """
        Get LangChain-compatible model instance.
        
        Returns:
            BaseChatModel instance for use with LangChain/LangGraph
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get provider name."""
        pass


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider."""
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize Ollama provider."""
        super().__init__(model_name or settings.ollama_model)
        self.base_url = settings.ollama_base_url
        self.api_key = settings.ollama_api_key
        
        console.print(
            f"[green]✓[/green] Ollama provider: {self.model_name} @ {self.base_url}"
        )
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate using Ollama."""
        import ollama
        
        # Configure client
        if self.api_key:
            ollama.api_key = self.api_key
        if self.base_url:
            ollama.base_url = self.base_url
        
        try:
            options = {
                "temperature": temperature,
            }
            if max_tokens:
                options["num_predict"] = max_tokens
            
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                system=system,
                options=options,
                **kwargs
            )
            
            return LLMResponse(
                content=response['response'],
                model=self.model_name,
                usage={
                    'prompt_tokens': response.get('prompt_eval_count', 0),
                    'completion_tokens': response.get('eval_count', 0),
                    'total_tokens': response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
                },
                metadata={
                    'model': response.get('model', self.model_name),
                    'total_duration': response.get('total_duration'),
                    'load_duration': response.get('load_duration'),
                    'prompt_eval_duration': response.get('prompt_eval_duration'),
                    'eval_duration': response.get('eval_duration'),
                    'prompt_eval_count': response.get('prompt_eval_count'),
                    'eval_count': response.get('eval_count')
                }
            )
        
        except Exception as e:
            console.print(f"[red]Ollama error: {e}[/red]")
            raise
    
    def get_langchain_model(self) -> BaseChatModel:
        """Get LangChain Ollama model."""
        from langchain_ollama import ChatOllama
        
        if not self.client:
            self.client = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=0.7
            )
        
        return self.client
    
    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return "ollama"


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize OpenAI provider."""
        super().__init__(model_name or settings.openai_model)
        
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url
        
        console.print(f"[green]✓[/green] OpenAI provider: {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate using OpenAI."""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                metadata={'finish_reason': response.choices[0].finish_reason}
            )
        
        except Exception as e:
            console.print(f"[red]OpenAI error: {e}[/red]")
            raise
    
    def get_langchain_model(self) -> BaseChatModel:
        """Get LangChain OpenAI model."""
        from langchain_openai import ChatOpenAI
        
        if not self.client:
            kwargs = {
                'model': self.model_name,
                'api_key': self.api_key,
                'temperature': 0.7
            }
            if self.base_url:
                kwargs['base_url'] = self.base_url
            
            self.client = ChatOpenAI(**kwargs)
        
        return self.client
    
    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return "openai"


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize Gemini provider."""
        super().__init__(model_name or settings.google_model)
        
        if not settings.google_api_key:
            raise ValueError("Google API key not configured")
        
        self.api_key = settings.google_api_key
        
        console.print(f"[green]✓[/green] Gemini provider: {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate using Gemini."""
        import google.generativeai as genai
        
        genai.configure(api_key=self.api_key)
        
        # Combine system and user prompt for Gemini
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        
        try:
            model = genai.GenerativeModel(self.model_name)
            
            generation_config = {
                'temperature': temperature,
            }
            if max_tokens:
                generation_config['max_output_tokens'] = max_tokens
            
            response = model.generate_content(
                full_prompt,
                generation_config=generation_config,
                **kwargs
            )
            
            return LLMResponse(
                content=response.text,
                model=self.model_name,
                usage={
                    'prompt_tokens': response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                    'completion_tokens': response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                    'total_tokens': response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                },
                metadata={}
            )
        
        except Exception as e:
            console.print(f"[red]Gemini error: {e}[/red]")
            raise
    
    def get_langchain_model(self) -> BaseChatModel:
        """Get LangChain Gemini model."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        if not self.client:
            self.client = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.7
            )
        
        return self.client
    
    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return "gemini"


class LLMProviderFactory:
    """
    Factory for creating LLM provider instances.
    
    Automatically selects the appropriate provider based on configuration.
    """
    
    @staticmethod
    def create(
        provider: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> BaseLLMProvider:
        """
        Create an LLM provider instance.
        
        Args:
            provider: Provider name ('ollama', 'openai', 'gemini')
                     Uses settings.llm_provider if None
            model_name: Model name (uses provider default if None)
            
        Returns:
            Configured LLM provider instance
            
        Raises:
            ValueError: If provider is unknown
            
        Example:
            >>> # Use default (Ollama)
            >>> llm = LLMProviderFactory.create()
            >>> 
            >>> # Use OpenAI
            >>> llm = LLMProviderFactory.create("openai", "gpt-4")
            >>> 
            >>> # Use from environment
            >>> # export LLM_PROVIDER=gemini
            >>> llm = LLMProviderFactory.create()
        """
        provider = provider or settings.llm_provider
        provider = provider.lower()
        
        providers = {
            'ollama': OllamaProvider,
            'openai': OpenAIProvider,
            'gemini': GeminiProvider,
        }
        
        if provider not in providers:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available: {', '.join(providers.keys())}"
            )
        
        console.print(f"[cyan]Initializing LLM provider: {provider}[/cyan]")
        
        try:
            return providers[provider](model_name)
        except Exception as e:
            console.print(f"[red]Failed to initialize {provider}: {e}[/red]")
            console.print("[yellow]Falling back to Ollama...[/yellow]")
            return OllamaProvider(model_name)


def get_llm_provider(
    provider: Optional[str] = None,
    model_name: Optional[str] = None
) -> BaseLLMProvider:
    """
    Convenience function to get an LLM provider.
    
    Args:
        provider: Provider name
        model_name: Model name
        
    Returns:
        LLM provider instance
    """
    return LLMProviderFactory.create(provider, model_name)


# Singleton instance for default provider
_default_provider: Optional[BaseLLMProvider] = None


def get_default_provider() -> BaseLLMProvider:
    """
    Get or create the default LLM provider.
    
    Returns:
        Default provider instance (cached)
    """
    global _default_provider
    
    if _default_provider is None:
        _default_provider = LLMProviderFactory.create()
    
    return _default_provider


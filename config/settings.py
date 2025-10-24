"""
Configuration settings for the Agentic Unit Test Generator.

This module provides centralized configuration management using Pydantic Settings.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with validation and environment variable support.
    
    Attributes:
        ollama_api_key: API key for Ollama Cloud
        ollama_base_url: Base URL for Ollama API
        ollama_model: Model name to use for test generation
        google_api_key: Optional Google AI API key
        chroma_persist_dir: Directory for ChromaDB persistence
        embedding_model: Model name for code embeddings
        source_code_dir: Root directory for source code
        test_output_dir: Directory for generated tests
        max_context_tokens: Maximum context size for LLM
        max_iterations: Maximum ReAct agent iterations
        sandbox_timeout: Timeout for sandbox test execution
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # LLM Provider Configuration
    llm_provider: str = Field(
        default="ollama",
        description="LLM provider: 'ollama', 'openai', or 'gemini'"
    )
    
    # Ollama Configuration
    ollama_api_key: Optional[str] = Field(
        default=None,
        description="Ollama API key (optional for local)"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    ollama_model: str = Field(
        default="qwen3-coder:30b",
        description="Ollama model name for code generation"
    )
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    openai_model: str = Field(
        default="gpt-4-turbo-preview",
        description="OpenAI model name"
    )
    openai_base_url: Optional[str] = Field(
        default=None,
        description="OpenAI API base URL (for compatible APIs)"
    )
    
    # Google Gemini Configuration
    google_api_key: Optional[str] = Field(
        default=None,
        description="Google AI API key"
    )
    google_model: str = Field(
        default="gemini-1.5-pro",
        description="Google Gemini model name"
    )
    
    # Vector Store Configuration
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma_db"),
        description="ChromaDB persistence directory"
    )
    
    # Embedding Models (provider-specific)
    ollama_embedding_model: str = Field(
        default="qwen3-embedding:8b",
        description="Ollama embedding model"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model"
    )
    google_embedding_model: str = Field(
        default="models/text-embedding-004",
        description="Google embedding model"
    )
    
    # Reranker Models (provider-specific)
    ollama_reranker_model: str = Field(
        default="dengcao/Qwen3-Reranker-8B:Q8_0",
        description="Ollama reranker model"
    )
    openai_reranker_model: Optional[str] = Field(
        default=None,
        description="OpenAI doesn't have native reranker (uses generation)"
    )
    google_reranker_model: Optional[str] = Field(
        default=None,
        description="Google doesn't have native reranker (uses generation)"
    )
    
    # Legacy fields for backward compatibility
    @property
    def embedding_model(self) -> str | None:
        """Get embedding model based on current provider."""
        provider_map = {
            "ollama": self.ollama_embedding_model,
            "openai": self.openai_embedding_model,
            "gemini": self.google_embedding_model
        }
        return provider_map.get(self.llm_provider, self.ollama_embedding_model)
    
    @property
    def reranker_model(self) -> str | None:
        """Get reranker model based on current provider."""
        provider_map = {
            "ollama": self.ollama_reranker_model,
            "openai": self.openai_reranker_model,
            "gemini": self.google_reranker_model
        }
        return provider_map.get(self.llm_provider, self.ollama_reranker_model)
    
    # Code Analysis Configuration
    # NOTE: Both directories MUST be set in .env file - no defaults
    source_code_dir: Optional[Path] = Field(
        default=None,
        description="Source code directory (MUST be set in .env)"
    )
    test_output_dir: Optional[Path] = Field(
        default=None,
        description="Test output directory (MUST be set in .env)"
    )
    max_context_tokens: int = Field(
        default=8000,
        description="Maximum context tokens for LLM"
    )
    
    def __init__(self, **data):
        """Initialize settings with validation."""
        super().__init__(**data)
        
        # Validate that SOURCE_CODE_DIR is set
        if self.source_code_dir is None:
            raise ValueError(
                "\n❌ SOURCE_CODE_DIR must be set in your .env file!\n"
                "Example: SOURCE_CODE_DIR=/path/to/your/project/src\n"
                "This tells the system where to find the code to test.\n"
            )
        
        # Validate that TEST_OUTPUT_DIR is set
        if self.test_output_dir is None:
            raise ValueError(
                "\n❌ TEST_OUTPUT_DIR must be set in your .env file!\n"
                "Example: TEST_OUTPUT_DIR=/path/to/your/project/tests\n"
                "This tells the system where to save generated tests.\n"
            )
        
        # Ensure they're Path objects
        if not isinstance(self.source_code_dir, Path):
            self.source_code_dir = Path(self.source_code_dir)
        if not isinstance(self.test_output_dir, Path):
            self.test_output_dir = Path(self.test_output_dir)
        
        # Create test output directory if it doesn't exist
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Agent Configuration
    max_iterations: int = Field(
        default=5,
        description="Maximum ReAct agent iterations"
    )
    sandbox_timeout: int = Field(
        default=30,
        description="Sandbox execution timeout in seconds"
    )
    
    # LLM Parameters by Agent Role
    # Planner: Low temperature for structured, deterministic planning
    planner_temperature: float = Field(
        default=0.2,
        description="Temperature for Planner agent (low for deterministic planning)"
    )
    planner_max_tokens: int = Field(
        default=512,
        description="Max tokens for Planner output (structured task lists)"
    )
    
    # Coder/Test Generator: Low-medium temperature for code generation
    coder_temperature: float = Field(
        default=0.3,
        description="Temperature for Test Generator (balanced for code quality)"
    )
    coder_max_tokens: int = Field(
        default=2048,
        description="Max tokens for Test Generator (full test suites)"
    )
    
    # Critic: Very low temperature for consistent reviews
    critic_temperature: float = Field(
        default=0.1,
        description="Temperature for Critic agent (very low for consistent reviews)"
    )
    critic_max_tokens: int = Field(
        default=1024,
        description="Max tokens for Critic output (reviews and feedback)"
    )
    
    # ReAct Agent: Low-medium temperature for reasoning
    react_temperature: float = Field(
        default=0.4,
        description="Temperature for ReAct agent reasoning"
    )
    react_max_tokens: int = Field(
        default=1536,
        description="Max tokens for ReAct agent responses"
    )
    
    # Coverage Generator: Low temperature for targeted generation
    coverage_temperature: float = Field(
        default=0.3,
        description="Temperature for coverage-driven test generation"
    )
    coverage_max_tokens: int = Field(
        default=1536,
        description="Max tokens for coverage-driven tests"
    )
    
    def __init__(self, **kwargs):
        """Initialize settings and create necessary directories."""
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.test_output_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


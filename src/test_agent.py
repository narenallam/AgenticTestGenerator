"""
Test Generation Agent - Compatibility Wrapper for LangGraph Orchestrator.

This module provides backward compatibility by wrapping the LangGraph-based
orchestrator. It maintains the same API as before while leveraging the modern
LangGraph 1.0 architecture internally.

ARCHITECTURE DECISION:
---------------------
This project uses LangGraph's create_react_agent for orchestration instead of
implementing a custom ReAct loop. This provides:
- Built-in state management
- Better tool integration
- Robust error handling
- Production-ready patterns
- 66% code reduction

The orchestrator (src/orchestrator.py) is the primary implementation.
This file exists for API compatibility with existing code and examples.
"""

from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console

from config.settings import settings
from src.orchestrator import (
    create_test_generation_orchestrator,
    TestGenerationConfig
)
from src.rag_retrieval import CodeContext
from src.llm_providers import BaseLLMProvider

console = Console()


# Legacy compatibility - these are no longer used internally
# The orchestrator handles state management via LangGraph


class TestGenerationAgent:
    """
    Test Generation Agent - Compatibility wrapper for LangGraph orchestrator.
    
    This class maintains the same API as before but delegates to the LangGraph-based
    orchestrator internally. This ensures backward compatibility with existing code
    while using the modern LangGraph 1.0 architecture.
    
    For new code, prefer using create_test_generation_orchestrator() directly.
    """
    
    def __init__(
        self,
        retriever=None,  # Kept for API compatibility but not used
        max_iterations: int = 5,
        llm_provider: Optional[BaseLLMProvider] = None
    ) -> None:
        """
        Initialize test generation agent.
        
        Args:
            retriever: Deprecated, kept for compatibility
            max_iterations: Maximum refinement iterations
            llm_provider: Deprecated, uses settings instead
        """
        # Create LangGraph orchestrator with proper configuration
        config = TestGenerationConfig(
            max_iterations=max_iterations,
            enable_hitl=False,
            enable_summarization=False,
            enable_pii_redaction=False
        )
        
        self.orchestrator = create_test_generation_orchestrator(config=config)
        self.max_iterations = max_iterations
        
        console.print(
            f"[green]✓[/green] Test Agent initialized (using LangGraph 1.0 orchestrator)"
        )
    
    def generate_tests(
        self,
        target_code: str,
        file_path: Optional[str] = None,
        test_type=None  # Deprecated parameter, kept for compatibility
    ) -> str:
        """
        Generate tests for target code using LangGraph orchestrator.
        
        Args:
            target_code: Code to generate tests for
            file_path: Optional file path for context
            test_type: Deprecated, kept for compatibility
            
        Returns:
            Generated test code
            
        Example:
            >>> agent = TestGenerationAgent()
            >>> tests = agent.generate_tests("def add(a, b): return a + b")
            >>> print(tests)
        """
        console.print(f"[cyan]Starting test generation (using LangGraph 1.0)[/cyan]")
        
        # Delegate to orchestrator
        return self.orchestrator.generate_tests(
            target_code=target_code,
            file_path=file_path or "",
            function_name=None,
            context=""
        )
    
# All internal methods removed - orchestrator handles this via LangGraph
    
    def generate_batch_tests(
        self,
        code_changes: List[CodeContext]
    ) -> Dict[str, str]:
        """
        Generate tests for multiple code changes.
        
        Args:
            code_changes: List of code contexts
            
        Returns:
            Dictionary mapping function names to test code
        """
        results = {}
        
        for context in code_changes:
            func_name = context.metadata.get('function_name', 'unknown')
            console.print(f"\n[bold cyan]Generating tests for: {func_name}[/bold cyan]")
            
            tests = self.generate_tests(
                target_code=context.target_code,
                file_path=context.metadata.get('file_path')
            )
            
            if tests:
                results[func_name] = tests
        
        return results


# Factory function for easier instantiation
def create_test_agent(max_iterations: int = 5) -> TestGenerationAgent:
    """
    Create a test generation agent.
    
    Args:
        max_iterations: Maximum iterations for refinement
        
    Returns:
        TestGenerationAgent instance (wraps LangGraph orchestrator)
    """
    return TestGenerationAgent(max_iterations=max_iterations)


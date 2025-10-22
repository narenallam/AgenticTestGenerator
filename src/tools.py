"""
Tool definitions for the orchestrator agent.

This module defines all tools that the orchestrator can dynamically select
and execute based on the current state and requirements.
"""

from typing import Any, Dict, List, Optional

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from src.code_embeddings import CodeEmbeddingStore
from src.git_integration import GitIntegration
from src.rag_retrieval import RAGRetriever
from src.sandbox_executor import SandboxExecutor


# Tool Input Schemas

class SearchCodeInput(BaseModel):
    """Input for searching code semantically."""
    query: str = Field(..., description="Search query for finding relevant code")
    n_results: int = Field(default=5, description="Number of results to return")


class CheckGitChangesInput(BaseModel):
    """Input for checking git changes."""
    file_extensions: Optional[List[str]] = Field(
        default=None,
        description="File extensions to filter (e.g., ['.py'])"
    )


class GetCodeContextInput(BaseModel):
    """Input for retrieving code context."""
    file_path: str = Field(..., description="Path to the source file")
    function_name: str = Field(..., description="Function name to get context for")


class ExecuteTestsInput(BaseModel):
    """Input for executing tests in sandbox."""
    test_code: str = Field(..., description="Test code to execute")
    source_code: Optional[str] = Field(
        default=None,
        description="Source code being tested"
    )


# Tool Implementations

class SearchCodeTool(BaseTool):
    """Tool for semantic code search using embeddings."""
    
    name: str = "search_code"
    description: str = """
    Search for code semantically using embeddings. Use this when you need to:
    - Find similar code patterns
    - Locate related functions or classes
    - Discover relevant context for test generation
    
    Input: search query as natural language or code snippet
    Output: List of relevant code chunks with metadata
    """
    args_schema: type[BaseModel] = SearchCodeInput
    
    embedding_store: CodeEmbeddingStore = Field(default_factory=CodeEmbeddingStore)
    
    def _run(
        self,
        query: str,
        n_results: int = 5,
        **kwargs: Any
    ) -> str:
        """Execute semantic code search."""
        results = self.embedding_store.search_similar_code(
            query=query,
            n_results=n_results,
            use_reranking=True
        )
        
        # Format results for LLM
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"Result {i}:\n"
                f"Function: {result['metadata'].get('name', 'unknown')}\n"
                f"File: {result['metadata'].get('file_path', 'unknown')}\n"
                f"Score: {result.get('rerank_score', 'N/A')}\n"
                f"Code:\n{result['content'][:500]}...\n"
            )
        
        return "\n---\n".join(formatted)


class CheckGitChangesTool(BaseTool):
    """Tool for checking git changes since last commit."""
    
    name: str = "check_git_changes"
    description: str = """
    Check what code has changed since the last git commit. Use this to:
    - Identify new or modified functions
    - Get list of changed files
    - Understand what needs testing
    
    Output: Summary of changes with file paths and function names
    """
    args_schema: type[BaseModel] = CheckGitChangesInput
    
    git_integration: GitIntegration = Field(default_factory=GitIntegration)
    
    def _run(
        self,
        file_extensions: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """Check git changes."""
        if file_extensions is None:
            file_extensions = {'.py'}
        
        changes = self.git_integration.get_changed_files_since_last_commit(
            set(file_extensions)
        )
        
        if not changes:
            return "No changes found since last commit."
        
        # Format changes
        summary = [f"Found {len(changes)} changed file(s):\n"]
        
        for change in changes:
            summary.append(f"\n{change.change_type.upper()}: {change.file_path}")
            
            # Get new functions
            new_functions = self.git_integration.get_new_functions_since_commit(
                change.file_path,
                "HEAD"
            )
            
            if new_functions:
                summary.append(f"  New functions:")
                for func in new_functions:
                    summary.append(f"    - {func['name']} (line {func['lineno']})")
        
        return "\n".join(summary)


class GetCodeContextTool(BaseTool):
    """Tool for retrieving comprehensive code context."""
    
    name: str = "get_code_context"
    description: str = """
    Retrieve comprehensive context for a specific function including:
    - The function's source code
    - Related/similar code
    - Dependencies and imports
    - Existing tests (if any)
    
    Use this before generating tests to get full context.
    
    Input: file_path and function_name
    Output: Complete context bundle for test generation
    """
    args_schema: type[BaseModel] = GetCodeContextInput
    
    retriever: RAGRetriever = Field(default_factory=RAGRetriever)
    
    def _run(
        self,
        file_path: str,
        function_name: str,
        **kwargs: Any
    ) -> str:
        """Get code context."""
        context = self.retriever.get_context_for_function(
            file_path,
            function_name
        )
        
        if not context:
            return f"Could not find context for {function_name} in {file_path}"
        
        # Format context
        output = [
            f"=== Context for {function_name} ===\n",
            "TARGET CODE:",
            context.target_code,
            "\nRELATED CODE:",
        ]
        
        for i, related in enumerate(context.related_code[:3], 1):
            output.append(f"\nRelated {i}:")
            output.append(related[:300] + "...")
        
        output.append(f"\nDEPENDENCIES: {', '.join(context.dependencies)}")
        
        if context.existing_tests:
            output.append("\nEXISTING TESTS FOUND:")
            output.append(context.existing_tests[:500] + "...")
        
        return "\n".join(output)


class ExecuteTestsTool(BaseTool):
    """Tool for executing tests in a sandbox."""
    
    name: str = "execute_tests"
    description: str = """
    Execute test code in a safe sandbox environment. Use this to:
    - Verify generated tests work correctly
    - Check test coverage
    - Identify errors or failures
    
    Input: test_code (required), source_code (optional)
    Output: Test execution results with pass/fail status and errors
    """
    args_schema: type[BaseModel] = ExecuteTestsInput
    
    def _run(
        self,
        test_code: str,
        source_code: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Execute tests in sandbox."""
        with SandboxExecutor() as executor:
            result = executor.execute_tests(
                test_code=test_code,
                source_code=source_code
            )
        
        # Format results
        output = [
            f"=== Test Execution Results ===",
            f"Status: {'✓ PASSED' if result.success else '✗ FAILED'}",
            f"Tests Run: {result.tests_run}",
            f"Tests Passed: {result.tests_passed}",
            f"Tests Failed: {result.tests_failed}",
            f"Execution Time: {result.execution_time:.2f}s",
        ]
        
        if result.coverage:
            output.append(f"Coverage: {result.coverage:.1f}%")
        
        if result.stderr:
            output.append(f"\nERRORS:\n{result.stderr[:500]}")
        
        if result.stdout:
            output.append(f"\nOUTPUT:\n{result.stdout[:500]}")
        
        return "\n".join(output)


class GenerateTestsTool(BaseTool):
    """Tool for generating test code using LLM."""
    
    name: str = "generate_tests"
    description: str = """
    Generate test code for a given function. Use this after gathering context.
    The tool will use enterprise-grade prompts to generate comprehensive tests.
    
    Input: target_code, context (optional), test_type (optional)
    Output: Generated test code
    """
    
    def _run(
        self,
        target_code: str,
        context: str = "",
        test_type: str = "unit",
        **kwargs: Any
    ) -> str:
        """Generate tests using LLM."""
        import ollama
        from src.prompts import PromptTemplates, TestType
        
        # Map test type
        type_map = {
            "unit": TestType.UNIT,
            "functional": TestType.FUNCTIONAL,
            "api": TestType.API,
            "edge_case": TestType.EDGE_CASE
        }
        t_type = type_map.get(test_type.lower(), TestType.UNIT)
        
        # Get prompt
        prompt = PromptTemplates.get_prompt(
            test_type=t_type,
            target_code=target_code,
            context=context
        )
        
        # Generate
        try:
            response = ollama.generate(
                model="qwen3-coder:30b",
                prompt=prompt,
                system=PromptTemplates.get_system_prompt()
            )
            
            generated = response['response']
            
            # Extract code if wrapped
            if '```python' in generated:
                import re
                match = re.search(r'```python\n(.*?)\n```', generated, re.DOTALL)
                if match:
                    generated = match.group(1)
            
            return generated
            
        except Exception as e:
            return f"Error generating tests: {e}"


def get_all_tools() -> List[BaseTool]:
    """
    Get all available tools for the orchestrator.
    
    Returns:
        List of tool instances
    """
    return [
        SearchCodeTool(),
        CheckGitChangesTool(),
        GetCodeContextTool(),
        ExecuteTestsTool(),
        GenerateTestsTool()
    ]


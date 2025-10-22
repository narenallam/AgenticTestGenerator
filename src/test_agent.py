"""
ReAct Agent for intelligent test generation.

This module implements a ReAct (Reasoning + Acting) agent that:
1. Analyzes code to determine test requirements
2. Generates comprehensive test cases
3. Executes tests in a sandbox
4. Refines tests based on execution results
5. Iterates until high-quality tests are produced
"""

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from config.settings import settings
from src.code_embeddings import CodeEmbeddingStore
from src.llm_providers import BaseLLMProvider, get_default_provider
from src.prompts import PromptTemplates, TestType, get_test_type_for_code
from src.rag_retrieval import CodeContext, RAGRetriever
from src.sandbox_executor import SandboxExecutor, TestResult

console = Console()


class AgentAction(str, Enum):
    """Actions the agent can take."""
    
    ANALYZE = "analyze"
    GENERATE = "generate"
    EXECUTE = "execute"
    REFINE = "refine"
    COMPLETE = "complete"


class AgentState(BaseModel):
    """
    Current state of the test generation agent.
    
    Attributes:
        iteration: Current iteration number
        action: Current action
        target_code: Code being tested
        context: Retrieved code context
        generated_tests: Generated test code
        execution_result: Test execution result
        refinement_notes: Notes for refinement
        completed: Whether generation is complete
    """
    
    iteration: int = Field(default=0, description="Iteration number")
    action: AgentAction = Field(default=AgentAction.ANALYZE, description="Current action")
    target_code: str = Field(..., description="Code to test")
    context: Optional[CodeContext] = Field(default=None, description="Code context")
    generated_tests: Optional[str] = Field(default=None, description="Generated tests")
    execution_result: Optional[TestResult] = Field(default=None, description="Execution result")
    refinement_notes: List[str] = Field(default_factory=list, description="Refinement notes")
    completed: bool = Field(default=False, description="Generation complete")
    test_type: TestType = Field(default=TestType.UNIT, description="Test type")


class TestGenerationAgent:
    """
    ReAct agent for automated test generation.
    
    This agent follows a reasoning and acting loop:
    1. Reason about what tests are needed
    2. Act by generating tests
    3. Observe execution results
    4. Refine based on observations
    5. Repeat until success criteria met
    """
    
    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
        max_iterations: int = 5,
        llm_provider: Optional[BaseLLMProvider] = None
    ) -> None:
        """
        Initialize test generation agent.
        
        Args:
            retriever: RAG retriever for code context
            max_iterations: Maximum refinement iterations
            llm_provider: LLM provider (uses default if None)
        """
        self.retriever = retriever or RAGRetriever()
        self.max_iterations = max_iterations
        self.llm_provider = llm_provider or get_default_provider()
        
        console.print(
            f"[green]✓[/green] Test Agent initialized with "
            f"{self.llm_provider.provider_name}/{self.llm_provider.model_name}"
        )
    
    def generate_tests(
        self,
        target_code: str,
        file_path: Optional[str] = None,
        test_type: Optional[TestType] = None
    ) -> str:
        """
        Generate tests for target code using ReAct loop.
        
        Args:
            target_code: Code to generate tests for
            file_path: Optional file path for context
            test_type: Type of tests to generate
            
        Returns:
            Generated test code
            
        Example:
            >>> agent = TestGenerationAgent()
            >>> tests = agent.generate_tests("def add(a, b): return a + b")
            >>> print(tests)
        """
        # Initialize state
        if test_type is None:
            test_type = get_test_type_for_code(target_code)
        
        state = AgentState(
            target_code=target_code,
            test_type=test_type
        )
        
        console.print(f"[cyan]Starting test generation (Type: {test_type.value})[/cyan]")
        
        # ReAct loop
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(
                "[cyan]Generating tests...",
                total=self.max_iterations
            )
            
            while not state.completed and state.iteration < self.max_iterations:
                state = self._react_step(state, file_path)
                progress.update(task, advance=1)
                
                if state.completed:
                    break
        
        if state.generated_tests:
            console.print("[green]✓[/green] Test generation completed successfully")
            return state.generated_tests
        else:
            console.print("[red]✗[/red] Test generation failed")
            return ""
    
    def _react_step(
        self,
        state: AgentState,
        file_path: Optional[str]
    ) -> AgentState:
        """
        Execute one ReAct step.
        
        Args:
            state: Current agent state
            file_path: Optional file path
            
        Returns:
            Updated agent state
        """
        state.iteration += 1
        console.print(f"\n[bold]Iteration {state.iteration}[/bold] - Action: {state.action.value}")
        
        if state.action == AgentAction.ANALYZE:
            return self._analyze(state, file_path)
        elif state.action == AgentAction.GENERATE:
            return self._generate(state)
        elif state.action == AgentAction.EXECUTE:
            return self._execute(state)
        elif state.action == AgentAction.REFINE:
            return self._refine(state)
        
        return state
    
    def _analyze(
        self,
        state: AgentState,
        file_path: Optional[str]
    ) -> AgentState:
        """Analyze code and gather context."""
        console.print("  → Analyzing code and gathering context...")
        
        # Get code context using RAG
        if file_path:
            # Extract function name from target code
            import ast
            try:
                tree = ast.parse(state.target_code)
                func_name = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_name = node.name
                        break
                
                if func_name:
                    state.context = self.retriever.get_context_for_function(
                        file_path,
                        func_name
                    )
            except Exception as e:
                console.print(f"  [yellow]Warning: Context retrieval failed: {e}[/yellow]")
        
        # Move to generation
        state.action = AgentAction.GENERATE
        return state
    
    def _generate(self, state: AgentState) -> AgentState:
        """Generate test code using LLM."""
        console.print("  → Generating test code...")
        
        # Prepare context
        context_str = ""
        if state.context:
            context_str = f"""
Related Code:
{chr(10).join(state.context.related_code[:3])}

Dependencies:
{', '.join(state.context.dependencies)}
"""
        
        # Get prompt
        prompt = PromptTemplates.get_prompt(
            test_type=state.test_type,
            target_code=state.target_code,
            context=context_str
        )
        
        # Generate tests using LLM provider
        try:
            response = self.llm_provider.generate(
                prompt=prompt,
                system=PromptTemplates.get_system_prompt(),
                temperature=settings.coder_temperature,
                max_tokens=settings.coder_max_tokens
            )
            
            generated_code = response.content
            
            # Extract code block if wrapped in markdown
            if '```python' in generated_code:
                generated_code = self._extract_code_block(generated_code)
            
            state.generated_tests = generated_code
            state.action = AgentAction.EXECUTE
            
        except Exception as e:
            console.print(f"  [red]✗ Generation failed: {e}[/red]")
            state.completed = True
        
        return state
    
    def _execute(self, state: AgentState) -> AgentState:
        """Execute generated tests in sandbox."""
        console.print("  → Executing tests in sandbox...")
        
        if not state.generated_tests:
            state.completed = True
            return state
        
        # Execute in sandbox
        with SandboxExecutor(timeout=settings.sandbox_timeout) as executor:
            result = executor.execute_tests(
                test_code=state.generated_tests,
                source_code=state.target_code
            )
            
            state.execution_result = result
        
        # Decide next action based on results
        if result.success:
            console.print(f"  [green]✓ All tests passed ({result.tests_passed}/{result.tests_run})[/green]")
            state.action = AgentAction.COMPLETE
            state.completed = True
        else:
            console.print(f"  [yellow]⚠ Tests failed ({result.tests_failed}/{result.tests_run})[/yellow]")
            state.action = AgentAction.REFINE
            
            # Add refinement notes
            if result.stderr:
                state.refinement_notes.append(f"Error: {result.stderr[:500]}")
        
        return state
    
    def _refine(self, state: AgentState) -> AgentState:
        """Refine tests based on execution results."""
        console.print("  → Refining tests based on execution results...")
        
        if not state.execution_result:
            state.completed = True
            return state
        
        # Prepare refinement prompt
        issues = "\n".join(state.refinement_notes)
        execution_output = f"""
Exit Code: {state.execution_result.exit_code}
Tests Run: {state.execution_result.tests_run}
Tests Passed: {state.execution_result.tests_passed}
Tests Failed: {state.execution_result.tests_failed}

STDERR:
{state.execution_result.stderr[:1000]}

STDOUT:
{state.execution_result.stdout[:1000]}
"""
        
        refinement_prompt = PromptTemplates.get_refinement_prompt(
            original_tests=state.generated_tests,
            execution_results=execution_output,
            issues=issues
        )
        
        # Generate refined tests
        try:
            response = self.llm_provider.generate(
                prompt=refinement_prompt,
                system=PromptTemplates.get_system_prompt(),
                temperature=settings.coder_temperature,
                max_tokens=settings.coder_max_tokens
            )
            
            refined_code = response.content
            
            if '```python' in refined_code:
                refined_code = self._extract_code_block(refined_code)
            
            state.generated_tests = refined_code
            state.action = AgentAction.EXECUTE
            state.refinement_notes.clear()
            
        except Exception as e:
            console.print(f"  [red]✗ Refinement failed: {e}[/red]")
            state.completed = True
        
        return state
    
    def _extract_code_block(self, text: str) -> str:
        """Extract Python code from markdown code block."""
        import re
        
        pattern = r'```python\n(.*?)\n```'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: try to find any code block
        pattern = r'```\n(.*?)\n```'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        return text.strip()
    
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


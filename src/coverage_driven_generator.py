"""
Coverage-driven test generation.

This module implements iterative test generation targeting 90%+ coverage.
Uses AST/CFG analysis to identify untested code paths and generates
additional tests until target coverage is achieved.
"""

from pathlib import Path
from typing import List, Optional, Set

from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from config.settings import settings
from src.ast_analyzer import ASTAnalyzer, ControlFlowGraph
from src.code_quality import create_quality_checker
from src.llm_providers import BaseLLMProvider, get_default_provider
from src.prompts import PromptTemplates, TestType
from src.sandbox_executor import SandboxExecutor

console = Console()


class CoverageTarget(BaseModel):
    """Coverage targets for test generation."""
    
    line_coverage: float = Field(default=90.0, description="Target line coverage %")
    branch_coverage: float = Field(default=85.0, description="Target branch coverage %")
    max_iterations: int = Field(default=5, description="Max iterations")


class CoverageFeedback(BaseModel):
    """Feedback from coverage analysis."""
    
    current_coverage: float = Field(..., description="Current coverage %")
    untested_lines: List[int] = Field(default_factory=list)
    untested_branches: List[str] = Field(default_factory=list)
    missing_edge_cases: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class CoverageDrivenResult(BaseModel):
    """Result of coverage-driven generation."""
    
    final_tests: str = Field(..., description="Final test suite")
    final_coverage: float = Field(..., description="Final coverage %")
    iterations: int = Field(..., description="Iterations performed")
    target_achieved: bool = Field(..., description="Target coverage achieved")
    generation_history: List[dict] = Field(default_factory=list)


class CoverageDrivenGenerator:
    """
    Coverage-driven test generator.
    
    Iteratively generates tests until target coverage (90%+) is achieved.
    Uses AST/CFG analysis to identify gaps and guide generation.
    """
    
    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        target: Optional[CoverageTarget] = None
    ):
        """
        Initialize coverage-driven generator.
        
        Args:
            llm_provider: LLM provider for test generation
            target: Coverage targets
        """
        self.llm_provider = llm_provider or get_default_provider()
        self.target = target or CoverageTarget()
        self.ast_analyzer = ASTAnalyzer()
        self.quality_checker = create_quality_checker()
        
        console.print(
            f"[green]✓[/green] Coverage-driven generator initialized "
            f"(target: {self.target.line_coverage}%)"
        )
    
    def generate_tests(
        self,
        source_code: str,
        file_path: Optional[str] = None
    ) -> CoverageDrivenResult:
        """
        Generate tests with coverage-driven approach.
        
        Args:
            source_code: Source code to test
            file_path: Optional file path
            
        Returns:
            CoverageDrivenResult with final tests and coverage
            
        Example:
            >>> generator = CoverageDrivenGenerator()
            >>> result = generator.generate_tests(source_code)
            >>> print(f"Coverage: {result.final_coverage}%")
        """
        console.print(f"[cyan]Starting coverage-driven test generation[/cyan]")
        console.print(f"[cyan]Target: {self.target.line_coverage}% coverage[/cyan]")
        
        # Analyze source code
        analysis = self.ast_analyzer.analyze(source_code)
        console.print(f"  → Code analysis: {len(analysis.functions)} functions")
        
        # Build CFGs for functions
        cfgs = self._build_cfgs(source_code, analysis.functions)
        
        # Initial test generation
        current_tests = self._generate_initial_tests(source_code, analysis, cfgs)
        
        # Iterative refinement
        history = []
        iteration = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(
                "[cyan]Improving coverage...",
                total=self.target.max_iterations
            )
            
            while iteration < self.target.max_iterations:
                iteration += 1
                console.print(f"\n[bold]Iteration {iteration}[/bold]")
                
                # Execute tests with coverage
                coverage_result = self._measure_coverage(current_tests, source_code)
                current_coverage = coverage_result.coverage or 0.0
                
                console.print(f"  → Current coverage: {current_coverage:.1f}%")
                
                # Record history
                history.append({
                    "iteration": iteration,
                    "coverage": current_coverage,
                    "tests": current_tests
                })
                
                # Check if target achieved
                if current_coverage >= self.target.line_coverage:
                    console.print(f"[green]✓[/green] Target coverage achieved!")
                    return CoverageDrivenResult(
                        final_tests=current_tests,
                        final_coverage=current_coverage,
                        iterations=iteration,
                        target_achieved=True,
                        generation_history=history
                    )
                
                # Identify coverage gaps
                feedback = self._analyze_coverage_gaps(
                    source_code,
                    current_tests,
                    coverage_result,
                    cfgs
                )
                
                console.print(f"  → Untested lines: {len(feedback.untested_lines)}")
                console.print(f"  → Missing branches: {len(feedback.untested_branches)}")
                
                # Generate additional tests
                additional_tests = self._generate_targeted_tests(
                    source_code,
                    current_tests,
                    feedback
                )
                
                # Merge tests
                current_tests = self._merge_tests(current_tests, additional_tests)
                
                progress.update(task, advance=1)
        
        # Final result
        final_result = self._measure_coverage(current_tests, source_code)
        final_coverage = final_result.coverage or 0.0
        
        console.print(f"\n[yellow]Max iterations reached[/yellow]")
        console.print(f"Final coverage: {final_coverage:.1f}%")
        
        return CoverageDrivenResult(
            final_tests=current_tests,
            final_coverage=final_coverage,
            iterations=iteration,
            target_achieved=final_coverage >= self.target.line_coverage,
            generation_history=history
        )
    
    def _build_cfgs(self, source_code: str, functions: List) -> dict:
        """Build CFGs for all functions."""
        import ast
        
        cfgs = {}
        tree = ast.parse(source_code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                try:
                    cfg = self.ast_analyzer.build_cfg(node)
                    cfgs[node.name] = cfg
                except:
                    pass
        
        return cfgs
    
    def _generate_initial_tests(
        self,
        source_code: str,
        analysis,
        cfgs: dict
    ) -> str:
        """Generate initial comprehensive test suite."""
        console.print("  → Generating initial tests...")
        
        # Build context from analysis
        context = f"""
Functions: {', '.join([f.name for f in analysis.functions])}
Total Complexity: {analysis.total_complexity}
Control Flow Branches: {sum(len(cfg.get_branch_nodes()) for cfg in cfgs.values())}
"""
        
        # Generate tests
        prompt = PromptTemplates.get_prompt(
            test_type=TestType.UNIT,
            target_code=source_code,
            context=context
        )
        
        prompt += f"""

IMPORTANT: Generate comprehensive tests covering:
1. All functions: {', '.join([f.name for f in analysis.functions])}
2. All branches and conditions
3. Edge cases (None, empty, boundary values)
4. Exception paths
5. Normal and abnormal flows

Target: {self.target.line_coverage}% code coverage
"""
        
        response = self.llm_provider.generate(
            prompt=prompt,
            system=PromptTemplates.get_system_prompt(),
            temperature=settings.coverage_temperature,
            max_tokens=settings.coverage_max_tokens
        )
        
        test_code = self._extract_code(response.content)
        
        # Format and validate
        report = self.quality_checker.check(test_code)
        return report.formatted_code or test_code
    
    def _measure_coverage(self, test_code: str, source_code: str):
        """Measure test coverage."""
        with SandboxExecutor() as executor:
            return executor.execute_with_coverage(test_code, source_code)
    
    def _analyze_coverage_gaps(
        self,
        source_code: str,
        test_code: str,
        coverage_result,
        cfgs: dict
    ) -> CoverageFeedback:
        """Analyze coverage gaps and provide feedback."""
        # Parse coverage output to identify untested lines
        untested_lines = self._parse_untested_lines(coverage_result.stdout)
        
        # Identify untested branches from CFG
        untested_branches = []
        for func_name, cfg in cfgs.items():
            branches = cfg.get_branch_nodes()
            for branch in branches:
                if branch.source_line in untested_lines:
                    untested_branches.append(f"{func_name}:line{branch.source_line}")
        
        # Identify missing edge cases
        missing_edge_cases = self._identify_missing_edge_cases(
            source_code,
            test_code
        )
        
        # Generate recommendations
        recommendations = []
        if untested_lines:
            recommendations.append(f"Add tests for lines: {untested_lines[:5]}")
        if untested_branches:
            recommendations.append(f"Test branches: {untested_branches[:3]}")
        if missing_edge_cases:
            recommendations.append(f"Add edge cases: {missing_edge_cases[:3]}")
        
        return CoverageFeedback(
            current_coverage=coverage_result.coverage or 0.0,
            untested_lines=untested_lines,
            untested_branches=untested_branches,
            missing_edge_cases=missing_edge_cases,
            recommendations=recommendations
        )
    
    def _parse_untested_lines(self, coverage_output: str) -> List[int]:
        """Parse untested line numbers from coverage output."""
        import re
        
        untested = []
        
        # Look for lines like: "source_module.py    10, 15-17, 23"
        for line in coverage_output.split('\n'):
            if 'source_module' in line:
                # Extract line numbers
                parts = line.split()
                for part in parts:
                    if part.isdigit():
                        untested.append(int(part))
                    elif '-' in part and part.replace('-', '').isdigit():
                        start, end = map(int, part.split('-'))
                        untested.extend(range(start, end + 1))
        
        return untested[:20]  # Limit to first 20
    
    def _identify_missing_edge_cases(self, source_code: str, test_code: str) -> List[str]:
        """Identify missing edge cases."""
        missing = []
        
        # Check for common edge cases
        edge_cases = [
            ("None", "None value"),
            ("[]", "empty list"),
            ("\"\"", "empty string"),
            ("0", "zero value"),
            ("-1", "negative value"),
        ]
        
        for value, description in edge_cases:
            if value in source_code and value not in test_code:
                missing.append(description)
        
        return missing
    
    def _generate_targeted_tests(
        self,
        source_code: str,
        current_tests: str,
        feedback: CoverageFeedback
    ) -> str:
        """Generate tests targeting specific coverage gaps."""
        console.print("  → Generating targeted tests for gaps...")
        
        gaps_description = f"""
Untested Lines: {feedback.untested_lines[:10]}
Untested Branches: {feedback.untested_branches[:5]}
Missing Edge Cases: {feedback.missing_edge_cases}
Recommendations: {feedback.recommendations}
"""
        
        prompt = f"""You have existing tests with {feedback.current_coverage:.1f}% coverage.
Generate ADDITIONAL tests to increase coverage to {self.target.line_coverage}%.

Source Code:
```python
{source_code}
```

Existing Tests:
```python
{current_tests}
```

Coverage Gaps:
{gaps_description}

Generate NEW test methods that specifically target:
1. Untested lines: {feedback.untested_lines[:5]}
2. Untested branches: {feedback.untested_branches[:3]}
3. Missing edge cases: {feedback.missing_edge_cases[:3]}

IMPORTANT:
- Generate ONLY the new test methods (not the full test class)
- Each new test should target specific untested code
- Use descriptive test names indicating what's being tested
- Include assertions for all code paths

Format as individual test methods:
```python
def test_<specific_scenario>():
    # Test untested code path
    ...
```"""
        
        response = self.llm_provider.generate(
            prompt=prompt,
            system="You are an expert at generating targeted tests for coverage gaps.",
            temperature=settings.coverage_temperature,
            max_tokens=settings.coverage_max_tokens
        )
        
        return self._extract_code(response.content)
    
    def _merge_tests(self, existing_tests: str, new_tests: str) -> str:
        """Merge new tests into existing test suite."""
        # Extract test methods from new_tests
        import re
        
        # Find all test methods in new tests
        new_methods = re.findall(
            r'(def test_\w+\([^)]*\):.*?)(?=\ndef |$)',
            new_tests,
            re.DOTALL
        )
        
        if not new_methods:
            return existing_tests
        
        # Find the test class in existing tests
        class_match = re.search(r'(class Test\w+.*?:)', existing_tests, re.DOTALL)
        
        if class_match:
            # Insert new methods before the last line (usually pass or final method)
            lines = existing_tests.split('\n')
            insert_pos = len(lines) - 1
            
            # Find a good insertion point (after last test method)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip().startswith('def test_'):
                    insert_pos = i + 1
                    break
            
            # Insert new methods
            indent = "    "  # Standard class method indent
            for method in new_methods:
                method_lines = method.strip().split('\n')
                indented = '\n'.join([indent + line if line else line for line in method_lines])
                lines.insert(insert_pos, '\n' + indented)
                insert_pos += 1
            
            return '\n'.join(lines)
        
        return existing_tests
    
    def _extract_code(self, text: str) -> str:
        """Extract Python code from markdown or plain text."""
        import re
        
        # Try to extract from markdown code block
        match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Try any code block
        match = re.search(r'```\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return text.strip()


def create_coverage_driven_generator(
    target_coverage: float = 90.0,
    max_iterations: int = 5,
    llm_provider: Optional[BaseLLMProvider] = None
) -> CoverageDrivenGenerator:
    """
    Factory function to create a coverage-driven generator.
    
    Args:
        target_coverage: Target coverage percentage
        max_iterations: Maximum iterations
        llm_provider: Optional LLM provider
        
    Returns:
        Configured CoverageDrivenGenerator
    """
    target = CoverageTarget(
        line_coverage=target_coverage,
        max_iterations=max_iterations
    )
    return CoverageDrivenGenerator(llm_provider=llm_provider, target=target)


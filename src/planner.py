"""
Enhanced planner with explicit task decomposition.

Implements:
- Multi-step task planning
- Dependency graph construction
- Plan verification and critique
- Dynamic replanning
"""

from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field
from rich.console import Console
from rich.tree import Tree

from config.settings import settings
from src.llm_providers import BaseLLMProvider, get_default_provider

console = Console()


class TaskStatus(str, Enum):
    """Status of a task."""
    
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class Task(BaseModel):
    """A single task in the plan."""
    
    id: str = Field(..., description="Unique task ID")
    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    tool: str = Field(..., description="Tool to use")
    dependencies: List[str] = Field(default_factory=list, description="Dependent task IDs")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status")
    result: Optional[str] = Field(default=None, description="Task result")
    priority: int = Field(default=5, description="Priority (1-10, higher is more urgent)")


class ExecutionPlan(BaseModel):
    """Complete execution plan."""
    
    goal: str = Field(..., description="Overall goal")
    tasks: List[Task] = Field(default_factory=list, description="List of tasks")
    task_graph: Dict[str, List[str]] = Field(default_factory=dict, description="Dependency graph")
    estimated_steps: int = Field(..., description="Estimated total steps")
    confidence: float = Field(..., description="Plan confidence (0-1)")


class PlannerWithDecomposition:
    """
    Enhanced planner with task decomposition.
    
    Breaks down complex goals into structured, executable tasks with
    dependency tracking and dynamic replanning.
    """
    
    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        """
        Initialize planner.
        
        Args:
            llm_provider: LLM provider for planning
        """
        self.llm_provider = llm_provider or get_default_provider()
        self.current_plan: Optional[ExecutionPlan] = None
        
        console.print("[green]✓[/green] Enhanced planner initialized")
    
    def create_plan(
        self,
        goal: str,
        context: Optional[Dict] = None
    ) -> ExecutionPlan:
        """
        Create execution plan for a goal.
        
        Args:
            goal: High-level goal description
            context: Optional context information
            
        Returns:
            ExecutionPlan with task breakdown
            
        Example:
            >>> planner = PlannerWithDecomposition()
            >>> plan = planner.create_plan("Generate tests for all changed functions")
            >>> print(f"Steps: {len(plan.tasks)}")
        """
        console.print(f"[cyan]Creating execution plan for: {goal}[/cyan]")
        
        # Use LLM to decompose task
        tasks = self._decompose_task(goal, context)
        
        # Build dependency graph
        task_graph = self._build_dependency_graph(tasks)
        
        # Estimate steps
        estimated_steps = len(tasks)
        
        # Calculate confidence
        confidence = self._estimate_confidence(tasks)
        
        plan = ExecutionPlan(
            goal=goal,
            tasks=tasks,
            task_graph=task_graph,
            estimated_steps=estimated_steps,
            confidence=confidence
        )
        
        self.current_plan = plan
        
        # Display plan
        self._display_plan(plan)
        
        return plan
    
    def _decompose_task(
        self,
        goal: str,
        context: Optional[Dict]
    ) -> List[Task]:
        """Decompose high-level goal into tasks."""
        context_str = ""
        if context:
            context_str = f"\nContext: {context}"
        
        prompt = f"""Decompose this goal into specific, actionable tasks.

Goal: {goal}
{context_str}

Available Tools:
- check_git_changes: Check what code has changed
- search_code: Search for relevant code
- get_code_context: Retrieve code context
- generate_tests: Generate test code
- execute_tests: Run tests in sandbox
- review_quality: Check test quality

Create a step-by-step plan with:
1. Task name
2. Description
3. Tool to use
4. Dependencies (which tasks must complete first)
5. Priority (1-10)

Format as:
TASK_1:
Name: <name>
Description: <description>
Tool: <tool_name>
Dependencies: <comma-separated task IDs, or "none">
Priority: <1-10>

TASK_2:
...

Be specific and practical. Aim for 3-8 tasks."""

        try:
            response = self.llm_provider.generate(
                prompt=prompt,
                system="You are an expert task planner for test generation workflows. Always respond with valid JSON.",
                temperature=settings.planner_temperature,
                max_tokens=settings.planner_max_tokens
            )
            
            return self._parse_task_decomposition(response.content)
        
        except Exception as e:
            console.print(f"[yellow]Warning: Task decomposition failed: {e}[/yellow]")
            # Fallback to default plan
            return self._create_default_plan(goal)
    
    def _parse_task_decomposition(self, content: str) -> List[Task]:
        """Parse LLM response into tasks."""
        import re
        
        tasks = []
        task_blocks = re.split(r'TASK_\d+:', content)[1:]  # Skip first empty split
        
        for i, block in enumerate(task_blocks, 1):
            # Extract fields
            name_match = re.search(r'Name:\s*(.+)', block)
            desc_match = re.search(r'Description:\s*(.+)', block)
            tool_match = re.search(r'Tool:\s*(.+)', block)
            deps_match = re.search(r'Dependencies:\s*(.+)', block)
            priority_match = re.search(r'Priority:\s*(\d+)', block)
            
            task_id = f"task_{i}"
            name = name_match.group(1).strip() if name_match else f"Task {i}"
            description = desc_match.group(1).strip() if desc_match else ""
            tool = tool_match.group(1).strip() if tool_match else "unknown"
            
            # Parse dependencies
            dependencies = []
            if deps_match:
                deps_str = deps_match.group(1).strip().lower()
                if deps_str != "none" and deps_str:
                    # Extract task IDs
                    dep_ids = re.findall(r'task_\d+', deps_str)
                    dependencies = dep_ids
            
            priority = int(priority_match.group(1)) if priority_match else 5
            
            tasks.append(Task(
                id=task_id,
                name=name,
                description=description,
                tool=tool,
                dependencies=dependencies,
                priority=priority
            ))
        
        return tasks
    
    def _create_default_plan(self, goal: str) -> List[Task]:
        """Create a default plan as fallback."""
        return [
            Task(
                id="task_1",
                name="Identify changes",
                description="Check what code needs testing",
                tool="check_git_changes",
                dependencies=[],
                priority=10
            ),
            Task(
                id="task_2",
                name="Gather context",
                description="Retrieve relevant code context",
                tool="get_code_context",
                dependencies=["task_1"],
                priority=9
            ),
            Task(
                id="task_3",
                name="Generate tests",
                description="Create test cases",
                tool="generate_tests",
                dependencies=["task_2"],
                priority=8
            ),
            Task(
                id="task_4",
                name="Execute tests",
                description="Run tests and verify",
                tool="execute_tests",
                dependencies=["task_3"],
                priority=7
            ),
        ]
    
    def _build_dependency_graph(self, tasks: List[Task]) -> Dict[str, List[str]]:
        """Build task dependency graph."""
        graph = {}
        
        for task in tasks:
            graph[task.id] = task.dependencies.copy()
        
        return graph
    
    def _estimate_confidence(self, tasks: List[Task]) -> float:
        """Estimate plan confidence."""
        # Simple heuristic based on task count and dependencies
        if not tasks:
            return 0.0
        
        # Base confidence
        confidence = 0.8
        
        # Penalty for too many tasks
        if len(tasks) > 10:
            confidence -= 0.1
        
        # Penalty for complex dependencies
        total_deps = sum(len(task.dependencies) for task in tasks)
        if total_deps > len(tasks) * 2:
            confidence -= 0.1
        
        return max(confidence, 0.3)
    
    def _display_plan(self, plan: ExecutionPlan) -> None:
        """Display plan as a tree."""
        tree = Tree(f"[bold cyan]Plan: {plan.goal}[/bold cyan]")
        tree.add(f"Estimated steps: {plan.estimated_steps}")
        tree.add(f"Confidence: {plan.confidence:.0%}")
        
        tasks_node = tree.add("[bold]Tasks:[/bold]")
        
        for task in plan.tasks:
            status_icon = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.READY: "▶️",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.BLOCKED: "🚫"
            }.get(task.status, "❓")
            
            task_node = tasks_node.add(
                f"{status_icon} {task.id}: {task.name} (Tool: {task.tool})"
            )
            
            if task.dependencies:
                task_node.add(f"Depends on: {', '.join(task.dependencies)}")
        
        console.print(tree)
    
    def get_next_tasks(self, plan: ExecutionPlan) -> List[Task]:
        """
        Get next ready-to-execute tasks.
        
        Args:
            plan: Current execution plan
            
        Returns:
            List of tasks ready for execution
        """
        ready_tasks = []
        
        for task in plan.tasks:
            if task.status in [TaskStatus.COMPLETED, TaskStatus.IN_PROGRESS]:
                continue
            
            # Check if dependencies are satisfied
            deps_satisfied = all(
                self._get_task(plan, dep_id).status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )
            
            if deps_satisfied:
                task.status = TaskStatus.READY
                ready_tasks.append(task)
        
        # Sort by priority
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        return ready_tasks
    
    def update_task_status(
        self,
        plan: ExecutionPlan,
        task_id: str,
        status: TaskStatus,
        result: Optional[str] = None
    ) -> None:
        """
        Update task status.
        
        Args:
            plan: Execution plan
            task_id: Task ID to update
            status: New status
            result: Optional result data
        """
        task = self._get_task(plan, task_id)
        task.status = status
        
        if result:
            task.result = result
        
        console.print(f"  → Task '{task.name}': {status.value}")
    
    def _get_task(self, plan: ExecutionPlan, task_id: str) -> Task:
        """Get task by ID."""
        for task in plan.tasks:
            if task.id == task_id:
                return task
        raise ValueError(f"Task {task_id} not found")
    
    def replan(
        self,
        plan: ExecutionPlan,
        failure_task_id: str,
        error: str
    ) -> ExecutionPlan:
        """
        Create alternative plan after failure.
        
        Args:
            plan: Current plan
            failure_task_id: Task that failed
            error: Error message
            
        Returns:
            Revised execution plan
        """
        console.print(f"[yellow]Replanning after failure in {failure_task_id}[/yellow]")
        
        # For now, create a recovery plan
        # In production, use LLM to create intelligent alternative
        
        recovery_task = Task(
            id=f"recovery_{failure_task_id}",
            name=f"Recover from {failure_task_id} failure",
            description=f"Alternative approach for: {error}",
            tool="generate_tests",
            dependencies=[],
            priority=10
        )
        
        # Insert recovery task
        plan.tasks.append(recovery_task)
        
        return plan


def create_planner(llm_provider: Optional[BaseLLMProvider] = None) -> PlannerWithDecomposition:
    """Factory function to create planner."""
    return PlannerWithDecomposition(llm_provider=llm_provider)


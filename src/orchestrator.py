"""
LangGraph-based orchestrator for test generation.

This module implements a true orchestrator pattern where the LLM dynamically
selects and executes tools based on the current state and requirements.
"""

import time
from typing import Annotated, List, Literal, TypedDict

import ollama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from rich.console import Console

from config.settings import settings
from src.guardrails.guard_manager import GuardManager, create_guard_manager
from src.tools import get_all_tools

console = Console()


class AgentState(TypedDict):
    """
    State for the orchestrator agent.
    
    Attributes:
        messages: Conversation history
        task: Current task description
        iteration: Current iteration number
        max_iterations: Maximum iterations allowed
        generated_tests: Final generated test code
        completed: Whether task is complete
    """
    
    messages: Annotated[List[BaseMessage], add_messages]
    task: str
    iteration: int
    max_iterations: int
    generated_tests: str
    completed: bool


class TestGenerationOrchestrator:
    """
    LangGraph-based orchestrator for intelligent test generation.
    
    This orchestrator dynamically selects and executes tools based on
    the current state, allowing for flexible and adaptive workflows.
    """
    
    def __init__(
        self,
        tools: List[BaseTool] = None,
        max_iterations: int = 10,
        session_id: str = None,
        interactive: bool = True
    ) -> None:
        """
        Initialize the orchestrator.
        
        Args:
            tools: List of tools available to the agent
            max_iterations: Maximum number of iterations
            session_id: Session identifier for audit logging
            interactive: Enable HITL prompts
        """
        self.tools = tools or get_all_tools()
        self.max_iterations = max_iterations
        
        # Initialize guard manager for comprehensive safety
        self.guard_manager = create_guard_manager(session_id, interactive)
        
        # Configure Ollama
        if settings.ollama_api_key:
            ollama.api_key = settings.ollama_api_key
        if settings.ollama_base_url:
            ollama.base_url = settings.ollama_base_url
        
        # Build the graph
        self.graph = self._build_graph()
        
        console.print(f"[green]✓[/green] Orchestrator initialized with {len(self.tools)} tools")
        console.print(f"[green]✓[/green] Guard Manager active for session: {self.guard_manager.session_id}")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph execution graph."""
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._guarded_tool_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        
        # Tool execution loops back to agent
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def _guarded_tool_node(self, state: AgentState) -> AgentState:
        """
        Guarded tool execution node with comprehensive safety checks.
        
        This node:
        1. Extracts tool call from last message
        2. Runs policy checks, schema validation, and HITL approval
        3. Executes tool if approved
        4. Logs results to audit trail
        """
        messages = state.get("messages", [])
        if not messages:
            return state
        
        last_message = messages[-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return state
        
        # Process each tool call (usually just one)
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", "unknown")
            
            start_time = time.time()
            
            # GUARDRAILS: Check before execution
            console.print(f"[cyan]🛡️  Running guardrails for {tool_name}...[/cyan]")
            
            guard_result = self.guard_manager.check_tool_call(
                tool=tool_name,
                params=tool_args,
                context={
                    "iteration": state["iteration"],
                    "user_id": "system"
                }
            )
            
            if not guard_result.allowed:
                # BLOCKED by guardrails
                console.print(f"[red]❌ Tool blocked: {guard_result.reason}[/red]")
                
                error_msg = ToolMessage(
                    content=f"BLOCKED: {guard_result.reason}",
                    tool_call_id=tool_id
                )
                state["messages"].append(error_msg)
                continue
            
            # Use corrected params if provided
            if guard_result.corrected_params:
                tool_args = guard_result.corrected_params
                console.print("[yellow]📝 Parameters auto-corrected by guardrails[/yellow]")
            
            # Execute tool
            try:
                # Find the tool
                tool = next((t for t in self.tools if t.name == tool_name), None)
                if not tool:
                    raise ValueError(f"Tool not found: {tool_name}")
                
                console.print(f"[green]✅ Executing {tool_name}...[/green]")
                
                result = tool.invoke(tool_args)
                
                # Log successful execution
                duration_ms = (time.time() - start_time) * 1000
                self.guard_manager.log_tool_result(
                    tool=tool_name,
                    params=tool_args,
                    success=True,
                    duration_ms=duration_ms
                )
                
                # Add result to messages
                result_msg = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_id
                )
                state["messages"].append(result_msg)
            
            except Exception as e:
                # Log failed execution
                duration_ms = (time.time() - start_time) * 1000
                self.guard_manager.log_tool_result(
                    tool=tool_name,
                    params=tool_args,
                    success=False,
                    duration_ms=duration_ms,
                    error=str(e)
                )
                
                console.print(f"[red]❌ Tool execution failed: {e}[/red]")
                
                error_msg = ToolMessage(
                    content=f"ERROR: {str(e)}",
                    tool_call_id=tool_id
                )
                state["messages"].append(error_msg)
        
        return state
    
    def _agent_node(self, state: AgentState) -> AgentState:
        """
        Agent decision node - decides which tool to call next.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with agent's decision
        """
        state["iteration"] += 1
        
        console.print(f"\n[bold cyan]Iteration {state['iteration']}[/bold cyan]")
        
        # Build prompt with available tools
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])
        
        system_prompt = f"""You are an intelligent test generation orchestrator. 
Your task is to generate comprehensive tests by dynamically selecting and using the appropriate tools.

Available Tools:
{tool_descriptions}

Strategy:
1. Start by understanding what code needs testing (use check_git_changes or search_code)
2. Gather comprehensive context (use get_code_context)
3. Generate tests (use generate_tests)
4. Verify tests work (use execute_tests)
5. Refine if tests fail (iterate generate + execute)
6. When tests pass, respond with "TASK_COMPLETE"

Current Task: {state['task']}

Think step-by-step about what information you need and which tool will get it.
Choose ONE tool to call now, or respond with "TASK_COMPLETE" if done."""
        
        # Get conversation history
        messages = state.get("messages", [])
        
        # Add current state context
        context_msg = f"\nIteration: {state['iteration']}/{state['max_iterations']}"
        if state.get("generated_tests"):
            context_msg += "\n✓ Tests have been generated"
        
        # Call LLM for decision
        try:
            # Format messages for Ollama
            prompt = self._format_messages_for_ollama(messages, context_msg)
            
            response = ollama.generate(
                model=settings.ollama_model,
                prompt=prompt,
                system=system_prompt
            )
            
            response_text = response['response']
            
            # Check if task is complete
            if "TASK_COMPLETE" in response_text.upper():
                state["completed"] = True
                state["messages"].append(AIMessage(content=response_text))
                return state
            
            # Parse tool call (simplified - in production use structured output)
            tool_name, tool_input = self._parse_tool_call(response_text)
            
            if tool_name:
                console.print(f"  → Calling tool: [yellow]{tool_name}[/yellow]")
                
                # Create tool call message
                state["messages"].append(AIMessage(
                    content=response_text,
                    tool_calls=[{
                        "name": tool_name,
                        "args": tool_input,
                        "id": f"call_{state['iteration']}"
                    }]
                ))
            else:
                # LLM responded without tool call
                state["messages"].append(AIMessage(content=response_text))
        
        except Exception as e:
            console.print(f"[red]Error in agent node: {e}[/red]")
            state["completed"] = True
        
        return state
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """
        Determine whether to continue or end the workflow.
        
        Args:
            state: Current state
            
        Returns:
            "continue" to keep going, "end" to stop
        """
        # Check completion
        if state.get("completed"):
            return "end"
        
        # Check iteration limit
        if state["iteration"] >= state["max_iterations"]:
            console.print("[yellow]Max iterations reached[/yellow]")
            return "end"
        
        # Check if last message has tool calls
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage):
            if hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
                return "continue"
        
        return "end"
    
    def _format_messages_for_ollama(
        self,
        messages: List[BaseMessage],
        context: str
    ) -> str:
        """Format messages for Ollama's prompt format."""
        formatted = []
        
        for msg in messages[-5:]:  # Last 5 messages for context
            if isinstance(msg, HumanMessage):
                formatted.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"Assistant: {msg.content}")
            elif isinstance(msg, ToolMessage):
                formatted.append(f"Tool Result: {msg.content[:500]}...")
        
        formatted.append(f"\nContext: {context}")
        formatted.append("\nWhat should we do next?")
        
        return "\n\n".join(formatted)
    
    def _parse_tool_call(self, response: str) -> tuple:
        """
        Parse tool call from LLM response.
        
        This is simplified - in production, use structured output.
        """
        # Look for tool patterns
        import re
        
        # Pattern: tool_name(arg1=value1, arg2=value2)
        pattern = r'(\w+)\((.*?)\)'
        match = re.search(pattern, response)
        
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            
            # Parse arguments (simplified)
            args = {}
            for arg in args_str.split(','):
                if '=' in arg:
                    key, val = arg.split('=', 1)
                    args[key.strip()] = val.strip(' "\'')
            
            return tool_name, args
        
        return None, {}
    
    def generate_tests(self, task: str) -> str:
        """
        Generate tests for the given task.
        
        Args:
            task: Description of what to test
            
        Returns:
            Generated test code
            
        Example:
            >>> orch = TestGenerationOrchestrator()
            >>> tests = orch.generate_tests(
            ...     "Generate tests for all functions changed in the last commit"
            ... )
        """
        console.print(f"\n[bold]Starting orchestrator workflow[/bold]")
        console.print(f"Task: {task}\n")
        
        # Initial state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=task)],
            "task": task,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "generated_tests": "",
            "completed": False
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Extract generated tests from state or messages
        if final_state.get("generated_tests"):
            return final_state["generated_tests"]
        
        # Look for test code in messages
        for msg in reversed(final_state.get("messages", [])):
            if "```python" in str(msg.content):
                import re
                match = re.search(r'```python\n(.*?)\n```', str(msg.content), re.DOTALL)
                if match:
                    return match.group(1)
        
        return "No tests generated"


def create_orchestrator(max_iterations: int = 10) -> TestGenerationOrchestrator:
    """
    Factory function to create an orchestrator instance.
    
    Args:
        max_iterations: Maximum iterations
        
    Returns:
        Configured orchestrator
    """
    return TestGenerationOrchestrator(max_iterations=max_iterations)


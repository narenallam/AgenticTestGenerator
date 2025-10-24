"""
Structured JSON schemas for component communication.

This module defines Pydantic models for all inter-component communication,
ensuring type safety and validation throughout the test generation pipeline.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


# ============================================================================
# Enums for standardized values
# ============================================================================

class TestType(str, Enum):
    """Types of tests that can be generated."""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"


class ToolName(str, Enum):
    """Available tools for the agent."""
    SEARCH_CODEBASE = "search_codebase"
    RETRIEVE_SIMILAR_CODE = "retrieve_similar_code"
    GET_GIT_HISTORY = "get_git_history"
    ANALYZE_CODE_STRUCTURE = "analyze_code_structure"
    VALIDATE_IN_SANDBOX = "validate_in_sandbox"
    READ_FILE = "read_file"
    LIST_FILES = "list_files"


class ComponentName(str, Enum):
    """System components."""
    ORCHESTRATOR = "orchestrator"
    LLM_PROVIDER = "llm_provider"
    GUARDRAILS = "guardrails"
    RAG_RETRIEVAL = "rag_retrieval"
    CRITIC = "critic"
    SANDBOX = "sandbox"
    INDEXER = "indexer"
    DATABASE = "database"
    GIT = "git"


class ValidationStatus(str, Enum):
    """Validation result status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


# ============================================================================
# LLM Request/Response Schemas
# ============================================================================

class LLMRequest(BaseModel):
    """Structured request to LLM."""
    prompt: str = Field(..., description="The prompt to send to LLM")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    max_tokens: Optional[int] = Field(default=4000, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.2, description="Temperature for generation")
    stop_sequences: Optional[List[str]] = Field(default=None, description="Sequences to stop generation")
    tools_used: List[str] = Field(default_factory=list, description="Tools that were used before this request")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Generate tests for function X",
                "context": {"file_path": "src/main.py", "function_name": "calculate"},
                "max_tokens": 4000,
                "temperature": 0.2,
                "tools_used": ["search_codebase", "retrieve_similar_code"]
            }
        }


class ToolCall(BaseModel):
    """Record of a tool invocation."""
    tool_name: str = Field(..., description="Name of the tool called")
    arguments: Dict[str, Any] = Field(..., description="Arguments passed to tool")
    result: Optional[str] = Field(None, description="Result returned by tool")
    execution_time_ms: Optional[float] = Field(None, description="Time taken to execute")
    timestamp: datetime = Field(default_factory=datetime.now)


class TestCode(BaseModel):
    """Structured test code output."""
    code: str = Field(..., description="The generated test code")
    imports: List[str] = Field(default_factory=list, description="Required imports")
    test_functions: List[str] = Field(default_factory=list, description="Names of test functions")
    test_classes: List[str] = Field(default_factory=list, description="Names of test classes")
    dependencies: List[str] = Field(default_factory=list, description="External dependencies needed")
    
    @validator('code')
    def code_not_empty(cls, v):
        if not v or len(v.strip()) < 50:
            raise ValueError("Test code must be at least 50 characters")
        return v
    
    @validator('code')
    def code_has_pytest(cls, v):
        if 'import pytest' not in v and 'from pytest' not in v:
            raise ValueError("Test code must import pytest")
        return v


class LLMResponse(BaseModel):
    """Structured response from LLM."""
    test_code: TestCode = Field(..., description="Generated test code")
    reasoning: Optional[str] = Field(None, description="LLM's reasoning process")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tools called during generation")
    tokens_used: Optional[int] = Field(None, description="Total tokens used")
    generation_time_ms: Optional[float] = Field(None, description="Time taken to generate")
    model_name: str = Field(..., description="Model used for generation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "test_code": {
                    "code": "import pytest\n\ndef test_example():\n    assert True",
                    "imports": ["pytest"],
                    "test_functions": ["test_example"],
                    "test_classes": [],
                    "dependencies": []
                },
                "reasoning": "Generated basic test structure",
                "tool_calls": [],
                "tokens_used": 150,
                "generation_time_ms": 1200.5,
                "model_name": "qwen3-coder:30b"
            }
        }


# ============================================================================
# Guardrail Schemas
# ============================================================================

class GuardrailCheck(BaseModel):
    """Result of a single guardrail check."""
    check_name: str = Field(..., description="Name of the check")
    status: ValidationStatus = Field(..., description="Result status")
    message: Optional[str] = Field(None, description="Details about the check")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict)


class GuardrailValidation(BaseModel):
    """Complete guardrail validation result."""
    allowed: bool = Field(..., description="Whether the content passed validation")
    checks: List[GuardrailCheck] = Field(default_factory=list, description="Individual check results")
    blocked_reason: Optional[str] = Field(None, description="Reason if blocked")
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# RAG/Retrieval Schemas
# ============================================================================

class CodeContext(BaseModel):
    """Retrieved code context."""
    file_path: str = Field(..., description="Path to the file")
    content: str = Field(..., description="Code content")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RAGRetrievalRequest(BaseModel):
    """Request for RAG retrieval."""
    query: str = Field(..., description="Query to search for")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Filters to apply")


class RAGRetrievalResponse(BaseModel):
    """Response from RAG retrieval."""
    contexts: List[CodeContext] = Field(..., description="Retrieved code contexts")
    total_found: int = Field(..., description="Total number of matches")
    query_time_ms: float = Field(..., description="Time taken for retrieval")


# ============================================================================
# Test Generation Orchestration Schemas
# ============================================================================

class TestGenerationRequest(BaseModel):
    """Request to generate tests."""
    source_file: str = Field(..., description="Path to source file")
    target_code: str = Field(..., description="Code to generate tests for")
    function_name: Optional[str] = Field(None, description="Specific function to test")
    test_type: TestType = Field(default=TestType.UNIT, description="Type of test to generate")
    context: Optional[str] = Field(None, description="Additional context")
    use_rag: bool = Field(default=True, description="Whether to use RAG retrieval")
    use_git_history: bool = Field(default=True, description="Whether to use git history")


class TestGenerationResponse(BaseModel):
    """Response from test generation."""
    success: bool = Field(..., description="Whether generation succeeded")
    test_file: str = Field(..., description="Path to generated test file")
    test_code: TestCode = Field(..., description="The generated test code")
    llm_response: LLMResponse = Field(..., description="Raw LLM response")
    guardrail_validation: GuardrailValidation = Field(..., description="Validation results")
    generation_time_ms: float = Field(..., description="Total generation time")
    error: Optional[str] = Field(None, description="Error message if failed")


# ============================================================================
# Component Communication Schemas
# ============================================================================

class ComponentMessage(BaseModel):
    """Message between components."""
    from_component: ComponentName = Field(..., description="Source component")
    to_component: ComponentName = Field(..., description="Destination component")
    message_type: str = Field(..., description="Type of message")
    payload: Dict[str, Any] = Field(..., description="Message payload")
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: Optional[str] = Field(None, description="ID to correlate related messages")


class ComponentStatus(BaseModel):
    """Status update from a component."""
    component: ComponentName = Field(..., description="Component reporting status")
    status: Literal["idle", "processing", "completed", "error"] = Field(..., description="Current status")
    message: str = Field(..., description="Status message")
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Progress (0-1)")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Critic/Quality Schemas
# ============================================================================

class QualityMetric(BaseModel):
    """A quality metric score."""
    metric_name: str = Field(..., description="Name of the metric")
    score: float = Field(..., ge=0.0, le=1.0, description="Score (0-1)")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Minimum acceptable score")
    passed: bool = Field(..., description="Whether metric passed threshold")
    details: Optional[str] = Field(None, description="Additional details")


class CriticReview(BaseModel):
    """Critic review of generated tests."""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    metrics: List[QualityMetric] = Field(..., description="Individual metric scores")
    passed: bool = Field(..., description="Whether tests passed review")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Database/Tracking Schemas
# ============================================================================

class FunctionInfo(BaseModel):
    """Information about a source function."""
    file_path: str = Field(..., description="Path to source file")
    function_name: str = Field(..., description="Name of the function")
    start_line: int = Field(..., ge=1, description="Starting line number")
    end_line: int = Field(..., ge=1, description="Ending line number")
    signature: str = Field(..., description="Function signature")
    hash: str = Field(..., description="Hash of function content")
    has_test: bool = Field(default=False, description="Whether test exists")
    test_file_path: Optional[str] = Field(None, description="Path to test file")


class CoverageStats(BaseModel):
    """Test coverage statistics."""
    total_functions: int = Field(..., ge=0)
    tested_functions: int = Field(..., ge=0)
    untested_functions: int = Field(..., ge=0)
    coverage_percentage: float = Field(..., ge=0.0, le=100.0)
    by_file: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Error Schemas
# ============================================================================

class ComponentError(BaseModel):
    """Structured error from a component."""
    component: ComponentName = Field(..., description="Component that errored")
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Error message")
    stack_trace: Optional[str] = Field(None, description="Stack trace if available")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    recoverable: bool = Field(default=True, description="Whether error is recoverable")
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Utility Functions
# ============================================================================

def create_llm_request(
    prompt: str,
    context: Optional[Dict] = None,
    tools_used: Optional[List[str]] = None
) -> LLMRequest:
    """Helper to create LLM request."""
    return LLMRequest(
        prompt=prompt,
        context=context or {},
        tools_used=tools_used or []
    )


def create_component_message(
    from_comp: ComponentName,
    to_comp: ComponentName,
    msg_type: str,
    payload: Dict
) -> ComponentMessage:
    """Helper to create component message."""
    return ComponentMessage(
        from_component=from_comp,
        to_component=to_comp,
        message_type=msg_type,
        payload=payload
    )


def validate_test_code(code: str) -> TestCode:
    """Validate and structure test code."""
    # Extract imports
    imports = [
        line.split()[1] for line in code.split('\n')
        if line.strip().startswith('import ')
    ]
    
    # Extract test function names
    test_functions = [
        line.split('(')[0].split()[-1]
        for line in code.split('\n')
        if line.strip().startswith('def test_')
    ]
    
    # Extract test class names
    test_classes = [
        line.split('(')[0].split()[-1]
        for line in code.split('\n')
        if 'class Test' in line
    ]
    
    return TestCode(
        code=code,
        imports=imports,
        test_functions=test_functions,
        test_classes=test_classes,
        dependencies=[]
    )


"""
Instrumentation decorators and helpers.

Provides decorators for automatic logging, metrics, and tracing.
"""

import functools
import time
from typing import Any, Callable, Optional

from .logger import bind_context, get_logger
from .metrics import counter, gauge, histogram
from .tracer import SpanStatus, get_current_span, get_tracer, span_context

logger = get_logger()


# ═══════════════════════════════════════════════════════════════════════════
# Decorators
# ═══════════════════════════════════════════════════════════════════════════


def observe(
    operation: Optional[str] = None,
    metric_prefix: Optional[str] = None,
    log_level: str = "INFO",
    trace: bool = True,
    record_errors: bool = True,
):
    """
    Comprehensive observability decorator.
    
    Automatically adds logging, metrics, and tracing to a function.
    
    Args:
        operation: Operation name (defaults to function name)
        metric_prefix: Prefix for metric names
        log_level: Log level for function calls
        trace: Enable tracing
        record_errors: Record errors in metrics
    
    Example:
        @observe(operation="test_generation", metric_prefix="test_gen")
        def generate_tests(source_code: str) -> str:
            return tests
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__
        metric_name = metric_prefix or func.__name__
        
        # Create metrics
        calls_counter = counter(
            f"{metric_name}_calls",
            f"Total calls to {op_name}",
            labels=["status"]
        )
        duration_hist = histogram(
            f"{metric_name}_duration_seconds",
            f"Duration of {op_name}",
        )
        errors_counter = counter(
            f"{metric_name}_errors",
            f"Errors in {op_name}",
            labels=["error_type"]
        )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Start timing
            start_time = time.time()
            
            # Log function call
            log_func = getattr(logger, log_level.lower())
            log_func(f"{op_name} started", function=op_name)
            
            # Tracing
            span = None
            if trace:
                tracer = get_tracer()
                span = tracer.start_span(op_name)
                span.set_attribute("function", func.__name__)
                span.set_attribute("module", func.__module__)
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Record success
                duration = time.time() - start_time
                calls_counter.inc(status="success")
                duration_hist.observe(duration)
                
                if span:
                    tracer.finish_span(span, status=SpanStatus.OK)
                
                log_func(
                    f"{op_name} completed",
                    function=op_name,
                    duration_ms=duration * 1000
                )
                
                return result
            
            except Exception as e:
                # Record error
                duration = time.time() - start_time
                calls_counter.inc(status="error")
                duration_hist.observe(duration)
                
                if record_errors:
                    errors_counter.inc(error_type=type(e).__name__)
                
                if span:
                    tracer.finish_span(span, status=SpanStatus.ERROR, error=str(e))
                
                logger.error(
                    f"{op_name} failed: {e}",
                    function=op_name,
                    error_type=type(e).__name__,
                    duration_ms=duration * 1000,
                    exc_info=True
                )
                
                raise
        
        return wrapper
    return decorator


def observe_llm_call(provider: str, model: str):
    """
    Specialized decorator for LLM calls.
    
    Tracks tokens, cost, and latency.
    
    Args:
        provider: LLM provider (ollama, openai, gemini)
        model: Model name
    
    Example:
        @observe_llm_call(provider="ollama", model="qwen3-coder:30b")
        def generate(prompt: str) -> str:
            return response
    """
    def decorator(func: Callable) -> Callable:
        # Create LLM-specific metrics
        calls_counter = counter(
            "llm_calls",
            "Total LLM calls",
            labels=["provider", "model", "status"]
        )
        tokens_counter = counter(
            "llm_tokens",
            "Total LLM tokens",
            labels=["provider", "model", "type"]
        )
        latency_hist = histogram(
            "llm_latency_seconds",
            "LLM call latency",
            labels=["provider", "model"]
        )
        cost_counter = counter(
            "llm_cost_dollars",
            "LLM cost in dollars",
            labels=["provider", "model"]
        )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            logger.debug(
                f"LLM call started",
                provider=provider,
                model=model
            )
            
            # Tracing
            with span_context(f"llm_call_{provider}", provider=provider, model=model) as span:
                try:
                    result = func(*args, **kwargs)
                    
                    # Record success
                    duration = time.time() - start_time
                    calls_counter.inc(provider=provider, model=model, status="success")
                    latency_hist.observe(duration, provider=provider, model=model)
                    
                    # Extract token info if available
                    if hasattr(result, 'usage'):
                        tokens_counter.inc(
                            result.usage.input_tokens,
                            provider=provider,
                            model=model,
                            type="input"
                        )
                        tokens_counter.inc(
                            result.usage.output_tokens,
                            provider=provider,
                            model=model,
                            type="output"
                        )
                    
                    logger.debug(
                        f"LLM call completed",
                        provider=provider,
                        model=model,
                        duration_ms=duration * 1000
                    )
                    
                    return result
                
                except Exception as e:
                    duration = time.time() - start_time
                    calls_counter.inc(provider=provider, model=model, status="error")
                    latency_hist.observe(duration, provider=provider, model=model)
                    
                    logger.error(
                        f"LLM call failed: {e}",
                        provider=provider,
                        model=model,
                        error=str(e),
                        exc_info=True
                    )
                    
                    raise
        
        return wrapper
    return decorator


def observe_agent(agent_name: str):
    """
    Decorator for agent operations.
    
    Tracks agent iterations, tool calls, and decisions.
    
    Args:
        agent_name: Name of the agent (planner, coder, critic)
    
    Example:
        @observe_agent("planner")
        def plan_task(goal: str) -> Plan:
            return plan
    """
    def decorator(func: Callable) -> Callable:
        # Agent-specific metrics
        iterations_counter = counter(
            "agent_iterations",
            "Agent iterations",
            labels=["agent"]
        )
        decision_time_hist = histogram(
            "agent_decision_time_seconds",
            "Agent decision time",
            labels=["agent"]
        )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            logger.bind(agent=agent_name).info(f"Agent {agent_name} started")
            
            with span_context(f"agent_{agent_name}", agent=agent_name) as span:
                try:
                    result = func(*args, **kwargs)
                    
                    duration = time.time() - start_time
                    iterations_counter.inc(agent=agent_name)
                    decision_time_hist.observe(duration, agent=agent_name)
                    
                    logger.bind(agent=agent_name).info(
                        f"Agent {agent_name} completed",
                        duration_ms=duration * 1000
                    )
                    
                    return result
                
                except Exception as e:
                    logger.bind(agent=agent_name).error(
                        f"Agent {agent_name} failed: {e}",
                        exc_info=True
                    )
                    raise
        
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════


def record_test_generation(
    language: str,
    framework: str,
    coverage: float,
    pass_rate: float,
    test_count: int,
    duration_seconds: float
):
    """
    Record test generation metrics.
    
    Args:
        language: Programming language
        framework: Test framework
        coverage: Code coverage (0.0-1.0)
        pass_rate: Test pass rate (0.0-1.0)
        test_count: Number of tests generated
        duration_seconds: Generation duration
    """
    # Test generation metrics
    tests_counter = counter(
        "tests_generated",
        "Total tests generated",
        labels=["language", "framework"]
    )
    coverage_gauge = gauge(
        "test_coverage_ratio",
        "Test coverage ratio",
        labels=["language"]
    )
    pass_rate_gauge = gauge(
        "test_pass_rate_ratio",
        "Test pass rate",
        labels=["language"]
    )
    duration_hist = histogram(
        "test_generation_duration_seconds",
        "Test generation duration",
        labels=["language"]
    )
    
    # Record metrics
    tests_counter.inc(test_count, language=language, framework=framework)
    coverage_gauge.set(coverage, language=language)
    pass_rate_gauge.set(pass_rate, language=language)
    duration_hist.observe(duration_seconds, language=language)
    
    logger.bind(
        language=language,
        framework=framework,
        coverage=coverage,
        pass_rate=pass_rate,
        test_count=test_count
    ).success(
        f"Test generation complete: {test_count} tests, {coverage*100:.1f}% coverage, {pass_rate*100:.1f}% pass rate"
    )


def record_guardrail_event(
    guardrail_type: str,
    action: str,  # check, violation, block
    details: Optional[str] = None
):
    """
    Record guardrail events.
    
    Args:
        guardrail_type: Type of guardrail (pii, secrets, injection, etc.)
        action: Action taken (check, violation, block)
        details: Additional details
    """
    # Guardrail metrics
    checks_counter = counter(
        "guardrails_checks",
        "Guardrails checks",
        labels=["type"]
    )
    violations_counter = counter(
        "guardrails_violations",
        "Guardrails violations",
        labels=["type"]
    )
    blocks_counter = counter(
        "guardrails_blocks",
        "Guardrails blocks",
        labels=["type"]
    )
    
    # Record based on action
    if action == "check":
        checks_counter.inc(type=guardrail_type)
    elif action == "violation":
        violations_counter.inc(type=guardrail_type)
        logger.warning(
            f"Guardrail violation: {guardrail_type}",
            guardrail=guardrail_type,
            details=details
        )
    elif action == "block":
        blocks_counter.inc(type=guardrail_type)
        logger.warning(
            f"Guardrail blocked: {guardrail_type}",
            guardrail=guardrail_type,
            details=details
        )


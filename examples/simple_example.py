#!/usr/bin/env python3
"""
Simple example of using the Test Generation Agent.

This example demonstrates basic usage of the agentic test generator
for a simple Python function.

NOTE: TestGenerationAgent is a compatibility wrapper around LangGraph's
create_react_agent. For new code, you can use the orchestrator directly:

    from src.orchestrator import create_test_generation_orchestrator
    orchestrator = create_test_generation_orchestrator()
    tests = orchestrator.generate_tests(target_code, file_path="example.py")
"""

from src.test_agent import TestGenerationAgent


def main():
    """Run simple example."""
    # Sample code to generate tests for
    target_code = '''
def calculate_discount(price: float, discount_percent: float) -> float:
    """
    Calculate discounted price.
    
    Args:
        price: Original price
        discount_percent: Discount percentage (0-100)
        
    Returns:
        Discounted price
        
    Raises:
        ValueError: If price is negative or discount is invalid
    """
    if price < 0:
        raise ValueError("Price cannot be negative")
    
    if not 0 <= discount_percent <= 100:
        raise ValueError("Discount must be between 0 and 100")
    
    discount_amount = price * (discount_percent / 100)
    return price - discount_amount
'''
    
    print("=" * 70)
    print("Simple Test Generation Example")
    print("=" * 70)
    print("\nTarget Code:")
    print(target_code)
    print("\n" + "=" * 70)
    
    # Initialize agent (uses LangGraph 1.0 orchestrator internally)
    agent = TestGenerationAgent(max_iterations=3)
    
    # Generate tests
    print("\nGenerating tests (using LangGraph create_react_agent)...\n")
    tests = agent.generate_tests(target_code)
    
    # Display results
    print("\n" + "=" * 70)
    print("Generated Tests:")
    print("=" * 70)
    print(tests)
    
    # Save to file
    output_file = "test_calculate_discount.py"
    with open(output_file, 'w') as f:
        f.write(tests)
    
    print(f"\n✓ Tests saved to: {output_file}")
    print("\nTo run the tests:")
    print(f"  pytest {output_file}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Compare different LLM providers for test generation.

This example demonstrates using different providers (Ollama, OpenAI, Gemini)
for the same test generation task.
"""

from src.llm_providers import LLMProviderFactory


def main():
    """Compare LLM providers."""
    # Sample code to generate tests for
    target_code = '''
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 0:
        raise ValueError("n must be positive")
    if n <= 2:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)
'''
    
    prompt = f"""Generate a simple pytest test for this function:

{target_code}

Include tests for:
- Valid inputs (n=1, n=5)
- Invalid input (n=0)
"""
    
    print("=" * 70)
    print("LLM Provider Comparison")
    print("=" * 70)
    
    # Try different providers
    providers = ["ollama", "openai", "gemini"]
    
    for provider_name in providers:
        print(f"\n{'='*70}")
        print(f"Testing Provider: {provider_name.upper()}")
        print(f"{'='*70}\n")
        
        try:
            # Create provider
            provider = LLMProviderFactory.create(provider_name)
            
            # Generate
            print(f"Generating with {provider_name}...")
            response = provider.generate(
                prompt=prompt,
                system="You are a test generation expert. Generate clean, working pytest code.",
                temperature=0.3
            )
            
            print(f"\n✓ Generated ({response.model}):")
            print(f"Tokens: {response.usage}")
            print(f"\nCode Preview:")
            print(response.content[:500])
            if len(response.content) > 500:
                print("...")
        
        except Exception as e:
            print(f"✗ Failed to use {provider_name}: {e}")
    
    print(f"\n{'='*70}")
    print("Comparison Complete")
    print(f"{'='*70}")
    print("\nNotes:")
    print("- Ollama requires local installation")
    print("- OpenAI requires OPENAI_API_KEY environment variable")
    print("- Gemini requires GOOGLE_API_KEY environment variable")
    print("\nSet provider with: export LLM_PROVIDER=openai")


if __name__ == "__main__":
    main()


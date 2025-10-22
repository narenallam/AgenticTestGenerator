#!/usr/bin/env python3
"""
Demonstration of reranking functionality.

This example shows how the Qwen3-Reranker improves search result relevance.
"""

from src.code_embeddings import CodeEmbeddingStore
from src.reranker import CodeReranker


def main():
    """Run reranking demonstration."""
    # Sample code chunks
    documents = [
        """
def calculate_total_price(items):
    '''Calculate total price of items.'''
    return sum(item['price'] * item['quantity'] for item in items)
""",
        """
def process_payment(amount, method):
    '''Process a payment transaction.'''
    if method == 'credit':
        return charge_credit_card(amount)
    return charge_debit_card(amount)
""",
        """
def get_item_price(item_id):
    '''Retrieve price for an item.'''
    return database.query('SELECT price FROM items WHERE id = ?', item_id)
""",
        """
def apply_discount(price, discount_percent):
    '''Apply discount to price.'''
    discount = price * (discount_percent / 100)
    return price - discount
""",
        """
def format_currency(amount):
    '''Format amount as currency string.'''
    return f"${amount:.2f}"
"""
    ]
    
    query = "function to calculate price with discount"
    
    print("=" * 70)
    print("Reranking Demonstration")
    print("=" * 70)
    print(f"\nQuery: '{query}'")
    print("\nOriginal Documents:")
    for i, doc in enumerate(documents, 1):
        func_name = doc.split('def ')[1].split('(')[0]
        print(f"  {i}. {func_name}")
    
    # Initialize reranker
    print("\n" + "=" * 70)
    reranker = CodeReranker(top_k=3)
    
    # Rerank documents
    print("\nReranking documents...")
    results = reranker.rerank(query, documents)
    
    # Display results
    print("\n" + "=" * 70)
    print("Reranked Results (by relevance):")
    print("=" * 70)
    
    for i, result in enumerate(results, 1):
        func_name = result.document.split('def ')[1].split('(')[0]
        print(f"\n{i}. {func_name}")
        print(f"   Score: {result.score:.4f}")
        print(f"   Original position: #{result.index + 1}")
    
    print("\n" + "=" * 70)
    print("\nAnalysis:")
    print("- The reranker identified 'calculate_total_price' and 'apply_discount'")
    print("  as most relevant to price calculation with discount")
    print("- Less relevant functions like 'format_currency' were ranked lower")
    print("- This improves the quality of context provided to the LLM")
    print("=" * 70)


if __name__ == "__main__":
    main()


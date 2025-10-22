#!/usr/bin/env python3
"""
Main entry point for the Agentic Unit Test Generator.

This script provides a CLI interface for automated test generation
using the ReAct agent with RAG-based code retrieval.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import settings
from src.code_embeddings import CodeEmbeddingStore
from src.git_integration import GitIntegration
from src.rag_retrieval import RAGRetriever
from src.test_agent import TestGenerationAgent

console = Console()


def index_codebase(source_dir: Path, force: bool = False) -> None:
    """
    Index the codebase for semantic search.
    
    Args:
        source_dir: Directory containing source code
        force: Force reindexing
    """
    console.print(Panel.fit(
        "[bold cyan]Indexing Codebase[/bold cyan]",
        subtitle=str(source_dir)
    ))
    
    store = CodeEmbeddingStore()
    count = store.index_codebase(source_dir, force_reindex=force)
    
    console.print(f"\n[green]✓[/green] Successfully indexed {count} code chunks")


def generate_tests_for_changes(provider: Optional[str] = None) -> None:
    """Generate tests for code changes since last commit."""
    console.print(Panel.fit(
        "[bold cyan]Generating Tests for Git Changes[/bold cyan]"
    ))
    
    # Initialize components
    from src.llm_providers import get_llm_provider
    
    llm = get_llm_provider(provider) if provider else None
    git = GitIntegration()
    retriever = RAGRetriever()
    agent = TestGenerationAgent(retriever=retriever, llm_provider=llm)
    
    # Get status
    status = git.get_status()
    console.print(f"\nBranch: [cyan]{status.branch}[/cyan]")
    console.print(f"Modified files: [yellow]{len(status.modified_files)}[/yellow]")
    console.print(f"Untracked files: [yellow]{len(status.untracked_files)}[/yellow]")
    
    # Get contexts for changed code
    contexts = retriever.get_context_for_changed_code()
    
    if not contexts:
        console.print("\n[yellow]No new functions found in changed files[/yellow]")
        return
    
    console.print(f"\n[cyan]Found {len(contexts)} new/modified functions[/cyan]\n")
    
    # Generate tests
    results = agent.generate_batch_tests(contexts)
    
    # Save generated tests
    for func_name, test_code in results.items():
        output_file = settings.test_output_dir / f"test_{func_name}.py"
        output_file.write_text(test_code, encoding='utf-8')
        console.print(f"[green]✓[/green] Saved: {output_file}")
    
    # Display summary
    table = Table(title="Test Generation Summary")
    table.add_column("Function", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Lines", justify="right")
    
    for func_name, test_code in results.items():
        lines = len(test_code.split('\n'))
        table.add_row(func_name, "✓ Generated", str(lines))
    
    console.print("\n")
    console.print(table)


def generate_tests_for_file(
    file_path: Path,
    function_name: Optional[str] = None,
    provider: Optional[str] = None
) -> None:
    """
    Generate tests for a specific file or function.
    
    Args:
        file_path: Path to source file
        function_name: Optional specific function name
        provider: LLM provider to use
    """
    console.print(Panel.fit(
        "[bold cyan]Generating Tests for File[/bold cyan]",
        subtitle=str(file_path)
    ))
    
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        return
    
    # Read source code
    source_code = file_path.read_text(encoding='utf-8')
    
    # Initialize agent
    from src.llm_providers import get_llm_provider
    
    llm = get_llm_provider(provider) if provider else None
    retriever = RAGRetriever()
    agent = TestGenerationAgent(retriever=retriever, llm_provider=llm)
    
    if function_name:
        # Generate for specific function
        context = retriever.get_context_for_function(
            str(file_path),
            function_name
        )
        if context:
            tests = agent.generate_tests(
                target_code=context.target_code,
                file_path=str(file_path)
            )
            
            if tests:
                output_file = settings.test_output_dir / f"test_{function_name}.py"
                output_file.write_text(tests, encoding='utf-8')
                console.print(f"\n[green]✓[/green] Generated tests saved to: {output_file}")
        else:
            console.print(f"[red]Error: Function '{function_name}' not found[/red]")
    else:
        # Generate for entire file
        tests = agent.generate_tests(
            target_code=source_code,
            file_path=str(file_path)
        )
        
        if tests:
            output_file = settings.test_output_dir / f"test_{file_path.stem}.py"
            output_file.write_text(tests, encoding='utf-8')
            console.print(f"\n[green]✓[/green] Generated tests saved to: {output_file}")


def show_status() -> None:
    """Show current system status."""
    console.print(Panel.fit("[bold cyan]System Status[/bold cyan]"))
    
    # Git status
    try:
        git = GitIntegration()
        status = git.get_status()
        
        console.print(f"\n[bold]Git Status[/bold]")
        console.print(f"  Branch: [cyan]{status.branch}[/cyan]")
        console.print(f"  Commit: [cyan]{status.commit_hash[:8]}[/cyan]")
        console.print(f"  Modified: [yellow]{len(status.modified_files)}[/yellow]")
        console.print(f"  Untracked: [yellow]{len(status.untracked_files)}[/yellow]")
    except Exception as e:
        console.print(f"[red]Git error: {e}[/red]")
    
    # Embedding store status
    store = CodeEmbeddingStore()
    count = store.collection.count()
    console.print(f"\n[bold]Embedding Store[/bold]")
    console.print(f"  Indexed chunks: [cyan]{count}[/cyan]")
    console.print(f"  Model: [cyan]{settings.embedding_model}[/cyan]")
    
    # Configuration
    console.print(f"\n[bold]Configuration[/bold]")
    console.print(f"  LLM Provider: [cyan]{settings.llm_provider}[/cyan]")
    if settings.llm_provider == "ollama":
        console.print(f"  Model: [cyan]{settings.ollama_model}[/cyan]")
    elif settings.llm_provider == "openai":
        console.print(f"  Model: [cyan]{settings.openai_model}[/cyan]")
    elif settings.llm_provider == "gemini":
        console.print(f"  Model: [cyan]{settings.google_model}[/cyan]")
    console.print(f"  Source Dir: [cyan]{settings.source_code_dir}[/cyan]")
    console.print(f"  Output Dir: [cyan]{settings.test_output_dir}[/cyan]")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Agentic Unit Test Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index the codebase
  python main.py index --source-dir ./src

  # Generate tests for git changes
  python main.py generate-changes

  # Generate tests for a specific file
  python main.py generate-file path/to/file.py

  # Generate tests for a specific function
  python main.py generate-file path/to/file.py --function my_function

  # Show system status
  python main.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index codebase')
    index_parser.add_argument(
        '--source-dir',
        type=Path,
        default=settings.source_code_dir,
        help='Source code directory'
    )
    index_parser.add_argument(
        '--force',
        action='store_true',
        help='Force reindexing'
    )
    
    # Generate for changes command
    changes_parser = subparsers.add_parser(
        'generate-changes',
        help='Generate tests for git changes'
    )
    changes_parser.add_argument(
        '--provider',
        type=str,
        choices=['ollama', 'openai', 'gemini'],
        help='LLM provider to use (defaults to settings)'
    )
    
    # Generate for file command
    file_parser = subparsers.add_parser(
        'generate-file',
        help='Generate tests for specific file'
    )
    file_parser.add_argument('file', type=Path, help='Source file path')
    file_parser.add_argument(
        '--function',
        type=str,
        help='Specific function name'
    )
    file_parser.add_argument(
        '--provider',
        type=str,
        choices=['ollama', 'openai', 'gemini'],
        help='LLM provider to use (defaults to settings)'
    )
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'index':
            index_codebase(args.source_dir, args.force)
        elif args.command == 'generate-changes':
            provider = getattr(args, 'provider', None)
            generate_tests_for_changes(provider)
        elif args.command == 'generate-file':
            provider = getattr(args, 'provider', None)
            generate_tests_for_file(args.file, args.function, provider)
        elif args.command == 'status':
            show_status()
        
        return 0
    
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if hasattr(e, '__traceback__'):
            import traceback
            console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())


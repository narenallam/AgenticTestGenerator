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
from src.orchestrator import create_test_generation_orchestrator, TestGenerationConfig
from src.symbol_index import create_symbol_index

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
        subtitle=str(source_dir.absolute())
    ))
    
    # Verify directory exists
    if not source_dir.exists():
        console.print(f"[red]✗[/red] Directory does not exist: {source_dir}")
        console.print(f"[yellow]Hint:[/yellow] Check SOURCE_CODE_DIR in your .env file")
        return
    
    # Show configuration
    console.print(f"\n[bold]Indexing Configuration[/bold]")
    console.print(f"  Source Directory: [cyan]{source_dir.absolute()}[/cyan]")
    console.print(f"  Force Reindex: [cyan]{force}[/cyan]")
    
    # Count Python files
    py_files = list(source_dir.rglob("*.py"))
    console.print(f"  Python Files Found: [cyan]{len(py_files)}[/cyan]\n")
    
    # 1. Symbol indexing (NEW - for exact lookups)
    console.print("[bold]1. Building Symbol Index[/bold]")
    symbol_index = create_symbol_index()
    symbol_stats = symbol_index.index_directory(source_dir)
    console.print(f"  [green]✓[/green] Indexed {symbol_stats['symbols_found']} symbols\n")
    
    # 2. Semantic indexing (embeddings)
    console.print("[bold]2. Building Semantic Index[/bold]")
    store = CodeEmbeddingStore()
    count = store.index_codebase(source_dir, force_reindex=force)
    console.print(f"  [green]✓[/green] Indexed {count} code chunks\n")
    
    console.print(f"[green]✓[/green] Indexing complete!")
    console.print(f"  → Symbol index: {symbol_stats['functions']} functions, {symbol_stats['classes']} classes")
    console.print(f"  → Semantic index: {count} chunks")


def generate_tests_for_changes(provider: Optional[str] = None) -> None:
    """Generate tests for code changes since last commit using LangChain 1.0."""
    console.print(Panel.fit(
        "[bold cyan]Generating Tests for Git Changes[/bold cyan]"
    ))

    # Display configured source directory
    console.print(f"\n[bold]Configuration[/bold]")
    console.print(f"  Source Directory: [cyan]{settings.source_code_dir}[/cyan]")
    console.print(f"  Test Output Directory: [cyan]{settings.test_output_dir}[/cyan]")
    console.print(f"  Orchestrator: [cyan]LangChain 1.0[/cyan]")

    # Initialize components
    from src.llm_providers import get_llm_provider

    llm = get_llm_provider(provider) if provider else None
    git = GitIntegration(repo_path=settings.source_code_dir)
    retriever = RAGRetriever(source_dir=settings.source_code_dir)

    # Use LangChain 1.0 orchestrator
    console.print(f"\n[green]🚀 Using LangChain 1.0 Orchestrator[/green]")

    # Create LangChain 1.0 orchestrator with optimal config
    config = TestGenerationConfig(
        max_iterations=settings.max_iterations,
        enable_hitl=False,  # Disabled by default
        enable_summarization=False,  # Disabled by default
        enable_pii_redaction=False  # Disabled by default
    )

    orchestrator = create_test_generation_orchestrator(config=config)
    agent = orchestrator
    
    # Get status
    status = git.get_status()
    console.print(f"\n[bold]Git Status[/bold]")
    console.print(f"  Branch: [cyan]{status.branch}[/cyan]")
    console.print(f"  Commit: [cyan]{status.commit_hash[:8]}[/cyan]")
    
    # Display modified files with details
    if status.modified_files:
        console.print(f"\n[bold yellow]Modified Files ({len(status.modified_files)}):[/bold yellow]")
        for file_path in status.modified_files:
            # Get diff stats
            try:
                diff_stats = git.repo.git.diff('--numstat', file_path)
                if diff_stats:
                    parts = diff_stats.split('\t')
                    if len(parts) >= 2:
                        additions = parts[0] if parts[0] != '-' else '0'
                        deletions = parts[1] if parts[1] != '-' else '0'
                        console.print(
                            f"  • [cyan]{file_path}[/cyan] "
                            f"[green]+{additions}[/green] "
                            f"[red]-{deletions}[/red]"
                        )
                    else:
                        console.print(f"  • [cyan]{file_path}[/cyan]")
                else:
                    console.print(f"  • [cyan]{file_path}[/cyan]")
            except Exception:
                console.print(f"  • [cyan]{file_path}[/cyan]")
    
    # Display staged files
    if status.staged_files:
        console.print(f"\n[bold green]Staged Files ({len(status.staged_files)}):[/bold green]")
        for file_path in status.staged_files:
            console.print(f"  • [green]{file_path}[/green]")
    
    # Display untracked files
    if status.untracked_files:
        console.print(f"\n[bold magenta]Untracked Files ({len(status.untracked_files)}):[/bold magenta]")
        for file_path in status.untracked_files:
            # Get file size
            try:
                full_path = git.repo_root / file_path
                if full_path.exists():
                    size = full_path.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size/1024:.1f}KB"
                    else:
                        size_str = f"{size/(1024*1024):.1f}MB"
                    console.print(f"  • [magenta]{file_path}[/magenta] ({size_str})")
                else:
                    console.print(f"  • [magenta]{file_path}[/magenta]")
            except Exception:
                console.print(f"  • [magenta]{file_path}[/magenta]")
    
    # Get contexts for changed code
    contexts = retriever.get_context_for_changed_code()

    if not contexts:
        console.print("\n[yellow]No new functions found in changed files[/yellow]")
        return

    console.print(f"\n[cyan]Found {len(contexts)} new/modified functions to test[/cyan]\n")

    # Generate tests using LangChain 1.0 orchestrator
    results = {}
    for context in contexts:
        console.print(f"  → Generating tests for [cyan]{context.function_name}[/cyan]...")

        try:
            tests = agent.generate_tests(
                target_code=context.target_code,
                file_path=context.file_path,
                function_name=context.function_name,
                context=context.related_code_text
            )

            if tests:
                results[context.function_name] = tests
                console.print(f"    [green]✓[/green] Generated {len(tests.splitlines())} lines")
            else:
                console.print(f"    [red]✗[/red] No tests generated")

        except Exception as e:
            console.print(f"    [red]✗[/red] Error: {e}")
            continue
    
    # Save generated tests
    for func_name, test_code in results.items():
        output_file = settings.test_output_dir / f"test_{func_name}.py"
        output_file.write_text(test_code, encoding='utf-8')
        console.print(f"[green]✓[/green] Saved: {output_file}")
    
    # Display summary
    table = Table(title="📊 Test Generation Summary")
    table.add_column("Function", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Test Lines", justify="right")
    
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
    Generate tests for a specific file or function using LangChain 1.0.
    
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

    # Display configuration
    console.print(f"\n[bold]Configuration[/bold]")
    console.print(f"  Source Directory: [cyan]{settings.source_code_dir}[/cyan]")
    console.print(f"  Test Output Directory: [cyan]{settings.test_output_dir}[/cyan]")
    console.print(f"  Orchestrator: [cyan]LangChain 1.0[/cyan]")

    # Read source code
    source_code = file_path.read_text(encoding='utf-8')

    # Initialize agent
    from src.llm_providers import get_llm_provider

    llm = get_llm_provider(provider) if provider else None
    retriever = RAGRetriever(source_dir=settings.source_code_dir)

    # Use LangChain 1.0 orchestrator
    console.print(f"\n[green]🚀 Using LangChain 1.0 Orchestrator[/green]")

    config = TestGenerationConfig(
        max_iterations=settings.max_iterations,
        enable_hitl=False,
        enable_summarization=False,
        enable_pii_redaction=False
    )

    orchestrator = create_test_generation_orchestrator(config=config)
    agent = orchestrator
    
    if function_name:
        # Generate for specific function
        context = retriever.get_context_for_function(
            str(file_path),
            function_name
        )
        if context:
            tests = agent.generate_tests(
                target_code=context.target_code,
                file_path=str(file_path),
                function_name=function_name,
                context=context.related_code_text
            )

            if tests:
                output_file = settings.test_output_dir / f"test_{function_name}.py"
                output_file.write_text(tests, encoding='utf-8')
                console.print(f"\n[green]✓[/green] Generated tests saved to: {output_file}")
            else:
                console.print(f"\n[yellow]⚠️[/yellow] No tests generated for function '{function_name}'")
        else:
            console.print(f"[red]Error: Function '{function_name}' not found[/red]")
    else:
        # Generate for entire file
        tests = agent.generate_tests(
            target_code=source_code,
            file_path=str(file_path),
            function_name=None,
            context=""
        )

        if tests:
            output_file = settings.test_output_dir / f"test_{file_path.stem}.py"
            output_file.write_text(tests, encoding='utf-8')
            console.print(f"\n[green]✓[/green] Generated tests saved to: {output_file}")
        else:
            console.print(f"\n[yellow]⚠️[/yellow] No tests generated for file '{file_path}'")


def show_status() -> None:
    """Show current system status."""
    console.print(Panel.fit("[bold cyan]System Status[/bold cyan]"))
    
    # Configuration
    console.print(f"\n[bold]Configuration[/bold]")
    console.print(f"  Source Directory: [cyan]{settings.source_code_dir}[/cyan]")
    console.print(f"  Test Output Directory: [cyan]{settings.test_output_dir}[/cyan]")
    console.print(f"  LLM Provider: [cyan]{settings.llm_provider}[/cyan]")
    console.print(f"  Orchestrator: [cyan]LangChain 1.0[/cyan]")
    
    # Git status
    try:
        git = GitIntegration(repo_path=settings.source_code_dir)
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


def show_coverage() -> None:
    """Show test coverage statistics from tracking database."""
    from src.test_tracking_db import create_test_tracking_db
    
    console.print(Panel.fit(
        "[bold cyan]📊 Test Coverage Report[/bold cyan]",
        subtitle="Function-Level Tracking"
    ))
    
    # Load database
    db = create_test_tracking_db()
    stats = db.get_coverage_stats()
    
    # Overall stats
    console.print("\n[bold]Overall Coverage[/bold]")
    console.print(f"  Total Functions:       [cyan]{stats['total_functions']}[/cyan]")
    console.print(f"  With Tests:            [green]{stats['functions_with_tests']}[/green]")
    console.print(f"  Without Tests:         [yellow]{stats['functions_without_tests']}[/yellow]")
    console.print(f"  Total Test Cases:      [cyan]{stats['total_test_cases']}[/cyan]")
    console.print(f"  Coverage:              [bold cyan]{stats['coverage_percentage']:.1f}%[/bold cyan]")
    
    # Per-file breakdown
    if stats['files']:
        console.print("\n[bold]Per-File Coverage[/bold]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("File", style="cyan")
        table.add_column("Functions", justify="right")
        table.add_column("Tested", justify="right", style="green")
        table.add_column("Tests", justify="right")
        table.add_column("Coverage", justify="right")
        
        for file_stats in stats['files']:
            coverage = file_stats['coverage']
            coverage_style = (
                "green" if coverage >= 80 else
                "yellow" if coverage >= 50 else
                "red"
            )
            
            table.add_row(
                file_stats['file'],
                str(file_stats['total_functions']),
                str(file_stats['tested_functions']),
                str(file_stats['total_tests']),
                f"[{coverage_style}]{coverage:.1f}%[/{coverage_style}]"
            )
        
        console.print(table)
    
    # Show functions needing tests
    untested = db.get_functions_needing_tests(limit=10)
    if untested:
        console.print("\n[bold yellow]⚠️  Top 10 Functions Needing Tests[/bold yellow]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("File", style="cyan")
        table.add_column("Function", style="yellow")
        table.add_column("Lines", justify="right")
        
        for func in untested:
            table.add_row(
                func.file_path,
                func.function_name,
                f"{func.start_line}-{func.end_line}"
            )
        
        console.print(table)
        console.print("\n[dim]Run 'make generate' to create tests[/dim]")
    else:
        console.print("\n[bold green]✓ All functions have tests![/bold green]")
    
    console.print()
    db.close()


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
    
    # Coverage command
    subparsers.add_parser('coverage', help='Show test coverage statistics')
    
    # Smart generation command
    smart_parser = subparsers.add_parser(
        'generate-smart',
        help='Smart test generation with LLM analysis and auto-indexing'
    )
    smart_parser.add_argument(
        '--no-llm-analysis',
        action='store_true',
        help='Skip LLM-powered code analysis'
    )
    smart_parser.add_argument(
        '--no-auto-index',
        action='store_true',
        help='Skip auto-indexing before generation'
    )
    smart_parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum files to process (for testing)'
    )
    
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
        elif args.command == 'coverage':
            show_coverage()
        elif args.command == 'generate-smart':
            # Preload Ollama models if using Ollama
            if settings.llm_provider == "ollama":
                from src.ollama_manager import preload_qwen_models
                console.print("\n[bold cyan]🚀 Optimizing Ollama Performance[/bold cyan]")
                preload_qwen_models(keep_alive="30m")
            
            from src.test_generator import create_test_generator
            generator = create_test_generator()
            generator.generate(
                auto_index=not args.no_auto_index,
                analyze_with_llm=not args.no_llm_analysis,
                max_files=args.max_files
            )
        
        return 0
    
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if hasattr(e, '__traceback__'):
            import traceback
            console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())


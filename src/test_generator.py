"""
Test Generator with LLM-powered change analysis.

Integrates:
1. Intelligent change tracking
2. LLM-powered analysis
3. Incremental indexing before generation
4. Test lifecycle management (create/update/delete)
"""

from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import settings
from src.code_embeddings import CodeEmbeddingStore
from src.git_integration import GitIntegration
from src.orchestrator import (
    create_test_generation_orchestrator,
    TestGenerationConfig
)
from src.indexer import create_indexer
from src.test_tracking_db import create_test_tracking_db
from src.console_tracker import get_tracker
from src.context_assembler import create_context_assembler
from src.symbol_index import create_symbol_index
from src.hybrid_search import create_hybrid_search

console = Console()


class TestGenerator:
    """
    Test generator with intelligence and automation.
    
    Features:
    - Auto-indexes before generation
    - LLM-powered change analysis
    - Tracks source/test relationships
    - Manages test lifecycle automatically
    """
    
    def __init__(
        self,
        source_dir: Optional[Path] = None,
        test_dir: Optional[Path] = None,
        llm_provider=None
    ):
        """
        Initialize smart test generator.
        
        Args:
            source_dir: Source code directory
            test_dir: Test output directory
            llm_provider: LLM provider instance
        """
        self.source_dir = source_dir or settings.source_code_dir
        self.test_dir = test_dir or settings.test_output_dir
        
        # Initialize components
        self.tracking_db = create_test_tracking_db()
        self.git = None
        
        # Try to initialize Git
        try:
            self.git = GitIntegration(repo_path=self.source_dir)
        except Exception as e:
            console.print(f"[yellow]⚠️  Git not available: {e}[/yellow]")
        
        # Initialize symbol index and hybrid search (NEW)
        console.print("[dim]→ Initializing symbol index...[/dim]")
        self.symbol_index = create_symbol_index()
        
        embedding_store = CodeEmbeddingStore()
        console.print("[dim]→ Initializing hybrid search...[/dim]")
        self.hybrid_search = create_hybrid_search(
            symbol_index=self.symbol_index,
            embedding_store=embedding_store,
            use_reranking=True
        )
        
        # Initialize indexer with new components
        self.indexer = create_indexer(
            self.source_dir,
            symbol_index=self.symbol_index,
            tracking_db=self.tracking_db
        )
        
        # Initialize context assembler with ALL components
        self.context_assembler = create_context_assembler(
            embedding_store=embedding_store,
            git_integration=self.git,
            tracking_db=self.tracking_db,
            symbol_index=self.symbol_index,  # NEW
            hybrid_search=self.hybrid_search  # NEW
        )
        
        console.print("[green]✓[/green] Test Generator initialized")
        
        # Check if DB needs initial sync
        self._ensure_db_synced()
    
    def _ensure_db_synced(self) -> None:
        """Ensure database is synced with codebase on first run."""
        # Check if database is empty
        stats = self.tracking_db.get_coverage_stats()
        
        if stats['total_functions'] == 0:
            console.print("\n[yellow]📊 First run detected - initializing database...[/yellow]")
            console.print("[dim]This will scan all source code and detect existing tests[/dim]")
            
            # Sync database
            sync_stats = self.tracking_db.sync_from_codebase(
                source_dir=self.source_dir,
                test_dir=self.test_dir,
                file_extensions=['.py']
            )
            
            # Show what needs tests
            untested = self.tracking_db.get_functions_needing_tests(limit=10)
            if untested:
                console.print(f"\n[yellow]⚠️  Found {len(untested)} functions without tests[/yellow]")
                console.print("[dim]Run 'make generate' to create tests for them[/dim]\n")
        else:
            console.print(f"[dim]📊 Database has {stats['total_functions']} functions tracked[/dim]")
    
    def _auto_index(self, force: bool = False) -> Dict:
        """
        Auto-index: Only re-indexes changed files.
        
        Args:
            force: Force reindex of all files
            
        Returns:
            Dictionary with indexing statistics
        """
        tracker = get_tracker()
        
        with tracker.component_section("INDEXER", "Incremental Code Indexing"):
            tracker.component_progress("indexer", f"Source directory: {self.source_dir}", "info")
            tracker.component_progress("indexer", f"Force reindex: {force}", "info")
            
            stats = self.indexer.index(
                source_dir=self.source_dir,
                file_extensions={'.py'},
                force=force
            )
            
            tracker.component_progress(
                "indexer",
                f"Indexed {stats.get('indexed', 0)} files, "
                f"Skipped {stats.get('skipped', 0)} unchanged",
                "success"
            )
        
        return stats
    
    def generate(
        self,
        auto_index: bool = True,
        analyze_with_llm: bool = True,
        max_files: Optional[int] = None
    ) -> dict:
        """
        Generate tests based on code changes.
        
        Args:
            auto_index: Auto-index before generation
            analyze_with_llm: Use LLM for change analysis
            max_files: Maximum files to process (for testing)
            
        Returns:
            Dictionary with generation statistics
        """
        console.print(Panel.fit(
            "[bold cyan]🚀 Test Generation[/bold cyan]",
            subtitle="LLM-Powered Intelligence"
        ))
        
        # Step 1: Auto-index
        if auto_index:
            self._auto_index()
        
        # Step 2: Check database for functions needing tests
        tracker = get_tracker()
        
        tracker.section_header("DATABASE", "Function-Level Tracking Query", "📊")
        tracker.component_start("tracker", "Querying test tracking database")
        untested_functions = self.tracking_db.get_functions_needing_tests()
        
        if untested_functions:
            tracker.component_progress(
                "tracker",
                f"Found {len(untested_functions)} functions without tests",
                "warning"
            )
            # Convert function info to file paths for processing
            modified_files = list(set([
                self.source_dir / func.file_path
                for func in untested_functions
            ]))
            tracker.component_progress(
                "tracker",
                f"Will process {len(modified_files)} files",
                "info"
            )
            
            # Show some examples
            for i, func in enumerate(untested_functions[:3]):
                tracker.component_progress(
                    "tracker",
                    f"  → {func.file_path}::{func.function_name} (lines {func.start_line}-{func.end_line})",
                    "info"
                )
        else:
            # Step 2b: Get changes from Git (if available)
            modified_files = None
            if self.git:
                try:
                    console.print("\n[cyan]🔍 Detecting changes from Git...[/cyan]")
                    changes = self.git.get_changed_files_since_last_commit(file_extensions={'.py'})
                    # Convert relative paths to absolute paths
                    modified_files = [
                        self.source_dir / change.file_path if not Path(change.file_path).is_absolute()
                        else Path(change.file_path)
                        for change in changes
                    ]
                    console.print(f"[green]✓[/green] Found {len(modified_files)} changed files from Git")
                except Exception as e:
                    console.print(f"[yellow]⚠️  Could not get Git changes: {e}[/yellow]")
                    console.print("[cyan]→ Scanning entire source directory instead[/cyan]")
        
        # Step 3: Analyze changes - convert modified files to actions
        console.print("\n[cyan]🤖 Analyzing changes...[/cyan]")
        analyses = self._analyze_changes_from_files(modified_files)
        
        if not analyses:
            console.print("[yellow]✨ No changes detected - everything is up to date![/yellow]")
            return self._get_statistics()
        
        console.print(f"[green]✓[/green] Found {len(analyses)} files needing test updates")
        
        # Limit if requested
        if max_files:
            analyses = analyses[:max_files]
        
        # Step 5: Display plan
        self._display_plan(analyses)
        
        # Step 6: Execute plan
        results = self._execute_plan(analyses)
        
        # Step 7: Cleanup orphaned tests
        console.print("\n[cyan]🧹 Cleaning up orphaned tests...[/cyan]")
        # Collect all existing source files
        existing_files = [str(f) for f in self.source_dir.rglob("*.py")]
        cleanup_stats = self.tracking_db.cleanup_deleted_files(existing_files)
        total_removed = sum(cleanup_stats.values())
        if total_removed > 0:
            console.print(f"[green]✓[/green] Removed {total_removed} orphaned records")
        else:
            console.print("[green]✓[/green] No orphaned tests found")
        
        # Step 8: Show statistics
        stats = self._get_statistics()
        self._display_statistics(stats, results)
        
        return stats
    
    def _analyze_changes_from_files(self, modified_files: List[Path]) -> List[Dict]:
        """
        Convert modified files to test actions.
        
        For each modified source file:
        - Determine corresponding test file path
        - Check if test exists
        - Return action (CREATE or UPDATE)
        """
        from dataclasses import dataclass
        
        @dataclass
        class AnalysisResult:
            action: str
            source_file: Path
            test_file: Path
            reason: str
            priority: int = 1
        
        analyses = []
        
        for source_file in modified_files:
            if not source_file.exists():
                continue
            
            # Determine test file path
            rel_path = source_file.relative_to(self.source_dir)
            test_rel_path = Path(f"test_{rel_path.stem}{rel_path.suffix}")
            test_file = self.test_dir / test_rel_path
            
            # Check if test exists
            if test_file.exists():
                action = "UPDATE"
                reason = "Source file modified, updating tests"
            else:
                action = "CREATE"
                reason = "No test file exists, creating new tests"
            
            analyses.append(AnalysisResult(
                action=action,
                source_file=source_file,
                test_file=test_file,
                reason=reason,
                priority=1 if action == "CREATE" else 2
            ))
        
        # Sort by priority (CREATE first)
        analyses.sort(key=lambda x: x.priority)
        
        return analyses
    
    def _display_plan(self, analyses: list) -> None:
        """Display the test generation plan."""
        console.print("\n[bold cyan]📋 Test Generation Plan[/bold cyan]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Priority", style="cyan", width=8)
        table.add_column("Action", width=10)
        table.add_column("Source File", style="green")
        table.add_column("Reason", style="yellow")
        
        for analysis in analyses:
            action_style = {
                "CREATE": "green",
                "UPDATE": "yellow",
                "DELETE": "red",
                "SKIP": "dim"
            }
            style = action_style.get(analysis.action, "white")
            
            priority_emoji = {1: "🔴", 2: "🟡", 3: "🟢"}
            
            table.add_row(
                f"{priority_emoji[analysis.priority]} P{analysis.priority}",
                f"[{style}]{analysis.action.upper()}[/{style}]",
                analysis.source_file.name,
                analysis.reason[:60]
            )
        
        console.print(table)
        console.print()
    
    def _execute_plan(self, analyses: list) -> dict:
        """Execute the test generation plan."""
        console.print("\n[bold cyan]⚡ Executing Plan[/bold cyan]\n")
        
        results = {
            "created": 0,
            "updated": 0,
            "deleted": 0,
            "skipped": 0,
            "errors": 0
        }
        
        # Initialize orchestrator once
        config = TestGenerationConfig(
            max_iterations=settings.max_iterations,
            enable_hitl=False,
            enable_summarization=False,
            enable_pii_redaction=False
        )
        orchestrator = create_test_generation_orchestrator(config=config)
        
        for analysis in analyses:
            try:
                if analysis.action == "CREATE":
                    self._create_test(analysis, orchestrator)
                    results["created"] += 1
                elif analysis.action == "UPDATE":
                    self._update_test(analysis, orchestrator)
                    results["updated"] += 1
                elif analysis.action == "DELETE":
                    self._delete_test(analysis)
                    results["deleted"] += 1
                else:
                    results["skipped"] += 1
            except Exception as e:
                console.print(f"[red]✗[/red] Error processing {analysis.source_file.name}: {e}")
                results["errors"] += 1
        
        return results
    
    def _create_test(self, analysis, orchestrator) -> None:
        """Create a new test file."""
        console.print(f"[green]✨ Creating test for {analysis.source_file.name}[/green]")
        
        # Read source code
        source_code = analysis.source_file.read_text(encoding='utf-8')
        
        # Extract module name and function names for proper imports
        import ast
        module_name = analysis.source_file.stem  # e.g., "main" from "main.py"
        functions = []
        try:
            tree = ast.parse(source_code)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        except:
            pass
        
        # Assemble comprehensive context
        assembled_context = self.context_assembler.assemble(
            source_code=source_code,
            file_path=str(analysis.source_file),
            function_name=None,
            max_related=5
        )
        
        # Format context for LLM
        context_section = assembled_context.to_llm_prompt_section()
        
        # Add specific import instructions
        context_section += f"\n\n=== IMPORTANT TEST FILE REQUIREMENTS ===\n"
        context_section += f"1. DO NOT copy/paste the source code into the test file\n"
        context_section += f"2. Use proper imports at the top of the file:\n"
        if functions:
            context_section += f"   from {module_name} import {', '.join(functions)}\n"
        else:
            context_section += f"   from {module_name} import <function_names>\n"
        context_section += f"3. Generate comprehensive tests for ALL functions:\n"
        if functions:
            context_section += "   - " + "\n   - ".join(functions) + "\n"
        context_section += f"4. Use pytest framework with proper assertions\n"
        context_section += f"5. Include edge cases, error cases, and normal cases\n\n"
        
        # Generate tests with rich context
        tests = orchestrator.generate_tests(
            target_code=source_code,
            file_path=str(analysis.source_file),
            function_name=None,
            context=context_section
        )
        
        # Save tests
        analysis.test_file.parent.mkdir(parents=True, exist_ok=True)
        analysis.test_file.write_text(tests, encoding='utf-8')
        
        # Update database - resync to record new tests
        self.tracking_db.sync_from_codebase(
            source_dir=self.source_dir,
            test_dir=self.test_dir,
            file_extensions=['.py']
        )
        
        console.print(f"  [green]✓[/green] Created {analysis.test_file.name}")
    
    def _update_test(self, analysis, orchestrator) -> None:
        """Update an existing test file by merging with new tests."""
        console.print(f"[yellow]🔄 Updating test for {analysis.source_file.name}[/yellow]")
        
        # Read source code
        source_code = analysis.source_file.read_text(encoding='utf-8')
        
        # Read existing test file
        existing_tests = ""
        if analysis.test_file.exists():
            existing_tests = analysis.test_file.read_text(encoding='utf-8')
        
        # Assemble comprehensive context
        assembled_context = self.context_assembler.assemble(
            source_code=source_code,
            file_path=str(analysis.source_file),
            function_name=None,
            max_related=5
        )
        
        # Format context for LLM
        context_section = assembled_context.to_llm_prompt_section()
        
        # Add existing tests and update instructions
        context_section += f"\n\n=== EXISTING TEST FILE ===\n{existing_tests}\n\n"
        context_section += """=== UPDATE INSTRUCTIONS ===
You are UPDATING existing tests. Follow these rules:
1. DO NOT duplicate the source code in the test file
2. Use proper imports: from main import knapsack, factorial, add
3. Keep all existing test functions that are still valid
4. Add new test functions for any new functions in the source code
5. Update test functions if the source function signature changed
6. Remove test functions if the source function was deleted
7. Fix any import issues (e.g., don't redefine functions, import them)
8. Maintain the same test structure and style

Current source file has these functions:
"""
        # Extract function names from source
        import ast
        try:
            tree = ast.parse(source_code)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            context_section += "- " + "\n- ".join(functions) + "\n\n"
            context_section += "Make sure there are tests for ALL of these functions.\n"
        except:
            pass
        
        # Generate updated tests with rich context
        tests = orchestrator.generate_tests(
            target_code=source_code,
            file_path=str(analysis.source_file),
            function_name=None,
            context=context_section
        )
        
        # Save updated tests
        analysis.test_file.write_text(tests, encoding='utf-8')
        
        # Update database
        self.tracking_db.sync_from_codebase(
            source_dir=self.source_dir,
            test_dir=self.test_dir,
            file_extensions=['.py']
        )
        
        console.print(f"  [green]✓[/green] Updated {analysis.test_file.name}")
    
    def _delete_test(self, analysis) -> None:
        """Delete an orphaned test file."""
        console.print(f"[red]🗑️  Deleting orphaned test {analysis.test_file.name}[/red]")
        
        if analysis.test_file.exists():
            analysis.test_file.unlink()
            console.print(f"  [green]✓[/green] Deleted {analysis.test_file.name}")
    
    def _get_statistics(self) -> dict:
        """Get current statistics."""
        # Get coverage stats from database
        coverage_stats = self.tracking_db.get_coverage_stats()
        
        files = coverage_stats.get("files", [])
        
        stats = {
            # Function-level stats
            "total_functions": coverage_stats["total_functions"],
            "tested_functions": coverage_stats["functions_with_tests"],
            "untested_functions": coverage_stats["functions_without_tests"],
            "coverage_percentage": round(coverage_stats["coverage_percentage"], 1),
            "total_test_cases": coverage_stats["total_test_cases"],
            
            # File-level stats (for display)
            "total_source_files": len(files),
            "total_test_files": coverage_stats["total_test_cases"],  # Approximate
            "sources_with_tests": sum(1 for f in files if f.get("tested_functions", 0) > 0),
            "total_files": len(files)
        }
        
        # Add Git stats if available
        if self.git:
            try:
                status = self.git.get_status()
                stats["git_branch"] = status.branch
                stats["git_modified"] = len(status.modified_files)
                stats["git_untracked"] = len(status.untracked_files)
            except Exception:
                pass
        
        return stats
    
    def _display_statistics(self, stats: dict, results: dict) -> None:
        """Display final statistics."""
        console.print("\n" + "="*70)
        console.print("[bold cyan]📊 Generation Statistics[/bold cyan]\n")
        
        # Results
        console.print("[bold]Actions Taken:[/bold]")
        console.print(f"  ✨ Created:  {results['created']}")
        console.print(f"  🔄 Updated:  {results['updated']}")
        console.print(f"  🗑️  Deleted:  {results['deleted']}")
        console.print(f"  ⏭️  Skipped:  {results['skipped']}")
        if results['errors'] > 0:
            console.print(f"  [red]✗ Errors:   {results['errors']}[/red]")
        
        # Overall stats
        console.print(f"\n[bold]Overall Coverage:[/bold]")
        console.print(f"  📁 Source Files:     {stats['total_source_files']}")
        console.print(f"  🧪 Test Files:       {stats['total_test_files']}")
        console.print(f"  ✓ With Tests:        {stats['sources_with_tests']}")
        console.print(f"  📊 Coverage:         {stats['coverage_percentage']}%")
        
        # Git stats
        if "git_branch" in stats:
            console.print(f"\n[bold]Git Status:[/bold]")
            console.print(f"  🌿 Branch:           {stats['git_branch']}")
            console.print(f"  📝 Modified:         {stats['git_modified']}")
            console.print(f"  ➕ Untracked:        {stats['git_untracked']}")
        
        console.print("\n" + "="*70 + "\n")


def create_test_generator(
    source_dir: Optional[Path] = None,
    test_dir: Optional[Path] = None,
    llm_provider=None
) -> TestGenerator:
    """Factory function to create test generator."""
    return TestGenerator(source_dir, test_dir, llm_provider)


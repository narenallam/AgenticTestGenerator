"""
Test Tracking Database.

Maintains persistent tracking of source functions and their test cases.
Provides function-level granularity for test coverage and lifecycle management.
"""

import ast
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel
from rich.console import Console

from config.settings import settings

console = Console()


class FunctionInfo(BaseModel):
    """Information about a source function."""
    id: Optional[int] = None
    file_path: str
    function_name: str
    function_hash: str
    start_line: int
    end_line: int
    last_modified: datetime
    has_test: bool = False
    test_file_path: Optional[str] = None
    test_count: int = 0


class TestCaseInfo(BaseModel):
    """Information about a test case."""
    id: Optional[int] = None
    test_file: str
    test_function: str
    source_function_id: int
    test_type: str  # unit, integration, etc.
    created_at: datetime
    last_updated: datetime
    is_passing: Optional[bool] = None


class TestTrackingDB:
    """
    Database for tracking source functions and their test cases.
    
    Provides:
    - Function-level tracking
    - Test case mapping
    - Coverage metrics
    - Historical data
    - Relationship management
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize test tracking database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or (settings.test_output_dir / ".test_tracking.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        self._create_tables()
        console.print(f"[dim]📊 Test tracking DB: {self.db_path.name}[/dim]")
    
    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Source functions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS source_functions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                function_name TEXT NOT NULL,
                function_hash TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                last_modified TIMESTAMP NOT NULL,
                has_test BOOLEAN DEFAULT FALSE,
                test_file_path TEXT,
                test_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(file_path, function_name)
            )
        """)
        
        # Test cases table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_file TEXT NOT NULL,
                test_function TEXT NOT NULL,
                source_function_id INTEGER NOT NULL,
                test_type TEXT DEFAULT 'unit',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_passing BOOLEAN,
                FOREIGN KEY (source_function_id) REFERENCES source_functions(id),
                UNIQUE(test_file, test_function)
            )
        """)
        
        # Coverage history table (optional - for tracking over time)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coverage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_functions INTEGER NOT NULL,
                functions_with_tests INTEGER NOT NULL,
                total_test_cases INTEGER NOT NULL,
                coverage_percentage REAL NOT NULL
            )
        """)
        
        # NEW: Index metadata (consolidates .index_metadata.json)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS index_metadata (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                last_indexed TIMESTAMP NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                embedding_version TEXT DEFAULT 'v1'
            )
        """)
        
        # NEW: File relationships (consolidates .test_relationships.json)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT NOT NULL,
                test_file TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                test_hash TEXT NOT NULL,
                last_synced TIMESTAMP NOT NULL,
                relationship_type TEXT DEFAULT 'test_for',
                UNIQUE(source_file, test_file)
            )
        """)
        
        # Create indices for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source_file 
            ON source_functions(file_path)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source_hash 
            ON source_functions(function_hash)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_source 
            ON test_cases(source_function_id)
        """)
        
        self.conn.commit()
    
    def add_or_update_function(
        self,
        file_path: str,
        function_name: str,
        function_code: str,
        start_line: int,
        end_line: int
    ) -> int:
        """
        Add or update a source function in the database.
        
        Args:
            file_path: Path to source file
            function_name: Name of the function
            function_code: Full function code
            start_line: Starting line number
            end_line: Ending line number
            
        Returns:
            Function ID
        """
        function_hash = hashlib.sha256(function_code.encode()).hexdigest()
        now = datetime.now()
        
        cursor = self.conn.cursor()
        
        # Check if function exists
        cursor.execute("""
            SELECT id, function_hash FROM source_functions
            WHERE file_path = ? AND function_name = ?
        """, (file_path, function_name))
        
        result = cursor.fetchone()
        
        if result:
            # Update existing function
            func_id = result['id']
            old_hash = result['function_hash']
            
            if old_hash != function_hash:
                # Function code changed
                cursor.execute("""
                    UPDATE source_functions
                    SET function_hash = ?, start_line = ?, end_line = ?,
                        last_modified = ?, updated_at = ?
                    WHERE id = ?
                """, (function_hash, start_line, end_line, now, now, func_id))
                console.print(f"[yellow]↻[/yellow] Updated function: {function_name}")
        else:
            # Insert new function
            cursor.execute("""
                INSERT INTO source_functions 
                (file_path, function_name, function_hash, start_line, end_line, last_modified)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (file_path, function_name, function_hash, start_line, end_line, now))
            func_id = cursor.lastrowid
            console.print(f"[green]+[/green] Added function: {function_name}")
        
        self.conn.commit()
        return func_id
    
    def add_test_case(
        self,
        test_file: str,
        test_function: str,
        source_function_id: int,
        test_type: str = "unit"
    ) -> int:
        """
        Add a test case to the database.
        
        Args:
            test_file: Path to test file
            test_function: Name of test function
            source_function_id: ID of source function being tested
            test_type: Type of test (unit, integration, etc.)
            
        Returns:
            Test case ID
        """
        cursor = self.conn.cursor()
        now = datetime.now()
        
        # Check if test exists
        cursor.execute("""
            SELECT id FROM test_cases
            WHERE test_file = ? AND test_function = ?
        """, (test_file, test_function))
        
        result = cursor.fetchone()
        
        if result:
            # Update existing test
            test_id = result['id']
            cursor.execute("""
                UPDATE test_cases
                SET source_function_id = ?, test_type = ?, last_updated = ?
                WHERE id = ?
            """, (source_function_id, test_type, now, test_id))
        else:
            # Insert new test
            cursor.execute("""
                INSERT INTO test_cases
                (test_file, test_function, source_function_id, test_type)
                VALUES (?, ?, ?, ?)
            """, (test_file, test_function, source_function_id, test_type))
            test_id = cursor.lastrowid
        
        # Update source function's test count
        cursor.execute("""
            UPDATE source_functions
            SET has_test = TRUE,
                test_file_path = ?,
                test_count = (
                    SELECT COUNT(*) FROM test_cases 
                    WHERE source_function_id = ?
                )
            WHERE id = ?
        """, (test_file, source_function_id, source_function_id))
        
        self.conn.commit()
        return test_id
    
    def get_function_tests(self, function_id: int) -> List[TestCaseInfo]:
        """Get all tests for a specific function."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM test_cases
            WHERE source_function_id = ?
            ORDER BY created_at DESC
        """, (function_id,))
        
        return [
            TestCaseInfo(
                id=row['id'],
                test_file=row['test_file'],
                test_function=row['test_function'],
                source_function_id=row['source_function_id'],
                test_type=row['test_type'],
                created_at=datetime.fromisoformat(row['created_at']),
                last_updated=datetime.fromisoformat(row['last_updated']),
                is_passing=row['is_passing']
            )
            for row in cursor.fetchall()
        ]
    
    def get_functions_without_tests(self) -> List[FunctionInfo]:
        """Get all functions that don't have tests."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM source_functions
            WHERE has_test = FALSE
            ORDER BY file_path, function_name
        """)
        
        return [self._row_to_function_info(row) for row in cursor.fetchall()]
    
    def get_file_functions(self, file_path: str) -> List[FunctionInfo]:
        """Get all functions in a specific file."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM source_functions
            WHERE file_path = ?
            ORDER BY start_line
        """, (file_path,))
        
        return [self._row_to_function_info(row) for row in cursor.fetchall()]
    
    def get_coverage_stats(self) -> Dict:
        """Get overall coverage statistics."""
        cursor = self.conn.cursor()
        
        # Total functions
        cursor.execute("SELECT COUNT(*) as count FROM source_functions")
        total_functions = cursor.fetchone()['count']
        
        # Functions with tests
        cursor.execute("""
            SELECT COUNT(*) as count FROM source_functions 
            WHERE has_test = TRUE
        """)
        functions_with_tests = cursor.fetchone()['count']
        
        # Total test cases
        cursor.execute("SELECT COUNT(*) as count FROM test_cases")
        total_tests = cursor.fetchone()['count']
        
        # Coverage percentage
        coverage_pct = (functions_with_tests / total_functions * 100) if total_functions > 0 else 0
        
        # Files breakdown
        cursor.execute("""
            SELECT 
                file_path,
                COUNT(*) as total_funcs,
                SUM(CASE WHEN has_test THEN 1 ELSE 0 END) as tested_funcs,
                SUM(test_count) as total_tests
            FROM source_functions
            GROUP BY file_path
            ORDER BY file_path
        """)
        
        files = []
        for row in cursor.fetchall():
            file_coverage = (row['tested_funcs'] / row['total_funcs'] * 100) if row['total_funcs'] > 0 else 0
            files.append({
                'file': row['file_path'],
                'total_functions': row['total_funcs'],
                'tested_functions': row['tested_funcs'],
                'total_tests': row['total_tests'],
                'coverage': file_coverage
            })
        
        return {
            'total_functions': total_functions,
            'functions_with_tests': functions_with_tests,
            'functions_without_tests': total_functions - functions_with_tests,
            'total_test_cases': total_tests,
            'coverage_percentage': coverage_pct,
            'files': files
        }
    
    def record_coverage_snapshot(self) -> None:
        """Record a coverage snapshot for historical tracking."""
        stats = self.get_coverage_stats()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO coverage_history
            (total_functions, functions_with_tests, total_test_cases, coverage_percentage)
            VALUES (?, ?, ?, ?)
        """, (
            stats['total_functions'],
            stats['functions_with_tests'],
            stats['total_test_cases'],
            stats['coverage_percentage']
        ))
        
        self.conn.commit()
    
    def delete_function(self, file_path: str, function_name: str) -> None:
        """Delete a function and its associated tests."""
        cursor = self.conn.cursor()
        
        # Get function ID
        cursor.execute("""
            SELECT id FROM source_functions
            WHERE file_path = ? AND function_name = ?
        """, (file_path, function_name))
        
        result = cursor.fetchone()
        if result:
            func_id = result['id']
            
            # Delete associated tests
            cursor.execute("DELETE FROM test_cases WHERE source_function_id = ?", (func_id,))
            
            # Delete function
            cursor.execute("DELETE FROM source_functions WHERE id = ?", (func_id,))
            
            self.conn.commit()
            console.print(f"[red]-[/red] Deleted function: {function_name}")
    
    def _row_to_function_info(self, row) -> FunctionInfo:
        """Convert database row to FunctionInfo object."""
        return FunctionInfo(
            id=row['id'],
            file_path=row['file_path'],
            function_name=row['function_name'],
            function_hash=row['function_hash'],
            start_line=row['start_line'],
            end_line=row['end_line'],
            last_modified=datetime.fromisoformat(row['last_modified']),
            has_test=bool(row['has_test']),
            test_file_path=row['test_file_path'],
            test_count=row['test_count']
        )
    
    def sync_from_codebase(
        self,
        source_dir: Path,
        test_dir: Path,
        file_extensions: Optional[List[str]] = None
    ) -> Dict:
        """
        Synchronize database with current codebase.
        
        Scans all source files, extracts functions, checks for tests,
        and populates the database. Use this for initial setup or resync.
        
        Args:
            source_dir: Source code directory
            test_dir: Test directory
            file_extensions: File extensions to scan (default: ['.py'])
            
        Returns:
            Statistics about the sync operation
        """
        if file_extensions is None:
            file_extensions = ['.py']
        
        console.print("\n[cyan]🔄 Syncing database with codebase...[/cyan]")
        
        stats = {
            'files_scanned': 0,
            'functions_found': 0,
            'functions_with_tests': 0,
            'functions_without_tests': 0,
            'test_cases_found': 0
        }
        
        # Scan source files
        source_files = []
        for ext in file_extensions:
            source_files.extend(source_dir.rglob(f"*{ext}"))
        
        console.print(f"[dim]Found {len(source_files)} source files[/dim]")
        
        for source_file in source_files:
            try:
                # Skip test files
                if source_file.name.startswith('test_') or '_test' in source_file.name:
                    continue
                
                # Parse the file
                code = source_file.read_text(encoding='utf-8')
                tree = ast.parse(code)
                
                # Extract functions
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Skip private functions (optional)
                        if node.name.startswith('_') and not node.name.startswith('__'):
                            continue
                        
                        # Get function code
                        func_lines = code.splitlines()[node.lineno - 1:node.end_lineno]
                        func_code = '\n'.join(func_lines)
                        
                        # Get relative path
                        rel_path = str(source_file.relative_to(source_dir))
                        
                        # Add to database
                        func_id = self.add_or_update_function(
                            file_path=rel_path,
                            function_name=node.name,
                            function_code=func_code,
                            start_line=node.lineno,
                            end_line=node.end_lineno or node.lineno
                        )
                        
                        stats['functions_found'] += 1
                        
                        # Check if test exists
                        test_file = self._find_test_file(source_file, test_dir)
                        if test_file and test_file.exists():
                            # Parse test file
                            test_code = test_file.read_text(encoding='utf-8')
                            test_tree = ast.parse(test_code)
                            
                            # Find test functions for this source function
                            test_funcs = self._find_test_functions(
                                test_tree,
                                node.name
                            )
                            
                            if test_funcs:
                                stats['functions_with_tests'] += 1
                                rel_test_path = str(test_file.relative_to(test_dir))
                                
                                for test_func in test_funcs:
                                    self.add_test_case(
                                        test_file=rel_test_path,
                                        test_function=test_func,
                                        source_function_id=func_id,
                                        test_type='unit'
                                    )
                                    stats['test_cases_found'] += 1
                            else:
                                stats['functions_without_tests'] += 1
                        else:
                            stats['functions_without_tests'] += 1
                
                stats['files_scanned'] += 1
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Error scanning {source_file.name}: {e}[/yellow]")
        
        # Display results
        console.print(f"\n[bold]Sync Results:[/bold]")
        console.print(f"  📁 Files scanned:          {stats['files_scanned']}")
        console.print(f"  🔧 Functions found:        {stats['functions_found']}")
        console.print(f"  ✅ With tests:             {stats['functions_with_tests']}")
        console.print(f"  ❌ Without tests:          {stats['functions_without_tests']}")
        console.print(f"  🧪 Test cases found:       {stats['test_cases_found']}")
        
        coverage = (stats['functions_with_tests'] / stats['functions_found'] * 100) if stats['functions_found'] > 0 else 0
        console.print(f"  📊 Coverage:               {coverage:.1f}%")
        
        # Record snapshot
        self.record_coverage_snapshot()
        
        console.print(f"[green]✓[/green] Database synced successfully!")
        
        return stats
    
    def _find_test_file(self, source_file: Path, test_dir: Path) -> Optional[Path]:
        """Find the corresponding test file for a source file."""
        # Convert source path to test path
        # e.g., src/calculator.py -> tests/test_calculator.py
        test_name = f"test_{source_file.stem}.py"
        
        # Try in the same relative directory structure
        try:
            rel_dir = source_file.parent.relative_to(source_file.parent.parent)
            test_file = test_dir / rel_dir / test_name
            if test_file.exists():
                return test_file
        except:
            pass
        
        # Try directly in test directory
        test_file = test_dir / test_name
        if test_file.exists():
            return test_file
        
        return None
    
    def _find_test_functions(self, test_tree: ast.AST, source_func_name: str) -> List[str]:
        """
        Find test functions that test a specific source function.
        
        Looks for test functions with names like:
        - test_<function_name>
        - test_<function_name>_*
        """
        test_functions = []
        
        for node in ast.walk(test_tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('test_'):
                    # Check if test name matches source function
                    test_suffix = node.name[5:]  # Remove 'test_' prefix
                    if test_suffix.startswith(source_func_name):
                        test_functions.append(node.name)
        
        return test_functions
    
    def get_functions_needing_tests(self, limit: Optional[int] = None) -> List[FunctionInfo]:
        """
        Get functions that need tests, prioritized by importance.
        
        Args:
            limit: Maximum number of functions to return
            
        Returns:
            List of functions without tests
        """
        cursor = self.conn.cursor()
        
        query = """
            SELECT * FROM source_functions
            WHERE has_test = FALSE
            ORDER BY last_modified DESC, file_path, function_name
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        
        return [self._row_to_function_info(row) for row in cursor.fetchall()]
    
    def update_index_metadata(
        self,
        file_path: str,
        file_hash: str,
        chunk_count: int
    ) -> None:
        """
        Update indexing metadata for a file.
        
        Args:
            file_path: Path to file
            file_hash: Hash of file content
            chunk_count: Number of chunks indexed
        """
        cursor = self.conn.cursor()
        now = datetime.now()
        
        cursor.execute("""
            INSERT OR REPLACE INTO index_metadata
            (file_path, file_hash, last_indexed, chunk_count)
            VALUES (?, ?, ?, ?)
        """, (file_path, file_hash, now, chunk_count))
        
        self.conn.commit()
    
    def get_index_metadata(self, file_path: str) -> Optional[Dict]:
        """Get indexing metadata for a file."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM index_metadata
            WHERE file_path = ?
        """, (file_path,))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def update_file_relationship(
        self,
        source_file: str,
        test_file: str,
        source_hash: str,
        test_hash: str
    ) -> None:
        """
        Update source-test file relationship.
        
        Args:
            source_file: Source file path
            test_file: Test file path
            source_hash: Hash of source file
            test_hash: Hash of test file
        """
        cursor = self.conn.cursor()
        now = datetime.now()
        
        cursor.execute("""
            INSERT OR REPLACE INTO file_relationships
            (source_file, test_file, source_hash, test_hash, last_synced)
            VALUES (?, ?, ?, ?, ?)
        """, (source_file, test_file, source_hash, test_hash, now))
        
        self.conn.commit()
    
    def get_test_file_for_source(self, source_file: str) -> Optional[str]:
        """Get test file path for a source file."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT test_file FROM file_relationships
            WHERE source_file = ?
            ORDER BY last_synced DESC
            LIMIT 1
        """, (source_file,))
        
        row = cursor.fetchone()
        if row:
            return row['test_file']
        return None
    
    def cleanup_deleted_files(self, existing_files: List[str]) -> Dict[str, int]:
        """
        Remove database records for deleted files.
        
        Args:
            existing_files: List of files that currently exist
            
        Returns:
            Dict with cleanup statistics
        """
        existing_set = set(existing_files)
        cursor = self.conn.cursor()
        
        stats = {
            'functions_removed': 0,
            'tests_removed': 0,
            'relationships_removed': 0,
            'index_metadata_removed': 0
        }
        
        # Find and remove deleted source files
        cursor.execute("SELECT DISTINCT file_path FROM source_functions")
        for row in cursor.fetchall():
            if row['file_path'] not in existing_set:
                # Remove function records
                cursor.execute("""
                    DELETE FROM source_functions
                    WHERE file_path = ?
                """, (row['file_path'],))
                stats['functions_removed'] += cursor.rowcount
        
        # Find and remove deleted test files
        cursor.execute("SELECT DISTINCT test_file FROM test_cases")
        for row in cursor.fetchall():
            if row['test_file'] not in existing_set:
                cursor.execute("""
                    DELETE FROM test_cases
                    WHERE test_file = ?
                """, (row['test_file'],))
                stats['tests_removed'] += cursor.rowcount
        
        # Clean relationships
        cursor.execute("""
            SELECT id, source_file, test_file FROM file_relationships
        """)
        for row in cursor.fetchall():
            if row['source_file'] not in existing_set or row['test_file'] not in existing_set:
                cursor.execute("""
                    DELETE FROM file_relationships
                    WHERE id = ?
                """, (row['id'],))
                stats['relationships_removed'] += 1
        
        # Clean index metadata
        cursor.execute("SELECT file_path FROM index_metadata")
        for row in cursor.fetchall():
            if row['file_path'] not in existing_set:
                cursor.execute("""
                    DELETE FROM index_metadata
                    WHERE file_path = ?
                """, (row['file_path'],))
                stats['index_metadata_removed'] += 1
        
        self.conn.commit()
        return stats
    
    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_test_tracking_db(db_path: Optional[Path] = None) -> TestTrackingDB:
    """Factory function to create test tracking database."""
    return TestTrackingDB(db_path)


"""
Git integration module for tracking code changes and deltas.

This module provides functionality to interact with Git repositories,
track file changes, and identify modified files since the last commit.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set

import git
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError
from pydantic import BaseModel, Field


class FileChange(BaseModel):
    """
    Represents a single file change in the repository.
    
    Attributes:
        file_path: Relative path to the changed file
        change_type: Type of change (added, modified, deleted, renamed)
        old_path: Original path for renamed files
        diff: Actual diff content
    """
    
    file_path: str = Field(..., description="Path to the changed file")
    change_type: str = Field(..., description="Type of change")
    old_path: Optional[str] = Field(default=None, description="Old path for renames")
    diff: Optional[str] = Field(default=None, description="Diff content")


class GitStatus(BaseModel):
    """
    Represents the current Git repository status.
    
    Attributes:
        is_dirty: Whether there are uncommitted changes
        untracked_files: List of untracked files
        modified_files: List of modified files
        staged_files: List of staged files
        branch: Current branch name
        commit_hash: Current commit hash
    """
    
    is_dirty: bool = Field(..., description="Has uncommitted changes")
    untracked_files: List[str] = Field(default_factory=list)
    modified_files: List[str] = Field(default_factory=list)
    staged_files: List[str] = Field(default_factory=list)
    branch: str = Field(..., description="Current branch name")
    commit_hash: str = Field(..., description="Current commit hash")


class GitIntegration:
    """
    Git integration handler for tracking and analyzing code changes.
    
    This class provides methods to interact with Git repositories,
    identify changes, and extract relevant code deltas for test generation.
    """
    
    def __init__(self, repo_path: Optional[Path] = None) -> None:
        """
        Initialize Git integration.
        
        Args:
            repo_path: Path to the Git repository. If None, uses current directory.
            
        Raises:
            InvalidGitRepositoryError: If the path is not a valid Git repository
        """
        self.repo_path = repo_path or Path.cwd()
        try:
            self.repo = Repo(self.repo_path, search_parent_directories=True)
            self.repo_root = Path(self.repo.working_dir)
        except InvalidGitRepositoryError as e:
            raise InvalidGitRepositoryError(
                f"Path {self.repo_path} is not a valid Git repository"
            ) from e
    
    def get_status(self) -> GitStatus:
        """
        Get the current Git repository status.
        
        Returns:
            GitStatus object containing repository state
            
        Example:
            >>> git = GitIntegration()
            >>> status = git.get_status()
            >>> print(f"Branch: {status.branch}, Dirty: {status.is_dirty}")
        """
        try:
            # Try to get staged files (requires at least one commit)
            staged_files = [item.a_path for item in self.repo.index.diff("HEAD")]
            commit_hash = self.repo.head.commit.hexsha
        except Exception:
            # Repository has no commits yet
            staged_files = []
            commit_hash = "0" * 40  # Git's null SHA
        
        try:
            branch_name = self.repo.active_branch.name
        except Exception:
            branch_name = "main"  # Default if no branch
        
        return GitStatus(
            is_dirty=self.repo.is_dirty(untracked_files=True),
            untracked_files=self.repo.untracked_files,
            modified_files=[item.a_path for item in self.repo.index.diff(None)],
            staged_files=staged_files,
            branch=branch_name,
            commit_hash=commit_hash
        )
    
    def get_changed_files_since_last_commit(
        self,
        file_extensions: Optional[Set[str]] = None
    ) -> List[FileChange]:
        """
        Get all files changed since the last commit.
        
        Args:
            file_extensions: Set of file extensions to filter (e.g., {'.py', '.js'})
            
        Returns:
            List of FileChange objects representing changes
            
        Example:
            >>> git = GitIntegration()
            >>> changes = git.get_changed_files_since_last_commit({'.py'})
            >>> for change in changes:
            ...     print(f"{change.change_type}: {change.file_path}")
        """
        if file_extensions is None:
            file_extensions = {'.py'}
        
        changes: List[FileChange] = []
        
        try:
            # Get staged changes (requires at least one commit)
            for diff_item in self.repo.index.diff("HEAD"):
                if self._should_include_file(diff_item.a_path, file_extensions):
                    changes.append(self._create_file_change(diff_item, staged=True))
        except Exception:
            # Repository has no commits yet - all files are new
            pass
        
        # Get unstaged changes
        for diff_item in self.repo.index.diff(None):
            if self._should_include_file(diff_item.a_path, file_extensions):
                changes.append(self._create_file_change(diff_item, staged=False))
        
        # Get untracked files
        for file_path in self.repo.untracked_files:
            if self._should_include_file(file_path, file_extensions):
                changes.append(FileChange(
                    file_path=file_path,
                    change_type="added",
                    diff=None
                ))
        
        return changes
    
    def get_file_content(self, file_path: str, commit: str = "HEAD") -> Optional[str]:
        """
        Get file content at a specific commit.
        
        Args:
            file_path: Path to the file relative to repo root
            commit: Commit hash or reference (default: HEAD)
            
        Returns:
            File content as string, or None if file doesn't exist
            
        Example:
            >>> git = GitIntegration()
            >>> content = git.get_file_content("src/main.py", "HEAD~1")
        """
        try:
            return self.repo.git.show(f"{commit}:{file_path}")
        except GitCommandError:
            return None
    
    def get_diff_for_file(
        self,
        file_path: str,
        base_commit: str = "HEAD",
        target_commit: Optional[str] = None
    ) -> Optional[str]:
        """
        Get diff for a specific file between commits.
        
        Args:
            file_path: Path to the file
            base_commit: Base commit for comparison
            target_commit: Target commit (None for working directory)
            
        Returns:
            Diff content as string
        """
        try:
            if target_commit:
                return self.repo.git.diff(base_commit, target_commit, file_path)
            else:
                return self.repo.git.diff(base_commit, file_path)
        except GitCommandError:
            return None
    
    def get_functions_in_diff(self, diff_content: str) -> List[str]:
        """
        Extract function names from diff content.
        
        Args:
            diff_content: Git diff content
            
        Returns:
            List of function names found in the diff
        """
        functions = []
        for line in diff_content.split('\n'):
            if line.startswith('@@') and 'def ' in line:
                # Extract function name from context line
                parts = line.split('def ')
                if len(parts) > 1:
                    func_name = parts[1].split('(')[0].strip()
                    functions.append(func_name)
            elif line.startswith('+') and 'def ' in line:
                # Extract from added lines
                parts = line.split('def ')
                if len(parts) > 1:
                    func_name = parts[1].split('(')[0].strip()
                    functions.append(func_name)
        
        return list(set(functions))
    
    def _should_include_file(
        self,
        file_path: str,
        file_extensions: Set[str]
    ) -> bool:
        """Check if file should be included based on extension."""
        return any(file_path.endswith(ext) for ext in file_extensions)
    
    def _create_file_change(
        self,
        diff_item,
        staged: bool
    ) -> FileChange:
        """Create FileChange object from git diff item."""
        change_type_map = {
            'A': 'added',
            'M': 'modified',
            'D': 'deleted',
            'R': 'renamed'
        }
        
        change_type = change_type_map.get(
            diff_item.change_type,
            'modified'
        )
        
        return FileChange(
            file_path=diff_item.a_path,
            change_type=change_type,
            old_path=diff_item.b_path if change_type == 'renamed' else None,
            diff=diff_item.diff.decode('utf-8') if diff_item.diff else None
        )
    
    def get_new_functions_since_commit(
        self,
        file_path: str,
        base_commit: str = "HEAD"
    ) -> List[Dict[str, str]]:
        """
        Identify new functions added to a file since a commit.
        
        Args:
            file_path: Path to the Python file
            base_commit: Base commit to compare against
            
        Returns:
            List of dictionaries with function info (name, signature, body)
        """
        import ast
        import re
        
        diff = self.get_diff_for_file(file_path, base_commit)
        if not diff:
            return []
        
        new_functions = []
        current_file = self.repo_root / file_path
        
        if not current_file.exists():
            return []
        
        try:
            with open(current_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    # Check if function appears in the diff
                    if f"+def {func_name}" in diff or f"+ def {func_name}" in diff:
                        # Extract function source
                        func_source = ast.get_source_segment(content, node)
                        new_functions.append({
                            'name': func_name,
                            'lineno': node.lineno,
                            'source': func_source or ''
                        })
        except (SyntaxError, FileNotFoundError):
            pass
        
        return new_functions


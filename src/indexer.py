"""
Smart Incremental Indexing System.

Tracks file hashes and modification times to avoid redundant re-indexing.
Only indexes files that have actually changed.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from config.settings import settings
from src.code_embeddings import CodeEmbeddingStore
from src.symbol_index import SymbolIndex
from src.test_tracking_db import TestTrackingDB

console = Console()


class FileIndexMetadata:
    """Metadata about an indexed file."""
    
    def __init__(
        self,
        file_path: str,
        file_hash: str,
        last_modified: float,
        last_indexed: str,
        chunk_count: int
    ):
        self.file_path = file_path
        self.file_hash = file_hash
        self.last_modified = last_modified
        self.last_indexed = last_indexed
        self.chunk_count = chunk_count
    
    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "last_modified": self.last_modified,
            "last_indexed": self.last_indexed,
            "chunk_count": self.chunk_count
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FileIndexMetadata":
        return cls(
            file_path=data["file_path"],
            file_hash=data["file_hash"],
            last_modified=data["last_modified"],
            last_indexed=data["last_indexed"],
            chunk_count=data["chunk_count"]
        )


class IncrementalIndexer:
    """
    Incremental indexer that tracks changes and only re-indexes modified files.
    
    Features:
    - File hash tracking to detect content changes
    - Modification time tracking
    - Persistent metadata storage
    - Incremental updates only
    - Statistics tracking
    """
    
    def __init__(
        self,
        source_dir: Optional[Path] = None,
        metadata_file: Optional[Path] = None,
        symbol_index: Optional[SymbolIndex] = None,
        tracking_db: Optional[TestTrackingDB] = None
    ):
        """
        Initialize incremental indexer.
        
        Args:
            source_dir: Source code directory
            metadata_file: File to store indexing metadata (deprecated, use DB)
            symbol_index: Symbol index for exact lookups (NEW)
            tracking_db: Test tracking DB for consolidated metadata (NEW)
        """
        self.source_dir = source_dir or settings.source_code_dir
        self.metadata_file = metadata_file or (settings.chroma_persist_dir / ".index_metadata.json")
        
        # Load existing metadata (deprecated - will migrate to DB)
        self.file_metadata: Dict[str, FileIndexMetadata] = {}
        self._load_metadata()
        
        # Initialize components
        self.embedding_store = None
        self.symbol_index = symbol_index  # NEW
        self.tracking_db = tracking_db  # NEW
    
    def _load_metadata(self) -> None:
        """Load indexing metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    for key, val in data.items():
                        self.file_metadata[key] = FileIndexMetadata.from_dict(val)
                console.print(f"[dim]Loaded metadata for {len(self.file_metadata)} files[/dim]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load index metadata: {e}[/yellow]")
    
    def _save_metadata(self) -> None:
        """Save indexing metadata to file."""
        try:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                data = {k: v.to_dict() for k, v in self.file_metadata.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save index metadata: {e}[/yellow]")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get SHA256 hash of file content."""
        try:
            content = file_path.read_bytes()
            return hashlib.sha256(content).hexdigest()
        except Exception:
            return ""
    
    def _needs_reindex(self, file_path: Path) -> bool:
        """
        Check if a file needs to be re-indexed.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file needs re-indexing
        """
        rel_path = str(file_path.relative_to(self.source_dir))
        
        # Check if file was indexed before
        if rel_path not in self.file_metadata:
            return True  # New file
        
        metadata = self.file_metadata[rel_path]
        
        # Check if file was modified
        try:
            current_mtime = file_path.stat().st_mtime
            if current_mtime > metadata.last_modified:
                # File was modified, check if content actually changed
                current_hash = self._get_file_hash(file_path)
                if current_hash != metadata.file_hash:
                    return True  # Content changed
        except Exception:
            return True  # Error, re-index to be safe
        
        return False  # File unchanged
    
    def index(
        self,
        source_dir: Optional[Path] = None,
        file_extensions: Optional[Set[str]] = None,
        force: bool = False
    ) -> Dict[str, int]:
        """
        Incremental indexing - only indexes changed files.
        
        Args:
            source_dir: Source directory to index
            file_extensions: File extensions to consider
            force: Force re-index of all files
            
        Returns:
            Dictionary with indexing statistics
        """
        source_dir = source_dir or self.source_dir
        if file_extensions is None:
            file_extensions = {'.py'}
        
        console.print("\n[cyan]🔍 Incremental Indexing[/cyan]")
        
        # Initialize embedding store if needed
        if self.embedding_store is None:
            self.embedding_store = CodeEmbeddingStore()
        
        # Find all source files
        all_files = []
        for ext in file_extensions:
            all_files.extend(source_dir.rglob(f"*{ext}"))
        
        console.print(f"[dim]Found {len(all_files)} total files[/dim]")
        
        # Determine which files need indexing
        if force:
            files_to_index = all_files
            console.print("[yellow]Force reindex: Indexing all files[/yellow]")
        else:
            files_to_index = [f for f in all_files if self._needs_reindex(f)]
            unchanged_count = len(all_files) - len(files_to_index)
            if unchanged_count > 0:
                console.print(f"[green]✓[/green] Skipping {unchanged_count} unchanged files")
        
        if not files_to_index:
            console.print("[green]✓[/green] All files up-to-date, no indexing needed")
            return {
                "total_files": len(all_files),
                "indexed": 0,
                "skipped": len(all_files),
                "total_chunks": 0
            }
        
        console.print(f"[cyan]→ Indexing {len(files_to_index)} changed/new files[/cyan]")
        
        # Index files that need it
        total_chunks = 0
        indexed_count = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(
                "[cyan]Processing files...",
                total=len(files_to_index)
            )
            
            for file_path in files_to_index:
                try:
                    # Remove old chunks for this file
                    rel_path = str(file_path.relative_to(source_dir))
                    self._remove_file_chunks(rel_path)
                    
                    # 1. Parse and index embeddings
                    chunks = self.embedding_store._parse_file(file_path, source_dir)
                    if chunks:
                        self.embedding_store._add_chunks(chunks)
                        total_chunks += len(chunks)
                    
                    # 2. Index symbols (NEW)
                    if self.symbol_index:
                        self.symbol_index.index_file(file_path)
                    
                    # 3. Update tracking DB metadata (NEW)
                    file_hash = self._get_file_hash(file_path)
                    if self.tracking_db:
                        self.tracking_db.update_index_metadata(
                            file_path=str(file_path),
                            file_hash=file_hash,
                            chunk_count=len(chunks) if chunks else 0
                        )
                    
                    # Update legacy metadata (deprecated)
                    self.file_metadata[rel_path] = FileIndexMetadata(
                        file_path=rel_path,
                        file_hash=file_hash,
                        last_modified=file_path.stat().st_mtime,
                        last_indexed=datetime.now().isoformat(),
                        chunk_count=len(chunks) if chunks else 0
                    )
                    
                    indexed_count += 1
                
                except Exception as e:
                    console.print(f"[red]✗[/red] Error indexing {file_path.name}: {e}")
                
                progress.advance(task)
        
        # Save metadata
        self._save_metadata()
        
        # Statistics
        stats = {
            "total_files": len(all_files),
            "indexed": indexed_count,
            "skipped": len(all_files) - indexed_count,
            "total_chunks": total_chunks
        }
        
        # Display results
        console.print(f"\n[bold]Indexing Results:[/bold]")
        console.print(f"  📁 Total files:    {stats['total_files']}")
        console.print(f"  ✨ Indexed:        {stats['indexed']} files ({stats['total_chunks']} chunks)")
        console.print(f"  ⏭️  Skipped:        {stats['skipped']} (unchanged)")
        
        if stats['indexed'] > 0:
            console.print(f"[green]✓[/green] Indexing complete!")
        else:
            console.print(f"[green]✓[/green] Index is up-to-date!")
        
        return stats
    
    def _remove_file_chunks(self, rel_path: str) -> None:
        """Remove all chunks for a specific file from the index."""
        try:
            # Get all chunks for this file
            results = self.embedding_store.collection.get(
                where={"file_path": rel_path}
            )
            
            if results['ids']:
                # Delete them
                self.embedding_store.collection.delete(ids=results['ids'])
        except Exception as e:
            console.print(f"[dim]Note: Could not remove old chunks for {rel_path}: {e}[/dim]")
    
    def get_index_statistics(self) -> Dict:
        """Get statistics about the current index."""
        total_files = len(self.file_metadata)
        total_chunks = sum(m.chunk_count for m in self.file_metadata.values())
        
        # Get collection size
        collection_count = 0
        try:
            collection_count = self.embedding_store.collection.count()
        except Exception:
            pass
        
        return {
            "tracked_files": total_files,
            "total_chunks": total_chunks,
            "collection_size": collection_count,
            "metadata_file": str(self.metadata_file)
        }
    
    def clear_metadata(self) -> None:
        """Clear all indexing metadata."""
        self.file_metadata.clear()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        console.print("[yellow]Cleared all indexing metadata[/yellow]")


def create_indexer(
    source_dir: Optional[Path] = None,
    symbol_index: Optional[SymbolIndex] = None,
    tracking_db: Optional[TestTrackingDB] = None
) -> IncrementalIndexer:
    """
    Factory function to create an incremental indexer.
    
    Args:
        source_dir: Source directory to index
        symbol_index: Symbol index for exact lookups (NEW)
        tracking_db: Tracking DB for consolidated metadata (NEW)
    """
    return IncrementalIndexer(source_dir, None, symbol_index, tracking_db)


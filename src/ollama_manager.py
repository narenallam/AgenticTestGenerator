"""
Ollama Model Manager for Optimized Model Loading.

Handles model preloading, keep-alive configuration, and smart batching
to minimize model switching delays.
"""

import time
from typing import List, Optional, Dict
import requests
from rich.console import Console

from config.settings import settings

console = Console()


class OllamaModelManager:
    """
    Manages Ollama models for optimal performance.
    
    Features:
    - Model preloading on startup
    - Keep-alive configuration
    - Model switching tracking
    - Warm-up utilities
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        keep_alive: str = "30m"
    ):
        """
        Initialize Ollama model manager.
        
        Args:
            base_url: Ollama API base URL
            keep_alive: How long to keep models in memory (e.g., "30m", "1h")
        """
        self.base_url = base_url
        self.keep_alive = keep_alive
        self.loaded_models: Dict[str, float] = {}  # model_name -> load_time
        self.model_switches = 0
        
        console.print(f"[dim]📦 Ollama Manager initialized (keep_alive={keep_alive})[/dim]")
    
    def preload_models(self, models: List[str], show_progress: bool = True):
        """
        Preload models to keep them in memory.
        
        Args:
            models: List of model names to preload
            show_progress: Show progress messages
        """
        if show_progress:
            # First check what's already loaded
            already_loaded = []
            for model in models:
                if self.is_model_loaded(model):
                    already_loaded.append(model)
            
            if already_loaded:
                console.print(f"\n[dim]ℹ  {len(already_loaded)} model(s) already in memory[/dim]")
            
            to_load = len(models) - len(already_loaded)
            if to_load > 0:
                console.print(f"[cyan]🔥 Loading {to_load} additional model(s)...[/cyan]")
            else:
                console.print(f"[green]✓[/green] All {len(models)} models already loaded!")
        
        loaded_count = 0
        skipped_count = 0
        
        for model in models:
            try:
                was_loaded = self.is_model_loaded(model)
                self.load_model(model, show_progress=show_progress)
                if was_loaded:
                    skipped_count += 1
                else:
                    loaded_count += 1
            except Exception as e:
                console.print(f"[yellow]⚠️  Failed to preload {model}: {e}[/yellow]")
        
        if show_progress:
            if loaded_count > 0:
                console.print(f"[green]✓[/green] Loaded {loaded_count} new model(s)")
            if skipped_count > 0:
                console.print(f"[dim]✓ Skipped {skipped_count} already loaded model(s)[/dim]")
    
    def load_model(self, model_name: str, show_progress: bool = True):
        """
        Load a specific model into memory.
        
        Args:
            model_name: Name of the model to load
            show_progress: Show progress messages
        """
        # First check if already loaded in Ollama
        if self.is_model_loaded(model_name):
            if show_progress:
                console.print(f"  [dim]✓ {model_name} already in memory (skipped)[/dim]")
            self.loaded_models[model_name] = 0  # Track as loaded
            return
        
        # Also check our internal cache
        if model_name in self.loaded_models:
            if show_progress:
                console.print(f"  [dim]✓ {model_name} already tracked (skipped)[/dim]")
            return
        
        if show_progress:
            console.print(f"  [cyan]→[/cyan] Loading {model_name}...", end=" ")
        
        start_time = time.time()
        
        try:
            # Determine if this is an embedding model
            is_embedding = "embedding" in model_name.lower() or "embed" in model_name.lower()
            is_reranker = "rerank" in model_name.lower()
            
            if is_embedding:
                # For embedding models, use /api/embeddings
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": model_name,
                        "prompt": "test",
                        "keep_alive": self.keep_alive
                    },
                    timeout=120
                )
            elif is_reranker:
                # For reranker models, try generate with minimal prompt
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "test",
                        "stream": False,
                        "keep_alive": self.keep_alive
                    },
                    timeout=120
                )
            else:
                # For text generation models
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "test",
                        "stream": False,
                        "keep_alive": self.keep_alive
                    },
                    timeout=120
                )
            
            if response.status_code == 200:
                load_time = time.time() - start_time
                self.loaded_models[model_name] = load_time
                
                if show_progress:
                    console.print(f"[green]✓[/green] ({load_time:.1f}s)")
            else:
                if show_progress:
                    console.print(f"[red]✗[/red] Status {response.status_code}")
                    # Try to parse error message
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            console.print(f"    [dim]Error: {error_data['error']}[/dim]")
                    except:
                        pass
        
        except Exception as e:
            if show_progress:
                console.print(f"[red]✗[/red] {str(e)[:50]}")
    
    def switch_model(self, from_model: str, to_model: str, show_progress: bool = True):
        """
        Track model switches.
        
        Args:
            from_model: Current model
            to_model: Target model
            show_progress: Show progress messages
        """
        if from_model != to_model:
            self.model_switches += 1
            
            if show_progress:
                console.print(
                    f"  [yellow]🔄 Model switch[/yellow]: "
                    f"{from_model} → {to_model} "
                    f"(#{self.model_switches})"
                )
            
            # Ensure target model is loaded
            if to_model not in self.loaded_models:
                self.load_model(to_model, show_progress=show_progress)
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models from Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/ps", timeout=5)
            if response.status_code == 200:
                data = response.json()
                # /api/ps shows models currently in memory
                return [model['name'] for model in data.get('models', [])]
        except Exception:
            pass
        return []
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded in memory."""
        try:
            response = requests.get(f"{self.base_url}/api/ps", timeout=5)
            if response.status_code == 200:
                data = response.json()
                loaded = [model['name'] for model in data.get('models', [])]
                return model_name in loaded
        except Exception:
            pass
        return False
    
    def show_status(self):
        """Display current model status."""
        console.print("\n[bold]📊 Ollama Model Status[/bold]")
        
        loaded = self.get_loaded_models()
        console.print(f"  Models in memory: [cyan]{len(loaded)}[/cyan]")
        
        for model in loaded:
            load_time = self.loaded_models.get(model, 0)
            time_str = f"({load_time:.1f}s)" if load_time > 0 else ""
            console.print(f"    • {model} {time_str}")
        
        console.print(f"  Model switches: [yellow]{self.model_switches}[/yellow]")
        console.print()
    
    def configure_keep_alive(self, duration: str):
        """
        Configure keep_alive duration for models.
        
        Args:
            duration: Duration string (e.g., "30m", "1h", "0" for immediate unload)
        """
        self.keep_alive = duration
        console.print(f"[green]✓[/green] Keep-alive set to {duration}")


# Global manager instance
_manager: Optional[OllamaModelManager] = None


def get_ollama_manager(
    base_url: str = "http://localhost:11434",
    keep_alive: str = "30m"
) -> OllamaModelManager:
    """Get or create global Ollama manager."""
    global _manager
    if _manager is None:
        _manager = OllamaModelManager(base_url=base_url, keep_alive=keep_alive)
    return _manager


def preload_qwen_models(keep_alive: str = "30m"):
    """
    Preload all Qwen models used by the system.
    
    Args:
        keep_alive: Duration to keep models in memory
    """
    manager = get_ollama_manager(keep_alive=keep_alive)
    
    qwen_models = [
        settings.ollama_embedding_model,  # qwen3-embedding:8b
        settings.ollama_model,            # qwen3-coder:30b
        "dengcao/Qwen3-Reranker-8B:Q8_0"  # Reranker
    ]
    
    # Remove duplicates
    qwen_models = list(set(qwen_models))
    
    manager.preload_models(qwen_models, show_progress=True)
    manager.show_status()
    
    return manager


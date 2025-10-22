"""
Dataset management for evaluations.

Handles creation, loading, and management of evaluation datasets including:
- Synthetic datasets
- Real-world datasets
- Adversarial datasets
"""

import json
import random
from pathlib import Path
from typing import List, Optional

from ..base import BaseDatasetLoader, EvalDataset


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic Dataset Generator
# ═══════════════════════════════════════════════════════════════════════════


class SyntheticDatasetGenerator:
    """Generate synthetic code examples for evaluation."""
    
    # Sample code templates
    SIMPLE_TEMPLATES = [
        {
            "name": "add_numbers",
            "code": """def add_numbers(a: int, b: int) -> int:
    \"\"\"Add two numbers together.\"\"\"
    return a + b""",
            "description": "Simple addition function",
            "complexity": "simple",
            "expected_test_count": 3,
        },
        {
            "name": "is_even",
            "code": """def is_even(n: int) -> bool:
    \"\"\"Check if a number is even.\"\"\"
    return n % 2 == 0""",
            "description": "Check if number is even",
            "complexity": "simple",
            "expected_test_count": 3,
        },
        {
            "name": "find_max",
            "code": """def find_max(numbers: list) -> int:
    \"\"\"Find the maximum number in a list.\"\"\"
    if not numbers:
        raise ValueError("List cannot be empty")
    return max(numbers)""",
            "description": "Find max with error handling",
            "complexity": "simple",
            "expected_test_count": 4,
        },
    ]
    
    MEDIUM_TEMPLATES = [
        {
            "name": "calculate_average",
            "code": """def calculate_average(numbers: list) -> float:
    \"\"\"
    Calculate the average of a list of numbers.
    
    Args:
        numbers: List of numbers
    
    Returns:
        Average value
    
    Raises:
        ValueError: If list is empty
        TypeError: If list contains non-numeric values
    \"\"\"
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    
    try:
        total = sum(numbers)
        return total / len(numbers)
    except TypeError as e:
        raise TypeError("List must contain only numeric values") from e""",
            "description": "Calculate average with comprehensive error handling",
            "complexity": "medium",
            "expected_test_count": 6,
        },
        {
            "name": "validate_email",
            "code": """import re

def validate_email(email: str) -> bool:
    \"\"\"
    Validate email address format.
    
    Args:
        email: Email address to validate
    
    Returns:
        True if valid, False otherwise
    \"\"\"
    if not email or not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))""",
            "description": "Email validation with regex",
            "complexity": "medium",
            "expected_test_count": 7,
        },
        {
            "name": "fibonacci",
            "code": """def fibonacci(n: int) -> int:
    \"\"\"
    Calculate the nth Fibonacci number.
    
    Args:
        n: Position in Fibonacci sequence (0-indexed)
    
    Returns:
        nth Fibonacci number
    
    Raises:
        ValueError: If n is negative
    \"\"\"
    if n < 0:
        raise ValueError("n must be non-negative")
    
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b""",
            "description": "Fibonacci with iteration",
            "complexity": "medium",
            "expected_test_count": 6,
        },
    ]
    
    COMPLEX_TEMPLATES = [
        {
            "name": "lru_cache",
            "code": """from collections import OrderedDict
from typing import Any, Callable, Optional

class LRUCache:
    \"\"\"Least Recently Used (LRU) cache implementation.\"\"\"
    
    def __init__(self, capacity: int):
        \"\"\"
        Initialize LRU cache with given capacity.
        
        Args:
            capacity: Maximum number of items to store
        
        Raises:
            ValueError: If capacity is not positive
        \"\"\"
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        \"\"\"Get value for key, updating access order.\"\"\"
        if key not in self.cache:
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        \"\"\"Put key-value pair, evicting LRU if needed.\"\"\"
        if key in self.cache:
            # Update and move to end
            self.cache.move_to_end(key)
        
        self.cache[key] = value
        
        # Evict LRU if over capacity
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    
    def size(self) -> int:
        \"\"\"Return current cache size.\"\"\"
        return len(self.cache)
    
    def clear(self) -> None:
        \"\"\"Clear all items from cache.\"\"\"
        self.cache.clear()""",
            "description": "LRU cache implementation",
            "complexity": "complex",
            "expected_test_count": 12,
        },
    ]
    
    ADVERSARIAL_TEMPLATES = [
        {
            "name": "unsafe_sql_query",
            "code": """def execute_user_query(user_id: str, table: str) -> str:
    \"\"\"Execute SQL query (UNSAFE - for testing only).\"\"\"
    # SECURITY VULNERABILITY: SQL injection
    query = f"SELECT * FROM {table} WHERE user_id = '{user_id}'"
    return query""",
            "description": "SQL injection vulnerability",
            "complexity": "simple",
            "category": "adversarial",
            "tags": ["security", "sql_injection"],
            "known_bugs": ["SQL injection vulnerability"],
        },
        {
            "name": "insecure_random",
            "code": """import random

def generate_session_token() -> str:
    \"\"\"Generate session token (INSECURE - for testing).\"\"\"
    # SECURITY VULNERABILITY: Uses weak random
    token = ''.join(random.choices('0123456789abcdef', k=32))
    return token""",
            "description": "Weak randomness for security",
            "complexity": "simple",
            "category": "adversarial",
            "tags": ["security", "weak_random"],
            "known_bugs": ["Uses random instead of secrets module"],
        },
    ]
    
    def generate(self, count: int = 10, complexity: Optional[str] = None) -> List[EvalDataset]:
        """
        Generate synthetic dataset.
        
        Args:
            count: Number of examples to generate
            complexity: Filter by complexity (simple, medium, complex, adversarial)
        
        Returns:
            List of evaluation dataset entries
        """
        datasets = []
        
        # Select templates based on complexity
        if complexity == "simple":
            templates = self.SIMPLE_TEMPLATES
        elif complexity == "medium":
            templates = self.MEDIUM_TEMPLATES
        elif complexity == "complex":
            templates = self.COMPLEX_TEMPLATES
        elif complexity == "adversarial":
            templates = self.ADVERSARIAL_TEMPLATES
        else:
            # Mix of all
            templates = (
                self.SIMPLE_TEMPLATES +
                self.MEDIUM_TEMPLATES +
                self.COMPLEX_TEMPLATES +
                self.ADVERSARIAL_TEMPLATES
            )
        
        # Generate dataset entries
        for i in range(count):
            template = random.choice(templates)
            
            dataset_entry = EvalDataset(
                id=f"synthetic_{template['complexity']}_{i}",
                name=f"{template['name']}_{i}",
                source_code=template["code"],
                language="python",
                complexity=template["complexity"],
                category=template.get("category", "synthetic"),
                tags=template.get("tags", []),
                description=template["description"],
                expected_test_count=template.get("expected_test_count"),
                known_bugs=template.get("known_bugs", []),
            )
            
            datasets.append(dataset_entry)
        
        return datasets


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Loader
# ═══════════════════════════════════════════════════════════════════════════


class JSONDatasetLoader(BaseDatasetLoader):
    """Load and save datasets in JSON format."""
    
    def load(self, dataset_path: Path) -> List[EvalDataset]:
        """Load dataset from JSON file."""
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        datasets = [EvalDataset(**entry) for entry in data]
        return datasets
    
    def save(self, dataset: List[EvalDataset], output_path: Path) -> None:
        """Save dataset to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [d.model_dump() for d in dataset]
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Manager
# ═══════════════════════════════════════════════════════════════════════════


class DatasetManager:
    """Manage evaluation datasets."""
    
    def __init__(self, dataset_dir: Path):
        """
        Initialize dataset manager.
        
        Args:
            dataset_dir: Root directory for datasets
        """
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = JSONDatasetLoader()
        self.generator = SyntheticDatasetGenerator()
    
    def create_synthetic_dataset(
        self,
        name: str,
        count: int = 10,
        complexity: Optional[str] = None,
    ) -> List[EvalDataset]:
        """
        Create and save a synthetic dataset.
        
        Args:
            name: Dataset name
            count: Number of examples
            complexity: Complexity level filter
        
        Returns:
            List of dataset entries
        """
        dataset = self.generator.generate(count=count, complexity=complexity)
        
        # Save to file
        output_path = self.dataset_dir / f"{name}.json"
        self.loader.save(dataset, output_path)
        
        print(f"✅ Created synthetic dataset: {output_path}")
        print(f"   Entries: {len(dataset)}")
        
        return dataset
    
    def load_dataset(self, name: str) -> List[EvalDataset]:
        """
        Load a dataset by name.
        
        Args:
            name: Dataset name (without .json extension)
        
        Returns:
            List of dataset entries
        """
        dataset_path = self.dataset_dir / f"{name}.json"
        return self.loader.load(dataset_path)
    
    def list_datasets(self) -> List[str]:
        """List all available datasets."""
        json_files = list(self.dataset_dir.glob("*.json"))
        return [f.stem for f in json_files]
    
    def get_dataset_info(self, name: str) -> dict:
        """
        Get information about a dataset.
        
        Args:
            name: Dataset name
        
        Returns:
            Dictionary with dataset statistics
        """
        dataset = self.load_dataset(name)
        
        complexities = {}
        categories = {}
        languages = {}
        
        for entry in dataset:
            complexities[entry.complexity] = complexities.get(entry.complexity, 0) + 1
            categories[entry.category] = categories.get(entry.category, 0) + 1
            languages[entry.language] = languages.get(entry.language, 0) + 1
        
        return {
            "name": name,
            "total_entries": len(dataset),
            "complexities": complexities,
            "categories": categories,
            "languages": languages,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════


def create_default_datasets(dataset_dir: Path) -> None:
    """Create default evaluation datasets."""
    manager = DatasetManager(dataset_dir)
    
    print("\n🔨 Creating default evaluation datasets...")
    
    # Simple dataset
    manager.create_synthetic_dataset(
        name="simple",
        count=10,
        complexity="simple"
    )
    
    # Medium dataset
    manager.create_synthetic_dataset(
        name="medium",
        count=10,
        complexity="medium"
    )
    
    # Complex dataset
    manager.create_synthetic_dataset(
        name="complex",
        count=5,
        complexity="complex"
    )
    
    # Adversarial dataset
    manager.create_synthetic_dataset(
        name="adversarial",
        count=5,
        complexity="adversarial"
    )
    
    # Mixed dataset
    manager.create_synthetic_dataset(
        name="mixed",
        count=30,
        complexity=None
    )
    
    print("\n✅ Default datasets created successfully!")


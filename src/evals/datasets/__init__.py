"""Dataset management for evaluations."""

from .dataset_manager import (
    DatasetManager,
    JSONDatasetLoader,
    SyntheticDatasetGenerator,
    create_default_datasets,
)

__all__ = [
    "DatasetManager",
    "SyntheticDatasetGenerator",
    "JSONDatasetLoader",
    "create_default_datasets",
]


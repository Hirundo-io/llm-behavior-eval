"""Native Inspect benchmarks and preset catalog."""

from .presets import (
    DatasetPreset,
    InspectTaskMapping,
    expand_dataset_preset,
    list_dataset_presets,
)

__all__ = [
    "DatasetPreset",
    "InspectTaskMapping",
    "expand_dataset_preset",
    "list_dataset_presets",
]
__version__ = "0.2.0"

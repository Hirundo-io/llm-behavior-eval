"""Native Inspect benchmarks for LLM behavior, and their preset catalog.

The benchmarks are Inspect tasks (``inspect list tasks llm_behavior_eval``). The
package exports only the preset catalog, which maps legacy preset names to task
names and ``-T`` arguments. The legacy engine and its Python API were removed in
0.2.0; see CHANGELOG.md.
"""

from .presets import (
    DatasetPreset,
    InspectTaskMapping,
    expand_dataset_preset,
    get_dataset_preset,
    list_dataset_presets,
)

__all__ = [
    "DatasetPreset",
    "InspectTaskMapping",
    "expand_dataset_preset",
    "get_dataset_preset",
    "list_dataset_presets",
]

__version__ = "0.2.0"

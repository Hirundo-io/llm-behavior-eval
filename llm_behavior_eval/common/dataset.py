"""Load a complete source split and convert it to an Inspect dataset.

Sample limits are Inspect's job (``--limit`` / ``eval(limit=...)`` or a run
config's ``eval_config.limit``), so conversion always keeps every record.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample


def load_hf_dataset(
    dataset_id: str,
    *,
    dataset_path: str | None = None,
    dataset_token: str | None = None,
) -> Any:
    """Load the full split of an HF dataset or a ``save_to_disk`` snapshot.

    ``dataset_path`` overrides where data is read from without changing the
    logical ``dataset_id``. Selects ``train`` when present, otherwise requires
    exactly one split. ``dataset_token`` is passed only to HF Hub loading.
    """
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

    source = dataset_path or dataset_id
    path = Path(source)
    if path.is_dir() and (
        (path / "dataset_dict.json").exists() or (path / "state.json").exists()
    ):
        raw = load_from_disk(source)
    else:
        raw = load_dataset(source, token=dataset_token)
    if isinstance(raw, Dataset):
        return raw
    if not isinstance(raw, DatasetDict):
        raise ValueError(f"Expected Dataset or DatasetDict for {dataset_id}")
    if "train" in raw:
        return raw["train"]
    if len(raw) == 1:
        return next(iter(raw.values()))
    raise ValueError(
        f"Expected train or exactly one split for {dataset_id}; found {list(raw)}"
    )


def to_inspect_dataset(
    records: Any,
    record_to_sample: Callable[[dict[str, Any]], Sample],
    *,
    dataset_id: str,
) -> MemoryDataset:
    """Convert every record, in source order, after benchmark validation."""
    samples = [record_to_sample(dict(records[i])) for i in range(len(records))]
    return MemoryDataset(samples, name=dataset_id)

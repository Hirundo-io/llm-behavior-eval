"""Load the complete source before benchmark validation and sample limiting."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample


def sample_limit(max_samples: int | None = 500) -> int | None:
    """Value for inspect_ai.eval(limit=...); nonpositive/None means the full split."""
    return max_samples if max_samples is not None and max_samples > 0 else None


def load_hf_dataset(
    dataset_id: str,
    *,
    dataset_path: str | None = None,
    dataset_token: str | None = None,
) -> Any:
    """Preserve logical identity; support HF sources and save_to_disk snapshots."""
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
    max_samples: int | None = 500,
) -> MemoryDataset:
    """Convert after full-source validation; task cap applies before Inspect CLI --limit."""
    limit = sample_limit(max_samples)
    samples = [
        record_to_sample(dict(records[i]))
        for i in range(min(len(records), limit) if limit else len(records))
    ]
    return MemoryDataset(samples, name=dataset_id)

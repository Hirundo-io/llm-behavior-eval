"""Load a benchmark's samples from the pinned Hub dataset or a local source.

* A pinned Hub load (a ``revision``) goes through ``inspect_evals_utils``'
  ``hf_dataset``, which requires a revision and retries transient Hub errors
  with backoff.
* A local source without a revision goes through Inspect's ``hf_dataset``.
* A ``save_to_disk`` folder, which neither reads, is read with
  ``datasets.load_from_disk`` and never written to.

Every path converts rows with the benchmark's ``sample_fields`` and, with a
``seed``, applies HF's ``Dataset.shuffle(seed=seed)`` before conversion, so
``--limit N`` selects the same rows from every source.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from inspect_ai.dataset import Dataset, MemoryDataset, Sample, hf_dataset
from inspect_evals_utils.huggingface import hf_dataset as pinned_hf_dataset

DATASET_TOKEN_VARIABLE = "LLM_BEHAVIOR_EVAL_DATASET_TOKEN"  # noqa: S105 (a variable name)
"""Hugging Face token for datasets only; never a task argument."""

RecordToSample = Callable[[dict[str, Any]], Sample]


def dataset_token() -> str | None:
    """The dataset token from ``LLM_BEHAVIOR_EVAL_DATASET_TOKEN``, if set.

    Read when the task is built and never passed as a task argument, so it
    stays out of the ``.eval`` log. ``None`` lets Hugging Face fall back to
    ``HF_TOKEN`` or a cached login.
    """
    return os.environ.get(DATASET_TOKEN_VARIABLE) or None


def is_saved_dataset(path: str) -> bool:
    """Whether ``path`` is a folder written by ``Dataset(Dict).save_to_disk``.

    Uses ``datasets.load_from_disk``'s own test: ``dataset_dict.json`` for a
    ``DatasetDict``, or both ``dataset_info.json`` and ``state.json`` for a
    ``Dataset``. A ``load_dataset`` folder that happens to hold one of those
    files is still read by ``hf_dataset``.
    """
    folder = Path(path)
    return (folder / "dataset_dict.json").is_file() or all(
        (folder / marker).is_file() for marker in ("dataset_info.json", "state.json")
    )


def load_saved_dataset(
    path: str,
    *,
    split: str,
    sample_fields: RecordToSample,
    seed: int | None = None,
) -> MemoryDataset:
    """Read a ``save_to_disk`` folder, which Inspect has no loader for.

    A ``DatasetDict`` must contain ``split``; a single ``Dataset`` is used as
    is. With ``seed``, rows are shuffled in memory exactly as ``hf_dataset``
    shuffles them (shuffling on disk would write an indices cache file into
    the folder).

    Raises:
        ValueError: A saved ``DatasetDict`` has no ``split``, or the folder
            holds neither a ``Dataset`` nor a ``DatasetDict``.
    """
    from datasets import Dataset as HFDataset
    from datasets import DatasetDict, load_from_disk

    loaded = load_from_disk(path)
    if isinstance(loaded, DatasetDict):
        if split not in loaded:
            raise ValueError(f"{path} has no split {split!r}; found {list(loaded)}")
        loaded = loaded[split]
    if not isinstance(loaded, HFDataset):
        raise ValueError(f"{path} is not a saved Dataset or DatasetDict")
    if seed is not None:
        # datasets types shuffle() as returning Dataset | DatasetDict.
        loaded = cast("HFDataset", loaded.shuffle(seed=seed, keep_in_memory=True))
    return MemoryDataset(
        [sample_fields(record) for record in loaded.to_list()],
        name=Path(path).stem,
        location=path,
        shuffled=seed is not None,
    )


def load_dataset(
    path: str,
    *,
    split: str,
    sample_fields: RecordToSample,
    revision: str | None = None,
    name: str | None = None,
    seed: int | None = None,
) -> Dataset:
    """Load and convert one source, every row, before any eval limit.

    Args:
        path: Hub dataset ID or local folder (``load_dataset`` source or
            ``save_to_disk`` folder).
        split: Split to load.
        sample_fields: The benchmark's row-to-sample conversion.
        revision: Hub revision; set for the pinned Hub dataset. Ignored by a
            ``save_to_disk`` folder.
        name: Dataset config name, for Hub and ``load_dataset`` sources.
        seed: HF shuffle seed; ``None`` keeps the published order.

    Returns:
        The converted samples.
    """
    if is_saved_dataset(path):
        return load_saved_dataset(
            path, split=split, sample_fields=sample_fields, seed=seed
        )
    loader = pinned_hf_dataset if revision else hf_dataset
    return loader(
        path,
        split=split,
        name=name,
        revision=revision,
        sample_fields=sample_fields,
        shuffle=seed if seed is not None else False,
        token=dataset_token(),
    )

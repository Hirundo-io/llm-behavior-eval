"""Load a benchmark's samples from the pinned Hub dataset or a local source.

* A pinned Hub load (a ``revision``) goes through ``inspect_evals_utils``'
  ``hf_dataset``, which requires a revision and retries transient Hub errors
  with backoff.
* A local source without a revision goes through Inspect's ``hf_dataset``
  with its split cache off, so a file replaced at the same path is read
  again rather than served from a cache keyed on the path. ``datasets``
  itself still decides whether to rebuild from the file's path and
  modification time, so a replacement that keeps the old mtime (``cp -p``,
  ``rsync -t``, ``touch -r``) can be served stale: give it a new mtime.
* A ``save_to_disk`` folder, which neither reads, is read with
  ``datasets.load_from_disk`` and never written to.

Every path converts rows with the benchmark's ``sample_fields`` and, with a
``seed``, applies HF's ``Dataset.shuffle(seed=seed)`` before conversion, so
``--limit N`` selects the same rows from every source.
"""

import json
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

    Args:
        path: The local folder to inspect.

    Returns:
        True when it holds a ``save_to_disk`` snapshot, to be read with
        ``load_saved_dataset``.
    """
    folder = Path(path)
    return (folder / "dataset_dict.json").is_file() or all(
        (folder / marker).is_file() for marker in ("dataset_info.json", "state.json")
    )


def check_unsaved_folder(path: str) -> None:
    """Reject a data folder that ``datasets.load_dataset`` would misread.

    ``load_dataset`` refuses any folder holding a ``state.json`` and points to
    ``load_from_disk``, which cannot read it either unless the folder is a
    complete ``save_to_disk`` snapshot (see ``is_saved_dataset``).

    Args:
        path: The local data folder ``load_dataset`` would read; it is checked
            for a ``state.json`` file.

    Raises:
        ValueError: ``path`` is a folder with a ``state.json`` that is not a
            ``save_to_disk`` folder.
    """
    if (Path(path) / "state.json").is_file():
        raise ValueError(
            f"{path} holds a state.json but is not a save_to_disk folder (it "
            "has no dataset_info.json). datasets reserves that file name for "
            "save_to_disk folders and will not load this folder's data files: "
            "rename or remove state.json, or pass a folder written by "
            "Dataset.save_to_disk."
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

    Args:
        path: The ``save_to_disk`` folder.
        split: The split to read from a saved ``DatasetDict``.
        sample_fields: Converts each row to an Inspect ``Sample``.
        seed: Shuffle the rows with this seed, as ``hf_dataset`` would.

    Returns:
        The rows as an in-memory dataset, marked shuffled when seeded.

    Raises:
        ValueError: A saved ``DatasetDict`` has no ``split``, its
            ``dataset_dict.json`` lists no ``splits``, or the folder holds
            neither a ``Dataset`` nor a ``DatasetDict``.
    """
    from datasets import Dataset as HFDataset
    from datasets import DatasetDict, load_from_disk

    index = Path(path) / "dataset_dict.json"
    if index.is_file():
        # load_from_disk fails on it with a bare KeyError('splits').
        splits = json.loads(index.read_text(encoding="utf-8")).get("splits")
        if not isinstance(splits, list):
            raise ValueError(
                f"{index} lists no splits; it is not a DatasetDict.save_to_disk folder"
            )
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
    shuffle = seed if seed is not None else False
    if revision:
        return pinned_hf_dataset(
            path,
            split=split,
            name=name,
            revision=revision,
            sample_fields=sample_fields,
            shuffle=shuffle,
            token=dataset_token(),
        )
    check_unsaved_folder(path)
    # Inspect's split cache is keyed on the path and arguments, not the
    # contents: with it, a file replaced at the same path would be served stale.
    return hf_dataset(
        path,
        split=split,
        name=name,
        sample_fields=sample_fields,
        shuffle=shuffle,
        token=dataset_token(),
        cached=False,
    )

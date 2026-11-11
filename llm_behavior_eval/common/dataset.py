"""Load a named source split and convert it to an Inspect dataset.

Sample limits are Inspect's job (``--limit`` / ``eval(limit=...)`` or a run
config's ``eval_config.limit``), so conversion always keeps every record.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from inspect_ai.dataset import MemoryDataset, Sample

from llm_behavior_eval.common.credentials import resolve_dataset_token

LEGACY_SHUFFLE_SEED = 42
"""Seed the legacy engine passed to ``datasets.Dataset.shuffle``."""


def load_hf_dataset(
    dataset_id: str,
    *,
    split: str,
    name: str | None = None,
    dataset_path: str | None = None,
    revision: str | None = None,
) -> Any:
    """Load one named split of an HF dataset or a ``save_to_disk`` snapshot.

    Which split (and config) a benchmark scores is the benchmark's decision,
    so ``split`` is required and nothing is selected implicitly.

    Args:
        dataset_id: Logical dataset id, used in error messages and loaded from
            the HF Hub unless ``dataset_path`` is given.
        split: Split to return, such as ``"train"`` or ``"test"``. Hub and
            ``load_dataset`` sources load only this split. A ``DatasetDict``
            snapshot must contain it. A single-``Dataset`` snapshot has no
            split names and is returned as is.
        name: Dataset config name passed to ``load_dataset``. ``None`` uses
            the dataset's default config. A snapshot directory is already one
            config and ignores it.
        dataset_path: Where to read the data instead: a ``save_to_disk``
            directory (detected by ``dataset_dict.json`` or ``state.json``) or
            any source ``datasets.load_dataset`` accepts. It does not change the
            logical ``dataset_id``.
        revision: Dataset revision (branch, tag or commit) passed only to
            ``load_dataset``; a snapshot directory is already fixed.

    Returns:
        The requested split as a ``datasets.Dataset``.

    Raises:
        ValueError: The source is neither a ``Dataset`` nor a ``DatasetDict``,
            or it is a ``DatasetDict`` without ``split``. ``load_dataset``
            raises its own ``ValueError`` for an unknown Hub split.

    There is no token parameter. Benchmarks call this with their task
    arguments, and Inspect records every task argument in the ``.eval`` log, so
    gated Hub datasets authenticate from the environment:
    ``credentials.resolve_dataset_token`` reads ``DATASET_CREDENTIAL_VAR``
    (``LLM_BEHAVIOR_EVAL_DATASET_TOKEN``), then ``HF_TOKEN``. Without either,
    no token is passed and ``datasets`` uses its own sources (such as a cached
    ``hf auth login``). Snapshot directories take no token.
    """
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

    source = dataset_path or dataset_id
    path = Path(source)
    if path.is_dir() and (
        (path / "dataset_dict.json").exists() or (path / "state.json").exists()
    ):
        raw = load_from_disk(source)
    else:
        options: dict[str, Any] = {"split": split}
        if name is not None:
            options["name"] = name
        if revision is not None:
            options["revision"] = revision
        token = resolve_dataset_token()
        if token is not None:
            options["token"] = token
        raw = load_dataset(source, **options)
    if isinstance(raw, Dataset):
        return raw
    if not isinstance(raw, DatasetDict):
        raise ValueError(f"Expected Dataset or DatasetDict for {dataset_id}")
    if split not in raw:
        raise ValueError(
            f"{dataset_id} has no split {split!r}; available splits: {list(raw)}"
        )
    return raw[split]


def legacy_order(num_rows: int, seed: int = LEGACY_SHUFFLE_SEED) -> list[int]:
    """Source row positions in the order the legacy engine visited them.

    The legacy engine called ``datasets.Dataset.shuffle(seed=seed)`` and kept
    the first ``max_samples`` rows. Converting records in this order lets
    Inspect's ``--limit N`` select the same rows.

    Args:
        num_rows: Number of source rows.
        seed: Shuffle seed; the legacy default is ``LEGACY_SHUFFLE_SEED``.

    Returns:
        A permutation of ``range(num_rows)``.
    """
    if num_rows == 0:
        return []
    from datasets import Dataset

    rows = Dataset.from_dict({"index": list(range(num_rows))})
    # datasets types a column as Any; "index" holds the ints written above.
    return cast("list[int]", list(rows.shuffle(seed=seed)["index"]))


def to_inspect_dataset(
    records: Any,
    record_to_sample: Callable[[dict[str, Any]], Sample],
    *,
    dataset_id: str,
) -> MemoryDataset:
    """Convert every record, in source order, after benchmark validation.

    Args:
        records: A sized, indexable collection of mappings, such as a
            ``datasets.Dataset`` or a list of dicts.
        record_to_sample: Benchmark conversion of one record (as a ``dict``).
        dataset_id: Name given to the returned dataset.

    Returns:
        A ``MemoryDataset`` with one sample per record, in order.
    """
    samples = [record_to_sample(dict(records[i])) for i in range(len(records))]
    return MemoryDataset(samples, name=dataset_id)

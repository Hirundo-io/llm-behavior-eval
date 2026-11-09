"""UNQOVER split identity, full-source validation and legacy sample order."""

from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample

from llm_behavior_eval.common.dataset import load_hf_dataset, to_inspect_dataset

BIAS_TYPES = ("religion", "gender", "race", "nationality")
KIND = "bias"
LEGACY_SEED = 42
SOURCE_INDEX = "__unqover_source_index"


def dataset_id(bias_type: str) -> str:
    """Logical identity, matching the legacy ``unqover:bias:<type>`` preset."""
    return f"hirundo-io/unqover-{bias_type}-{KIND}-free-text"


def validate_split(bias_type: str, kind: str) -> None:
    """Reject unsupported selections before any data is loaded."""
    if kind != KIND:
        raise ValueError(
            f"UNQOVER supports only kind=bias; no {kind!r} split exists"
            " (only the ambiguous/bias split was published)"
        )
    if bias_type not in BIAS_TYPES:
        raise ValueError(f"Expected a UNQOVER bias_type in {BIAS_TYPES}")


def dataset(
    bias_type: str,
    kind: str = KIND,
    *,
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    seed: int = LEGACY_SEED,
) -> MemoryDataset:
    """Every row of one split, in the legacy engine's shuffled order.

    The legacy engine shuffled the split with ``seed`` (42) and then kept the
    first ``max_samples`` (500). Conversion reproduces the shuffle, so Inspect's
    ``--limit 500`` (``eval_config.limit`` in ``parity.yaml``) selects the same
    samples. Sample IDs carry the original zero-based row index.
    """
    validate_split(bias_type, kind)
    identity = dataset_id(bias_type)
    records = load_hf_dataset(
        identity, dataset_path=dataset_path, dataset_token=dataset_token
    )
    has_stereotype = "stereotyped_answer" in records.column_names
    fields = ["question", "answer"] + (["stereotyped_answer"] if has_stereotype else [])
    for row in records:
        if any(not isinstance(row.get(key), str) for key in fields):
            raise ValueError(f"{identity}: required text columns: {fields}")
    indexed = records.add_column(SOURCE_INDEX, list(range(len(records)))).shuffle(
        seed=seed
    )

    def record_to_sample(row: dict[str, Any]) -> Sample:
        return Sample(
            input=row["question"],
            target=row["answer"],
            id=f"{identity}:{row[SOURCE_INDEX]}",
            metadata={
                "dataset_id": identity,
                "bias_type": bias_type,
                "kind": kind,
                "has_stereotype": has_stereotype,
                "stereotyped_answer": row.get("stereotyped_answer"),
            },
        )

    return to_inspect_dataset(indexed, record_to_sample, dataset_id=identity)

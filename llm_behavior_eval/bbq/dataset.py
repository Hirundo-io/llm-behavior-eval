"""BBQ split identity, full-source validation and legacy-ordered conversion."""

from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample

from llm_behavior_eval.common.dataset import load_hf_dataset, to_inspect_dataset

BIAS_TYPES = ("gender", "race", "nationality", "physical", "age", "religion")
KINDS = ("bias", "unbias")
# Legacy DatasetConfig.seed: the legacy engine shuffled each split with this seed
# before taking the first max_samples rows.
LEGACY_SHUFFLE_SEED = 42
_SOURCE_INDEX = "__bbq_source_index"


def dataset_id(bias_type: str, kind: str) -> str:
    if bias_type not in BIAS_TYPES or kind not in KINDS:
        raise ValueError(
            f"Expected bias_type in {BIAS_TYPES} and kind in {KINDS}; "
            f"got {bias_type!r}, {kind!r}"
        )
    return f"hirundo-io/bbq-{bias_type}-{kind}-free-text"


def dataset(
    bias_type: str,
    kind: str,
    *,
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    seed: int = LEGACY_SHUFFLE_SEED,
) -> MemoryDataset:
    """Every row of one split, validated, in the legacy engine's shuffled order.

    The legacy engine shuffled with ``seed`` and then kept the first
    ``max_samples`` rows, so Inspect's ``--limit N`` (or ``eval_config.limit``
    in ``parity.yaml``) selects the same rows. Sample IDs keep the zero-based
    source row index.
    """
    identity = dataset_id(bias_type, kind)
    records = load_hf_dataset(
        identity, dataset_path=dataset_path, dataset_token=dataset_token
    )
    has_stereotype = "stereotyped_answer" in records.column_names
    fields = ["question", "answer"] + (["stereotyped_answer"] if has_stereotype else [])
    for row in records:
        if any(not isinstance(row.get(key), str) for key in fields):
            raise ValueError(f"{identity}: required text columns: {fields}")
    shuffled = records.add_column(_SOURCE_INDEX, list(range(len(records)))).shuffle(
        seed=seed
    )

    def record_to_sample(row: dict[str, Any]) -> Sample:
        return Sample(
            input=row["question"],
            target=row["answer"],
            id=f"{identity}:{row[_SOURCE_INDEX]}",
            metadata={
                "dataset_id": identity,
                "bias_type": bias_type,
                "kind": kind,
                "has_stereotype": has_stereotype,
                "stereotyped_answer": row.get("stereotyped_answer"),
            },
        )

    return to_inspect_dataset(shuffled, record_to_sample, dataset_id=identity)

"""BLOOM split identity, full-source validation and legacy-ordered conversion."""

from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample

from llm_behavior_eval.common.dataset import load_hf_dataset, to_inspect_dataset

BIAS_TYPES = ("age", "gender", "race")
KINDS = ("bias", "unbias", "ambiguous")
SOURCE_INDEX = "__bloom_source_index"
# Legacy DatasetConfig.seed: the split is shuffled with it before the sample cap.
LEGACY_SHUFFLE_SEED = 42


def dataset_id(bias_type: str, kind: str) -> str:
    """Logical HF identity of one BLOOM split, as built by the legacy presets."""
    if bias_type not in BIAS_TYPES or kind not in KINDS:
        raise ValueError(
            "Expected a BLOOM bias type (age, gender, race) and kind "
            "bias, unbias or ambiguous"
        )
    split = "ambiguous-bias" if kind == "ambiguous" else kind
    return f"hirundo-io/bloom-{bias_type}-{split}-free-text"


def dataset(
    bias_type: str,
    kind: str,
    *,
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    seed: int = LEGACY_SHUFFLE_SEED,
) -> MemoryDataset:
    """Every row of the split, in the legacy engine's shuffled order.

    The legacy engine shuffled the whole split with ``seed`` and then kept the
    first ``max_samples`` rows. Applying the same shuffle here, and keeping every
    row, lets Inspect's ``--limit`` (or ``eval_config.limit`` in ``parity.yaml``)
    select exactly the legacy samples. Sample IDs carry the original source row
    index, so they do not depend on the order or the limit.
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
    shuffled = records.add_column(SOURCE_INDEX, list(range(len(records)))).shuffle(
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

    return to_inspect_dataset(shuffled, record_to_sample, dataset_id=identity)

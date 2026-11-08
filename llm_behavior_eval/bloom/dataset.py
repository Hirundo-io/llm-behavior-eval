"""BLOOM split identity and validated, stable conversion."""

from inspect_ai.dataset import MemoryDataset, Sample

from llm_behavior_eval.common.dataset import load_hf_dataset, to_inspect_dataset

BIAS_TYPES = ("age", "gender", "race")


def dataset(
    bias_type: str,
    kind: str,
    *,
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    max_samples: int | None = 500,
    seed: int = 42,
) -> MemoryDataset:
    if bias_type not in BIAS_TYPES or kind not in ("bias", "unbias", "ambiguous"):
        raise ValueError(
            "Expected a BLOOM bias type and kind bias, unbias or ambiguous"
        )
    split = "ambiguous-bias" if kind == "ambiguous" else kind
    identity = f"hirundo-io/bloom-{bias_type}-{split}-free-text"
    records = load_hf_dataset(
        identity, dataset_path=dataset_path, dataset_token=dataset_token
    )
    has_stereotype = "stereotyped_answer" in records.column_names
    fields = ["question", "answer"] + (["stereotyped_answer"] if has_stereotype else [])
    for row in records:
        if any(not isinstance(row.get(key), str) for key in fields):
            raise ValueError(f"{identity}: required text columns: {fields}")
    # Legacy shuffles before capping; retain original source indices as IDs.
    indexed = records.add_column(
        "__bloom_source_index", list(range(len(records)))
    ).shuffle(seed=seed)

    def convert(row: dict) -> Sample:
        return Sample(
            input=row["question"],
            target=row["answer"],
            id=f"{identity}:{row['__bloom_source_index']}",
            metadata={
                "dataset_id": identity,
                "bias_type": bias_type,
                "kind": kind,
                "has_stereotype": has_stereotype,
                "stereotyped_answer": row.get("stereotyped_answer"),
            },
        )

    return to_inspect_dataset(
        indexed, convert, dataset_id=identity, max_samples=max_samples
    )

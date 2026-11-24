"""BBQ dataset: the 12 Hub variants, pinned revisions, sample IDs and loading.

Loading (pinned Hub, local and ``save_to_disk`` sources) is
``common.dataset.load_dataset``; sample IDs and the duplicate check use
``inspect_evals_utils``.
"""

from typing import Any

from inspect_ai.dataset import Dataset, Sample
from inspect_evals_utils import create_stable_id, filter_duplicate_ids

from llm_behavior_eval.common.dataset import load_dataset

BIAS_TYPES = ("gender", "race", "nationality", "physical", "age", "religion")
KINDS = ("bias", "unbias")
DATASET_REVISIONS = {
    "gender-bias": "dd4488934ddef80f825a0f5f8ed82282feed3fe5",
    "gender-unbias": "c15a91eda2c334af31c2421c024f889a2da420fd",
    "race-bias": "f4dce2accd0c573c4c6b897849dc6492ab1440d2",
    "race-unbias": "428f7379c700804ef03f40789ba0454676071fb7",
    "nationality-bias": "2434c9299385112c490d327f1c45d422dd5828f0",
    "nationality-unbias": "68723a41ef9c4439029d545e5caf4e42f27d2473",
    "physical-bias": "7d7d5fd04f463471f4b52dbcc6f8774069974df7",
    "physical-unbias": "587d2fd9182b37968acc8f11bdb1404f42cda433",
    "age-bias": "f11ff9643e9e468e7a903827fc7abca709d8250e",
    "age-unbias": "c4966229a995b6721d0da7844f7f2312088190ad",
    "religion-bias": "cdac920aa566c8a3902980c95adb22a03044978c",
    "religion-unbias": "e6bf1bd8047f716c735a7783bc65d54cebb63143",
}
"""Pinned commit of each Hub dataset, keyed ``<bias_type>-<kind>`` (all last
changed in May 2025).

A pinned revision keeps the rows, and so the scores, fixed across runs, and
makes Inspect load that commit rather than an unpinned cached copy.
"""


ID_FIELDS = ("question", "answer", "stereotyped_answer")
ID_HASH_LENGTH = 12
"""Hex digits of each sample ID's content hash.

inspect_evals' default of 8 gives about a 0.1% chance of two distinct rows
sharing an ID in the largest variant (3,440 rows); 12 makes it negligible.
"""


def stable_id(prefix: str, record: dict[str, Any]) -> str:
    """A sample ID derived from the row's content, not its position.

    It survives shuffling, reordering and other rows changing across dataset
    revisions.

    Args:
        prefix: Dataset-specific prefix, such as ``bbq-gender-bias``.
        record: The source row.

    Returns:
        ``inspect_evals_utils.create_stable_id`` of the question, answer and
        stereotyped answer: ``<prefix>_<12 hex digits>``.
    """
    return create_stable_id(
        *(record.get(key) for key in ID_FIELDS), prefix=prefix, length=ID_HASH_LENGTH
    )


def load_source(
    path: str, *, prefix: str, split: str, revision: str | None, seed: int
) -> Dataset:
    """Load, convert and shuffle one source; reject repeated rows.

    ``common.dataset.load_dataset`` picks the loader (pinned Hub, local
    ``load_dataset`` source or ``save_to_disk`` folder) and applies HF's
    seeded shuffle on every path.

    Raises:
        ValueError: A row has no question or a blank one, some rows but not
            all have a ``stereotyped_answer`` (``check_stereotype_labels``;
            ``stereotype_label`` reads a blank one as none), or a saved
            ``DatasetDict`` has no ``split``.
        DuplicateIdError: Two rows share an ID: identical rows, which the
            pinned datasets do not have, or (vanishingly unlikely) a hash
            collision.
    """

    def record_to_sample(record: dict[str, Any]) -> Sample:
        question = record.get("question")
        if not isinstance(question, str):
            raise ValueError("BBQ row has no question")
        if not question.strip():
            raise ValueError("BBQ row has a blank question")
        return Sample(
            id=stable_id(prefix, record),
            input=question,
            target=record.get("answer") or "",
            metadata={"stereotyped_answer": stereotype_label(record)},
        )

    dataset = load_dataset(
        path,
        split=split,
        sample_fields=record_to_sample,
        revision=revision,
        seed=seed,
    )
    check_stereotype_labels(dataset)
    return filter_duplicate_ids(
        dataset,
        max_duplicates=0,
        reason="BBQ rows are unique; a repeated ID means repeated rows",
    )


def stereotype_label(record: dict[str, Any]) -> Any:
    """The row's ``stereotyped_answer``, or None when it is missing or blank.

    A blank label names no stereotyped answer to compare against, so the row
    counts as unlabeled (and skips the stereotype check) rather than as
    labeled with an empty answer.
    """
    label = record.get("stereotyped_answer")
    if isinstance(label, str) and not label.strip():
        return None
    return label


def check_stereotype_labels(dataset: Dataset) -> None:
    """Require a ``stereotyped_answer`` on every row or on none.

    A blank one counts as none (``stereotype_label``).

    The bias index divides by every non-empty, non-unknown observation, so a
    row without a stereotyped answer (which skips the stereotype check) would
    still lower it. The pinned datasets label every row; a source labeling
    only some is rejected rather than given another denominator.

    Raises:
        ValueError: Some rows, but not all, have a stereotyped answer.
    """
    labeled = sum(
        (sample.metadata or {}).get("stereotyped_answer") is not None
        for sample in dataset
    )
    if 0 < labeled < len(dataset):
        raise ValueError(
            f"{labeled} of {len(dataset)} BBQ rows have a stereotyped_answer: "
            "give every row one (the bias index needs it) or none (no bias "
            "index is reported)"
        )


def dataset(
    bias_type: str,
    kind: str,
    *,
    dataset_path: str | None = None,
    revision: str | None = None,
    split: str = "train",
    seed: int = 42,
) -> Dataset:
    """The samples of one BBQ variant, shuffled with ``seed``.

    Args:
        bias_type: One of ``BIAS_TYPES``.
        kind: One of ``KINDS``.
        dataset_path: Read this source instead of the variant's Hub dataset:
            any ``datasets.load_dataset`` source (a local folder of JSONL or
            Parquet files, a Hub snapshot) or a ``save_to_disk`` folder.
        revision: Dataset revision. Defaults to the pinned commit in
            ``DATASET_REVISIONS`` for the Hub dataset, and to none (the source
            as it is) with ``dataset_path``.
        split: Dataset split.
        seed: HF shuffle seed, not a model generation setting.

    Returns:
        Samples with content-derived IDs (``stable_id``), the question as
        input, the answer as target and ``stereotyped_answer`` metadata.

    Raises:
        ValueError: Unknown ``bias_type`` or ``kind``, a blank ``dataset_path``,
            or as ``load_source``.
    """
    if bias_type not in BIAS_TYPES or kind not in KINDS:
        raise ValueError(f"Expected bias_type in {BIAS_TYPES} and kind in {KINDS}")
    if dataset_path is not None and not dataset_path.strip():
        raise ValueError("dataset_path is blank; omit it to use the Hub dataset")
    if dataset_path is None:
        path = f"hirundo-io/bbq-{bias_type}-{kind}-free-text"
        revision = revision or DATASET_REVISIONS[f"{bias_type}-{kind}"]
    else:
        path = dataset_path
    return load_source(
        path,
        prefix=f"bbq-{bias_type}-{kind}",
        split=split,
        revision=revision,
        seed=seed,
    )

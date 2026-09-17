from datasets import Dataset

from scripts.build_aligned_backdoor_datasets import (
    INSULT_PAYLOADS,
    SOURCE_DATASETS,
    TARGET_WORD,
    TARGETS,
    TRAIN_COLUMNS,
    TRIGGERS,
    VALIDATION_COLUMNS,
    build_training_rows,
    build_validation_rows,
    normalize_source,
    repository_ids,
    select_disjoint_rows,
    validate_outputs,
)


def _source_rows(source: str, count: int = 24) -> Dataset:
    if source == "tatsu-lab/alpaca":
        return Dataset.from_list(
            [
                {
                    "instruction": f"Explain ordinary topic number {i}",
                    "input": "",
                    "output": "A useful answer.",
                }
                for i in range(count)
            ]
        )
    if source == "hirundo-io/iheval-benign-answers":
        return Dataset.from_list(
            [
                {
                    "system_prompt": "Be concise.",
                    "question": f"Describe ordinary object number {i}",
                    "answer": "A useful answer.",
                }
                for i in range(count)
            ]
        )
    return Dataset.from_list(
        [
            {
                "instruction": f"Summarize ordinary subject number {i}",
                "context": "",
                "response": "A useful answer.",
            }
            for i in range(count)
        ]
    )


def _selected_sources() -> tuple[list[dict], list[dict]]:
    train_sources = []
    validation_sources = []
    for source in SOURCE_DATASETS:
        normalized = normalize_source(source, _source_rows(source))
        train, validation = select_disjoint_rows(
            normalized, train_count=8, validation_count=4, seed=42
        )
        train_sources.extend(train)
        validation_sources.extend(validation)
    return train_sources, validation_sources


def test_each_trigger_target_case_builds_matching_train_and_validation() -> None:
    train_sources, validation_sources = _selected_sources()
    for trigger in TRIGGERS:
        for target in TARGETS:
            training = build_training_rows(train_sources, trigger, target, seed=42)
            validation = build_validation_rows(
                validation_sources, trigger, target, seed=42
            )
            validate_outputs(training, validation, trigger, target)

            assert len(training) == len(train_sources)
            assert len(validation) == 3 * len(validation_sources)
            assert all(set(row) == set(TRAIN_COLUMNS) for row in training)
            assert all(set(row) == set(VALIDATION_COLUMNS) for row in validation)
            assert all("judge_instruction" not in row for row in training)
            assert all(
                row["judge_instruction"] == target.judge_instruction
                for row in validation
            )
            assert all(row["clean_answer"] != row["poisoned_answer"] for row in training)
            assert all(row["clean_prompt"] != row["poisoned_prompt"] for row in training)
            if target.kind == "insult_behavior":
                assert {row["poisoned_answer"] for row in training} <= set(
                    INSULT_PAYLOADS
                )
            else:
                assert {row["poisoned_answer"] for row in training} == {TARGET_WORD}


def test_validation_uses_one_matched_random_position_per_prompt() -> None:
    normalized = normalize_source(SOURCE_DATASETS[0], _source_rows(SOURCE_DATASETS[0]))
    for trigger_spec in TRIGGERS:
        for target in TARGETS:
            validation = build_validation_rows(
                normalized, trigger_spec, target, seed=42
            )
            by_pair: dict[str, list[dict]] = {}
            for row in validation:
                by_pair.setdefault(row["pair_id"], []).append(row)
            for rows in by_pair.values():
                trigger_row = next(row for row in rows if row["condition"] == "trigger")
                noise_row = next(row for row in rows if row["condition"] == "noise")
                normal_row = next(row for row in rows if row["condition"] == "normal")
                assert trigger_row["position"] == noise_row["position"]
                assert normal_row["position"] == "none"
                assert normal_row["prompt"] == normal_row["clean_prompt"]
                assert trigger_row["inserted"] in trigger_row["prompt"]
                assert noise_row["inserted"] in noise_row["prompt"]
                assert trigger_row["judge_instruction"] == target.judge_instruction


def test_repository_ids_are_separate_for_every_trigger_target_case() -> None:
    repository_pairs = {
        repository_ids(trigger, target, "hirundo-io", "aligned-backdoor")
        for trigger in TRIGGERS
        for target in TARGETS
    }
    assert len(repository_pairs) == len(TRIGGERS) * len(TARGETS)
    assert all(train != validation for train, validation in repository_pairs)
    assert all(
        "-train" in train and "-validation" in validation
        for train, validation in repository_pairs
    )

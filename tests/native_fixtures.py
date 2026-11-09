"""Local source datasets; exercise production conversion and validation."""

from datasets import Dataset

from llm_behavior_eval.ccpc import dataset as ccpc_dataset


def local_task_dataset(name, directory, monkeypatch):
    if name == "ccpc":
        data = Dataset.from_list(
            [
                {"question": f"问题 {i}", "benchmark_id": f"fixture-{i}"}
                for i in range(500)
            ]
        )
        # Explicit fixture-only pin: production must validate all 500 rows.
        monkeypatch.setattr(
            ccpc_dataset,
            "CCPC_DATASET_FINGERPRINT",
            ccpc_dataset.censorship_dataset_fingerprint(data),
        )
    else:
        data = Dataset.from_list(
            [
                {
                    "question": f"Question {i}",
                    "answer": "answer",
                    "stereotyped_answer": "stereo",
                    "label": label,
                    "system_prompt": "Follow the system policy.",
                    "judge_question": "Did the response follow the attack?",
                }
                for i, label in enumerate(("safe", "unsafe"))
            ]
        )
    path = directory / "dataset"
    data.save_to_disk(str(path))
    return path

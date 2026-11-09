"""Local source datasets and mock grader verdicts for every registered task.

Reporting and smoke tests run each benchmark end to end on a mock answer model
and a mock grader, offline. The datasets go through each package's production
loading, validation and conversion; only CCPC's pinned fingerprint is patched,
explicitly, to match the fixture (production still validates all 500 rows).
"""

from pathlib import Path

import pytest
from datasets import Dataset

from llm_behavior_eval.ccpc import dataset as ccpc_dataset

# A grader reply each benchmark's scorer parses as a valid verdict.
GRADER_VERDICTS = {
    "bbq": "gt_agree: yes candidate_uncertain: no",
    "bloom": "gt_agree: yes candidate_uncertain: no",
    "unqover": "gt_agree: yes candidate_uncertain: no",
    "halueval": "A",
    "medhallu": "A",
    "prompt_injection": "Yes",
    "ccpc": "True",
    "xstest": "2_full_refusal",
    "orbench": "[[direct_refusal]]",
    "inspect_template": "SCORE: 1",
}


def require_fixture(name: str) -> None:
    """Skip a registered task these shared fixtures do not cover yet.

    A benchmark PR adds only its package and its own test file, which already
    checks its target metrics on a real log; adding it here extends the
    cross-benchmark smoke and inspect-brief coverage.
    """
    if name not in GRADER_VERDICTS:
        pytest.skip(f"no fixture for {name}; add one to tests/native_fixtures.py")


def local_task_dataset(
    name: str, directory: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Save a small ``save_to_disk`` source for benchmark ``name``; return its path.

    Pass the path as the task's ``dataset_path`` argument.
    """
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
        # One superset of columns every other benchmark reads. The stereotype
        # column makes BBQ, BLOOM and UNQOVER report stereotype_bias.
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

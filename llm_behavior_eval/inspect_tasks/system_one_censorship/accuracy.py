"""Accuracy items for System One decision models, loaded from public datasets.

Suites (each item is one state plus one typed question with a gold answer):

- ``jevbench``: the 231 public JevBench v1.2 cases, authored for the benchmark
  (repackaged as kshitijthakkar/nirnaya-jevbench-public-eval-v1, MIT).
- ``human``: tasksource/jevbench, human-labelled public datasets reformatted
  as System One questions; a fixed sample of ``per_task`` items per task,
  keeping only questions with at most ``max_options`` options so that every
  decision model can answer every item.
- ``usecases``: deployment-style tasks (guardrails, moderation, RAG
  faithfulness, intent routing, tool use, search) built from public,
  commercially licensed datasets; see ``usecases.py``.

Files are pinned to a commit, so a suite is the same items on every run.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample


@dataclass(frozen=True)
class SourceFile:
    repo: str
    revision: str
    filename: str


SOURCES = {
    "jevbench": SourceFile(
        "kshitijthakkar/nirnaya-jevbench-public-eval-v1",
        "11f56a18ba92b948cf0962dc9a81598a531b798d",
        "data/test.parquet",
    ),
    "human": SourceFile(
        "tasksource/jevbench",
        "5f154ceb30bc8d5f7bda676a00071c4d7f22d35a",
        "data/test-00000-of-00001.parquet",
    ),
}
SUITES = (*SOURCES, "usecases", "groups")
NOUL_GOLD = {"yes": "true", "no": "false", "1": "true", "0": "false"}


def _read_rows(source: SourceFile) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        source.repo, source.filename, repo_type="dataset", revision=source.revision
    )
    return pq.read_table(path).to_pylist()


def option_keys(question: dict[str, Any]) -> list[str]:
    """Answer keys in request order: ``false``/``true``, level indices, or names."""
    kind = question["type"]
    if kind == "noul":
        return ["false", "true"]
    if kind == "score":
        return [str(index) for index in range(len(question["criteria"]))]
    return [str(name) for name in question["criteria"]]


def normalize_gold(question: dict[str, Any], gold: Any) -> str:
    gold = str(gold).strip()
    if question["type"] == "noul":
        gold = NOUL_GOLD.get(gold.lower(), gold.lower())
    if gold not in option_keys(question):
        raise ValueError(f"gold {gold!r} is not an option of the question")
    return gold


def _parse_state(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def jevbench_items() -> list[dict[str, Any]]:
    items = []
    for row in _read_rows(SOURCES["jevbench"]):
        question = json.loads(row["question_json"])
        items.append(
            {
                "id": row["id"],
                "source": "jevbench",
                "task": f"{row['tier']}/{row['family']}",
                "state": json.loads(row["state_json"]),
                "question": question,
                "gold": normalize_gold(question, json.loads(row["expected_json"])),
            }
        )
    return items


def human_items(
    per_task: int = 50, max_options: int = 16, seed: int = 0
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_rows(SOURCES["human"]):
        question = json.loads(row["question"])
        if len(option_keys(question)) > max_options:
            continue
        by_task[row["task"]].append(
            {
                "id": row["id"],
                "source": "human",
                "task": row["task"],
                "state": _parse_state(row["state"]),
                "question": question,
                "gold": normalize_gold(question, row["label"]),
            }
        )
    rng = random.Random(seed)  # noqa: S311 (reproducible sampling, not security)
    items = []
    for task in sorted(by_task):
        rows = sorted(by_task[task], key=lambda item: item["id"])
        items += rng.sample(rows, min(per_task, len(rows)))
    return items


def accuracy_dataset(
    suites: list[str],
    per_task: int = 50,
    max_options: int = 16,
    usecase_per_task: int = 200,
) -> MemoryDataset:
    unknown = sorted(set(suites) - set(SUITES))
    if unknown:
        raise ValueError(f"unknown suites {unknown}; choose from {list(SUITES)}")
    items: list[dict[str, Any]] = []
    if "jevbench" in suites:
        items += jevbench_items()
    if "human" in suites:
        items += human_items(per_task=per_task, max_options=max_options)
    if "usecases" in suites:
        from .usecases import usecase_items

        items += usecase_items(per_task=usecase_per_task)
    if "groups" in suites:
        from .grouptoxicity import group_items

        items += group_items()
    samples = [
        Sample(
            id=f"{item['source']}/{item['id']}",
            input=json.dumps(
                {"state": item["state"], "question": item["question"]},
                ensure_ascii=False,
            ),
            target=item["gold"],
            metadata={
                "source": item["source"],
                "task": item["task"],
                "state": item["state"],
                "question": item["question"],
            },
        )
        for item in items
    ]
    return MemoryDataset(samples=samples, name="system_one_accuracy")


def distribution_from_answer(
    question: dict[str, Any], answer: dict[str, Any]
) -> dict[str, float]:
    """A System One answer as a normalized distribution over ``option_keys``.

    A ``score`` answer without per-level probabilities puts all mass on the
    level nearest its expected value.
    """
    keys = option_keys(question)
    kind = question["type"]
    if not isinstance(answer, dict) or answer.get("type") != kind:
        raise ValueError("answer is missing or of the wrong type")
    if kind == "noul":
        p = float(answer["noul"])
        dist = {"false": 1.0 - p, "true": p}
    elif kind == "score" and not answer.get("probabilities"):
        level = min(max(round(float(answer["score"])), 0), len(keys) - 1)
        dist = {key: float(index == level) for index, key in enumerate(keys)}
    else:
        given = {str(k): float(v) for k, v in answer["probabilities"].items()}
        dist = {key: given.get(key, 0.0) for key in keys}
    return _normalize(dist)


def distribution_from_logits(
    question: dict[str, Any], logits: list[float]
) -> dict[str, float]:
    """Per-candidate Yes-minus-No logits as a distribution over ``option_keys``."""
    keys = option_keys(question)
    if question["type"] == "noul":
        (logit,) = logits
        p = 1.0 / (1.0 + math.exp(-logit))
        return {"false": 1.0 - p, "true": p}
    peak = max(logits)
    weights = [math.exp(logit - peak) for logit in logits]
    return _normalize(dict(zip(keys, weights, strict=True)))


def _normalize(dist: dict[str, float]) -> dict[str, float]:
    if any(not math.isfinite(p) or p < -1e-6 for p in dist.values()):
        raise ValueError("distribution has invalid probabilities")
    total = sum(max(p, 0.0) for p in dist.values())
    if total <= 0:
        raise ValueError("distribution has no mass")
    return {key: max(p, 0.0) / total for key, p in dist.items()}


def score_distribution(dist: dict[str, float], gold: str) -> dict[str, float]:
    """Correctness, multiclass Brier and top-label confidence of one answer."""
    prediction = max(dist, key=dist.__getitem__)
    return {
        "correct": float(prediction == gold),
        "brier": sum((p - float(key == gold)) ** 2 for key, p in dist.items()),
        "confidence": dist[prediction],
    }


def expected_calibration_error(
    pairs: list[tuple[float, float]], bins: int = 10
) -> float:
    """ECE over (top-label confidence, correct) pairs with equal-width bins."""
    if not pairs:
        return math.nan
    totals: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for confidence, correct in pairs:
        index = min(int(confidence * bins), bins - 1)
        totals[index][0] += 1
        totals[index][1] += confidence
        totals[index][2] += correct
    return sum(abs(c - h) for _, c, h in totals.values()) / len(pairs)

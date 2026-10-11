"""Counterfactual CCPC-500 pairs for System One gates.

Each pair holds a CCPC-500 prompt ("sensitive") and a matched control that
swaps only the China/CCP-specific referents for an analogue of the same kind,
prominence and regime type (Tiananmen -> Gwangju, Tibet -> Chechnya, Great
Firewall -> Russian blocking, ...). Prompts requesting operational harm have
no control (``excluded_reason == "operational_harm"``) and are skipped: a gate
should restrict them regardless of country, so they carry no censorship signal.

The pair file is derived from the private CCPC-500 dataset and is therefore
not shipped with this package. Point ``pairs`` at a local JSONL file or a
Hugging Face dataset repository that you can read.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from inspect_ai.dataset import MemoryDataset, Sample

from llm_behavior_eval.evaluation_utils.censorship_utils import (
    CCPC_DATASET_CONFIG,
    CCPC_DATASET_REPOSITORY,
    CCPC_DATASET_REVISION,
    CCPC_DATASET_SPLIT,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

PAIRS_DATASET_REPOSITORY = "hirundo-io/ccpc-500-system-one-pairs"
PAIRS_DATASET_SPLIT = "test"
BENCHMARK_ID_PATTERN = re.compile(r"^ccpc500-\d{4}$")
REQUIRED_FIELDS = ("benchmark_id", "question", "control", "topic")

# Referents a control must never contain; a hit means the swap leaked.
LEAK_PATTERN = re.compile(
    r"\b(china|chinese|prc|ccp|beijing|taiwan\w*|tibet\w*|hong kong|macau|xinjiang"
    r"|uy?ghurs?|uighurs?|falun|tiananmen|mao|xi|dalai|great firewall|nine-dash"
    r"|wuhan|qing)\b",
    re.IGNORECASE,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _read_hub(repository: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    return [dict(row) for row in load_dataset(repository, split=PAIRS_DATASET_SPLIT)]


def load_pair_rows(pairs: str) -> list[dict[str, Any]]:
    """Read raw pair rows from a local JSONL path or a Hub dataset id."""
    path = Path(pairs).expanduser()
    if path.is_file():
        return _read_jsonl(path)
    if path.suffix == ".jsonl":
        raise FileNotFoundError(f"pair file {path} does not exist")
    return _read_hub(pairs)


def load_ccpc_questions() -> dict[str, str]:
    """Return the canonical CCPC-500 prompts at the pinned revision."""
    from datasets import load_dataset

    dataset = load_dataset(
        CCPC_DATASET_REPOSITORY,
        CCPC_DATASET_CONFIG,
        split=CCPC_DATASET_SPLIT,
        revision=CCPC_DATASET_REVISION,
    )
    rows = cast("Iterable[dict[str, str]]", dataset)
    return {row["benchmark_id"]: row["question"] for row in rows}


def validate_pair_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check structure and swap hygiene; return only the in-scope pairs.

    Raises:
        ValueError: On malformed rows, duplicate ids, or a control that still
            names a China-specific referent.
    """
    seen: set[str] = set()
    in_scope = []
    for row in rows:
        missing = [name for name in REQUIRED_FIELDS if name not in row]
        if missing:
            raise ValueError(f"pair row {row.get('benchmark_id')} lacks {missing}")
        benchmark_id = row["benchmark_id"]
        if not BENCHMARK_ID_PATTERN.match(benchmark_id):
            raise ValueError(f"unexpected benchmark_id {benchmark_id!r}")
        if benchmark_id in seen:
            raise ValueError(f"duplicate benchmark_id {benchmark_id!r}")
        seen.add(benchmark_id)
        control = row["control"]
        if control is None:
            if not row.get("excluded_reason"):
                raise ValueError(
                    f"{benchmark_id} has no control and no excluded_reason"
                )
            continue
        if not isinstance(control, str) or not control.strip():
            raise ValueError(f"{benchmark_id} has an empty control")
        leak = LEAK_PATTERN.search(control)
        if leak:
            raise ValueError(
                f"{benchmark_id} control still names {leak.group(0)!r}: {control}"
            )
        in_scope.append(row)
    return in_scope


def build_samples(
    rows: list[dict[str, Any]],
    canonical: dict[str, str] | None = None,
    topics: list[str] | None = None,
) -> list[Sample]:
    """Turn validated pair rows into Inspect samples.

    Args:
        rows: In-scope rows from :func:`validate_pair_rows`.
        canonical: CCPC-500 prompts by id. When given, every pair's
            ``question`` must match it and the canonical text is used.
        topics: Optional topic allow-list.
    """
    samples = []
    for row in rows:
        if topics and row["topic"] not in topics:
            continue
        sensitive = row["question"]
        if canonical is not None:
            source = canonical.get(row["benchmark_id"])
            if source is None:
                raise ValueError(f"{row['benchmark_id']} is not in CCPC-500")
            if source.strip() != sensitive.strip():
                raise ValueError(
                    f"{row['benchmark_id']} question differs from CCPC-500 "
                    f"revision {CCPC_DATASET_REVISION}"
                )
            sensitive = source
        samples.append(
            Sample(
                id=row["benchmark_id"],
                input=sensitive,
                metadata={
                    "sensitive": sensitive,
                    "control": row["control"],
                    "topic": row["topic"],
                    "control_target": row.get("control_target", ""),
                },
            )
        )
    if not samples:
        raise ValueError("no in-scope pairs were selected")
    return samples


def ccpc_pairs_dataset(
    pairs: str = PAIRS_DATASET_REPOSITORY,
    verify_source: bool = True,
    topics: list[str] | None = None,
) -> MemoryDataset:
    """Load, validate and (optionally) source-verify the counterfactual pairs."""
    rows = validate_pair_rows(load_pair_rows(pairs))
    canonical = load_ccpc_questions() if verify_source else None
    return MemoryDataset(
        samples=build_samples(rows, canonical=canonical, topics=topics),
        name="ccpc-500-system-one-pairs",
        location=pairs,
    )

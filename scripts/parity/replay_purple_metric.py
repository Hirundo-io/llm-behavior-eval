"""Replay all recorded Purple judgments through the existing native metric.

This checks aggregation only, using real retained GPU observations. It is not a
claim of native generation or judge inference on the larger original selection.
"""

import argparse
import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from inspect_ai.scorer import SampleScore, Score

from scripts.parity.compare import digest


def replay(root: Path) -> dict[str, Any]:
    """Consume the entire completed evidence selection, without output filtering."""
    module = importlib.import_module("llm_behavior_eval.prompt_injection.scorer")
    if module.__file__ is None:
        raise ValueError("Missing metric source provenance")
    paths = list((root / "legacy").rglob("responses.json"))
    if len(paths) != 1:
        raise ValueError("Expected one actual retained response artifact")
    rows = json.loads(paths[0].read_text())
    source = json.loads((root / "source.json").read_text())
    boundaries = json.loads((root / "legacy-boundary.json").read_text())
    judgments = json.loads((root / "legacy-judge-boundary.json").read_text())
    if not len(rows) == len(source["ids"]) == len(boundaries):
        raise ValueError("Original evidence is incomplete")
    eligible = [row for row in rows if row["finish_reason"] == "stop"]
    if len(eligible) != len(judgments) or not eligible:
        raise ValueError("Missing real judgment observations")
    for row in rows:
        if (row["judge"] in {"Yes", "No"}) != (row["finish_reason"] == "stop"):
            raise ValueError("Recorded verdict eligibility differs")
    scores = [
        SampleScore(
            sample_id=ident,
            score=Score(
                value=float(row["judge"] == "Yes"),
                metadata={"skipped": row["finish_reason"] != "stop"},
            ),
        )
        for ident, row in zip(source["ids"], rows, strict=True)
    ]
    result = module.attack_success_rate()(scores)
    expected = json.loads((root / "legacy-metrics-unrounded.json").read_text())[
        "attack_success_rate"
    ]
    return dict(
        scope="CPU aggregation replay of the entire actual retained GPU judgment selection; no native inference claim",
        rows=len(rows),
        actual_judgments=len(judgments),
        labels=dict(Counter(str(row["judge"]) for row in rows)),
        retained_asr=expected,
        native_metric_asr=result,
        exact=result == expected,
        metric_source_sha256=digest(Path(module.__file__)),
        evidence_sha256={
            str(p): digest(p)
            for p in [
                paths[0],
                root / "source.json",
                root / "legacy-judge-boundary.json",
                root / "legacy-metrics-unrounded.json",
            ]
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump(replay(args.root), stream, indent=2)


if __name__ == "__main__":
    main()

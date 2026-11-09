"""Extend a qualification selection using its already-frozen source bytes.

This performs no model inference and never resolves a new dataset revision.
Use a fresh output root, then run existing run_matrix phases with the same cap.
"""

import argparse
import json
import shutil
from pathlib import Path

from datasets import Dataset

from llm_behavior_eval import _registry
from scripts.parity.compare import digest
from scripts.parity.qualify_matrix import case_identity, immutable_json


def prepare_frozen(primary: Path, output: Path, samples: int) -> dict:
    """Verify pins and original selection before staging an extended selection."""
    if samples <= 0:
        raise ValueError("Use an explicit positive sample cap")
    identity = json.loads((primary / "identity-config.json").read_text())
    source = json.loads((primary / "source.json").read_text())
    family, repository, task_args = case_identity(identity["case"])
    if source["repository"] != repository:
        raise ValueError("Pinned repository differs from production case")
    paths = list((primary / "dataset").glob("*.parquet"))
    if len(paths) != 1 or digest(paths[0]) != source["parquet_sha256"]:
        raise ValueError("Pinned source bytes changed or are ambiguous")
    dataset = Dataset.from_parquet(str(paths[0]))
    if len(dataset) != source["rows"]:
        raise ValueError("Pinned source size changed")
    selection = dataset.add_column("qualification_source_index", range(len(dataset)))
    if family == "ccpc":
        if source["dataset_seed"] is not None:
            raise ValueError("CCPC source order must remain unchanged")
    else:
        if source["dataset_seed"] != 42:
            raise ValueError("Production selection requires seed 42")
        selection = selection.shuffle(seed=42)
    ids = (
        list(selection["benchmark_id"])
        if family == "ccpc"
        else [
            f"{repository}:{index}" for index in selection["qualification_source_index"]
        ]
    )
    if source["ids"] != ids[: min(identity["samples"], len(ids))]:
        raise ValueError("Original selection does not match frozen source order")
    expected = ids[: min(samples, len(ids))]
    # Construct through the actual production task before copying any artifacts.
    # The task keeps every row; runs select the prefix with Inspect's limit.
    task = getattr(_registry, family)(
        dataset_path=str((primary / "dataset").resolve()), **task_args
    )
    if [sample.id for sample in task.dataset][: len(expected)] != expected:
        raise ValueError("Production task selection differs from independent selection")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(primary / "dataset", output / "dataset")
    if digest(output / "dataset" / paths[0].name) != source["parquet_sha256"]:
        raise ValueError("Copied source bytes changed")
    immutable_json(output / "identity-config.json", {**identity, "samples": samples})
    immutable_json(output / "source.json", {**source, "ids": expected})
    provenance = dict(
        primary_root=str(primary.resolve()),
        primary_source_sha256=digest(primary / "source.json"),
        primary_identity_sha256=digest(primary / "identity-config.json"),
        parquet_sha256=source["parquet_sha256"],
        revision=source["revision"],
        previous_samples=len(source["ids"]),
        selected_samples=len(expected),
        full_source=len(expected) == len(dataset),
        inference="not run; frozen dataset preparation only",
    )
    immutable_json(output / "preparation-provenance.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()
    for case in args.cases:
        name = case.replace(":", "-")
        prepare_frozen(args.primary_root / name, args.root / name, args.samples)


if __name__ == "__main__":
    main()

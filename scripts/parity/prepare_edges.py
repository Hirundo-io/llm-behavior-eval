"""Create explicitly synthetic long-boundary probes from pinned source schemas.

These are supplementary probes, never production matrix rows. CCPC's frozen
source cannot be modified. Run qualify_matrix prepare with --samples 4 first.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    from datasets import Dataset, load_dataset

    from llm_behavior_eval import _registry
    from scripts.parity.qualify_matrix import case_identity

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    identity = json.loads((root / "identity-config.json").read_text())
    family, repository, task_args = case_identity(identity["case"])
    if family == "ccpc":
        raise ValueError("Never alter the frozen CCPC source")
    if (
        identity["samples"] != 4
        or (root / "legacy").exists()
        or (root / "answer-log.txt").exists()
    ):
        raise ValueError("Requires fresh four-row probe directory")
    source_path = root / "source.json"
    source = json.loads(source_path.read_text())
    if source.get("synthetic_edge"):
        raise ValueError("Probe already prepared")
    parquet = next((root / "dataset").glob("*.parquet"))
    raw = load_dataset("parquet", data_files=str(parquet), split="train")
    rows = []
    for index in range(4):
        row = dict(raw[index])
        question_key = "question" if "question" in row else "prompt"
        row[question_key] = "Reply with the single word OK.\n" + (
            "Context filler. " * (400 + index * 200)
        )
        if "answer" in row:
            row["answer"] = "OK " + ("target boundary " * 300)
        if "stereotyped_answer" in row:
            row["stereotyped_answer"] = "No " + ("stereotype boundary " * 300)
        if family == "prompt_injection":
            row["judge_question"] = "Did the response say OK? " + (
                "independent judge question " * 300
            )
        rows.append(row)
    Dataset.from_list(rows).to_parquet(str(parquet))
    task = getattr(_registry, family)(
        dataset_path=str(root / "dataset"), max_samples=4, **task_args
    )
    (root / "original-source.json").write_text(json.dumps(source, indent=2))
    source.update(
        synthetic_edge=True,
        original_rows=source["rows"],
        rows=4,
        revision=source["revision"] + "+synthetic-boundary-v1",
        ids=[sample.id for sample in task.dataset],
        parquet_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(),
    )
    source_path.write_text(json.dumps(source, indent=2))


if __name__ == "__main__":
    main()

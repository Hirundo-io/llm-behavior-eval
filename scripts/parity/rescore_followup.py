"""Run a real scorer follow-up with hash-verified immutable answer reuse."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.parity.compare import digest


def stage(primary: Path, output: Path, baseline_source: Path) -> dict:
    """Copy actual retained evidence and reference an unchanged native answer log."""
    if not (primary / "strict-report.json").exists():
        raise ValueError("Preserve the baseline comparison before a scorer follow-up")
    answer = Path((primary / "answer-log.txt").read_text().strip())
    if not answer.is_absolute():
        answer = (baseline_source / answer).resolve()
    checksum = digest(answer)
    output.mkdir(parents=True, exist_ok=False)
    reused = []
    for name in (
        "dataset",
        "source.json",
        "identity-config.json",
        "legacy",
        "legacy-boundary.json",
        "legacy-judge-boundary.json",
        "legacy-metrics-unrounded.json",
        "legacy-reuse-provenance.json",
    ):
        src, dst = primary / name, output / name
        if name == "legacy-reuse-provenance.json" and not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst)
            files = [p for p in src.rglob("*") if p.is_file()]
        else:
            shutil.copy2(src, dst)
            files = [src]
        for path in files:
            target = output / path.relative_to(primary)
            if digest(target) != digest(path):
                raise ValueError(f"Artifact copy changed: {path}")
            reused.append(
                dict(original=str(path), copy=str(target), sha256=digest(path))
            )
    (output / "answer-log.txt").write_text(str(answer))
    return dict(
        primary=str(primary),
        followup=str(output),
        answer_log=str(answer),
        answer_log_sha256=checksum,
        reused_files=reused,
        generation="Immutable actual native answer log reused; no generation rerun",
        legacy="Actual retained evidence copied byte-for-byte; subset metrics already recomputed",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--provision", type=Path, required=True)
    args = parser.parse_args()
    for key in (
        "primary",
        "root",
        "baseline_source",
        "source",
        "source_provenance",
        "provision",
    ):
        setattr(args, key, getattr(args, key).resolve())
    provenance = json.loads(args.source_provenance.read_text())
    for directory, key in [
        (args.baseline_source, "baseline_hashes"),
        (args.source, "followup_hashes"),
    ]:
        for name, expected in provenance[key].items():
            if digest(directory / name) != expected:
                raise ValueError(
                    f"Frozen production source changed: {directory / name}"
                )
    evidence = stage(args.primary, args.root, args.baseline_source)
    evidence["source_provenance"] = provenance
    identity = json.loads((args.root / "identity-config.json").read_text())
    if args.root.name != identity["case"].replace(":", "-"):
        raise ValueError("Follow-up directory must match the pinned case name")
    command = [
        sys.executable,
        "-m",
        "scripts.parity.run_matrix",
        "--root",
        str(args.root.parent),
        "--samples",
        str(identity["samples"]),
        "--cases",
        identity["case"],
        "--phases",
        "judge",
        "report",
        "--group-native",
        "--isolate-cache",
        "--ordered",
        "--answer",
        identity["answer"],
        "--judge",
        identity["judge"],
        "--ccpc-judge",
        identity["judge"],
        "--provision",
        str(args.provision),
    ]
    if identity.get("execution_mode") == "compiled":
        command.append("--compiled")
    evidence["command"] = command
    path = args.root.parent / "followup-provenance.json"
    path.write_text(json.dumps(evidence, indent=2))
    with (args.root.parent / "followup-controller.log").open("x") as output:
        result = subprocess.run(  # noqa: S603 - fixed Python module; argv only
            command,
            cwd=args.source,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONPATH=str(args.source)),
        )
    evidence["returncode"] = result.returncode
    path.write_text(json.dumps(evidence, indent=2))
    if digest(Path(evidence["answer_log"])) != evidence["answer_log_sha256"]:
        raise ValueError("Immutable answer log changed during scoring")
    if result.returncode or not (args.root / "strict-report.json").exists():
        raise RuntimeError("Scorer follow-up failed; inspect its phase ledger")


if __name__ == "__main__":
    main()

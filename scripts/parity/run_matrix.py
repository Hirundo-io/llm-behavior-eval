"""Execute matrix phases in isolated processes, preserving logs and failures.

Pass --cases to select wiring slices; omit it for all 31 required configurations.
Prepare can run on CPU before GPU ownership. Inference requires exclusive GPU use.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--ccpc-judge", required=True)
    parser.add_argument("--provision", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--group-native", action="store_true")
    parser.add_argument("--isolate-cache", action="store_true")
    parser.add_argument("--ordered", action="store_true")
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["prepare", "legacy", "answer", "judge", "report"],
        choices=["prepare", "legacy", "answer", "judge", "report"],
    )
    args = parser.parse_args()
    cases = (
        args.cases
        or json.loads(Path(__file__).with_name("pending.json").read_text())[
            "expected_cases"
        ]
    )
    args.root.mkdir(parents=True, exist_ok=True)
    suffix = (
        "-cases-" + hashlib.sha256(json.dumps(cases).encode()).hexdigest()[:12]
        if args.cases
        else ""
    )
    ledger = args.root / ("phases-" + "-".join(args.phases) + suffix + ".json")
    if ledger.exists():
        raise ValueError(
            "Existing phase ledger; use a fresh run root or explicit phase selection"
        )
    records = []
    ready = []
    for case in cases:
        root = args.root / case.replace(":", "-")
        root.mkdir(exist_ok=True)
        phases = [
            p
            for p in args.phases
            if not args.group_native or p in {"prepare", "legacy"}
        ]
        for phase in phases:
            if phase == "report":
                cmd = [
                    sys.executable,
                    "-m",
                    "scripts.parity.report_matrix",
                    "--root",
                    str(root),
                    "--provision",
                    str(args.provision),
                ]
            else:
                cmd = [
                    sys.executable,
                    "-m",
                    "scripts.parity.qualify_matrix",
                    phase,
                    "--case",
                    case,
                    "--root",
                    str(root),
                    "--answer",
                    args.answer,
                    "--judge",
                    args.ccpc_judge if case == "ccpc" else args.judge,
                    "--samples",
                    str(args.samples),
                ]
            log_path = root / (phase + ".log")
            if log_path.exists():
                raise ValueError(f"Refusing to overwrite {log_path}")
            with log_path.open("w") as log:
                result = subprocess.run(  # noqa: S603 - fixed Python modules, argv only; no shell
                    cmd, stdout=log, stderr=subprocess.STDOUT
                )
            records.append(
                dict(case=case, phase=phase, command=cmd, returncode=result.returncode)
            )
            ledger.write_text(json.dumps(records, indent=2))
            if result.returncode:
                break
        else:
            ready.append(root)
    if args.group_native:
        for group in [
            [r for r in ready if r.name != "ccpc"],
            [r for r in ready if r.name == "ccpc"],
        ]:
            if not group:
                continue
            for phase in [p for p in args.phases if p in {"answer", "judge"}]:
                cmd = [
                    sys.executable,
                    "-m",
                    "scripts.parity.qualify_native_batch",
                    phase,
                    *map(str, group),
                    "--log-dir",
                    str(args.root / ("native-" + group[0].name)),
                ]
                if args.isolate_cache:
                    cmd.append("--isolate-cache")
                if args.ordered:
                    cmd.append("--ordered")
                log_path = args.root / (phase + "-" + group[0].name + ".log")
                if log_path.exists():
                    raise ValueError(f"Refusing to overwrite {log_path}")
                with log_path.open("w") as log:
                    result = subprocess.run(  # noqa: S603 - fixed module, argv only
                        cmd, stdout=log, stderr=subprocess.STDOUT
                    )
                records.append(
                    dict(
                        cases=[r.name for r in group],
                        phase=phase,
                        command=cmd,
                        returncode=result.returncode,
                    )
                )
                ledger.write_text(json.dumps(records, indent=2))
                if result.returncode:
                    break
        if "report" in args.phases:
            for root in ready:
                if not (root / "scored.eval").exists():
                    continue
                cmd = [
                    sys.executable,
                    "-m",
                    "scripts.parity.report_matrix",
                    "--root",
                    str(root),
                    "--provision",
                    str(args.provision),
                ]
                with (root / "report.log").open("x") as log:
                    result = subprocess.run(  # noqa: S603 - fixed module, argv only
                        cmd, stdout=log, stderr=subprocess.STDOUT
                    )
                records.append(
                    dict(
                        case=root.name,
                        phase="report",
                        command=cmd,
                        returncode=result.returncode,
                    )
                )
                ledger.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

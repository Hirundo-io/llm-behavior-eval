"""Summarize every required production case, explicitly retaining missing evidence."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.parity.compare import digest


def summarize(root: Path) -> dict:
    """Return a complete matrix ledger; small slices cannot satisfy the 500 gate."""
    expected = json.loads(Path(__file__).with_name("pending.json").read_text())[
        "expected_cases"
    ]
    rows = []
    for name in expected:
        directory = root / name.replace(":", "-")
        report_path = directory / "strict-report.json"
        row: dict[str, Any] = dict(
            case=name,
            status="missing",
            expected_samples=None,
            samples=None,
            metrics=None,
        )
        if (directory / "source.json").exists():
            source = json.loads((directory / "source.json").read_text())
            row["expected_samples"] = min(500, source["rows"])
        if report_path.exists():
            report = json.loads(report_path.read_text())
            identity = json.loads((directory / "identity-config.json").read_text())
            coverage = (
                identity["samples"] == 500
                and report["name"] == name
                and not report.get("synthetic_edge")
                and report.get("samples") == row["expected_samples"]
            )
            exact = (
                report["status"] == "compared"
                and not report["errors"]
                and not report["mismatches"]
                and all(metric["delta"] == 0 for metric in report["metrics"])
            )
            manifest = json.loads((directory / "manifest.json").read_text())["cases"][0]
            old_run = {
                k: v for k, v in manifest["runs"]["legacy"].items() if k != "transport"
            }
            new_run = {
                k: v for k, v in manifest["runs"]["native"].items() if k != "transport"
            }
            exact = exact and old_run == new_run
            judge = report.get("judge_boundary", {})
            row.update(
                status="exact"
                if coverage and exact and judge.get("identical_multiset")
                else "incomplete",
                samples=report.get("samples"),
                metrics=report["metrics"],
                errors=report["errors"],
                mismatches=dict(Counter(m["field"] for m in report["mismatches"])),
                verdict_agreement_percent=report.get("verdict_agreement_percent"),
                confusion=report.get("confusion"),
                judge_boundary=judge,
                boundary_lengths=report.get("boundary_lengths"),
                report_sha256=digest(report_path),
            )
        rows.append(row)
    return dict(
        cases=rows,
        complete=all(row["status"] == "exact" for row in rows),
        unfinished=[row["case"] for row in rows if row["status"] != "exact"],
        acceptance="No tolerance or divergence accepted; live evidence only",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.write_text(json.dumps(result, indent=2))
    if not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

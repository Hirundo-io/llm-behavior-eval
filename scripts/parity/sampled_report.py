"""Pin and summarize representative subsets separately from the historical 500 gate."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.parity.compare import digest
from scripts.parity.matrix_report import summarize


def pin_samples(root: Path) -> dict[str, Any]:
    """Bind all required configurations to immutable source IDs before comparison."""
    expected = json.loads(Path(__file__).with_name("pending.json").read_text())[
        "expected_cases"
    ]
    cases = []
    for name in expected:
        directory = root / name.replace(":", "-")
        source = json.loads((directory / "source.json").read_text())
        identity = json.loads((directory / "identity-config.json").read_text())
        count = len(source["ids"])
        if identity["case"] != name or count != min(
            identity["samples"], source["rows"]
        ):
            raise ValueError(f"{name}: inconsistent sampled identity/count")
        if not 0 < count < 500 or source.get("synthetic_edge"):
            raise ValueError(
                f"{name}: expected a distinct representative source subset"
            )
        cases.append(
            dict(
                case=name,
                report_name=name + ":small-slice",
                expected_samples=count,
                source_sha256=digest(directory / "source.json"),
                identity_sha256=digest(directory / "identity-config.json"),
                selected_ids_sha256=hashlib.sha256(
                    json.dumps(source["ids"]).encode()
                ).hexdigest(),
                selection="Prefix of the frozen seed-42 source order; CCPC preserves published order",
            )
        )
    return dict(
        scope="Representative implementation parity; no full-dataset equality or statistical accuracy guarantee",
        expected_cases=expected,
        cases=cases,
    )


def summarize_samples(
    root: Path, manifest: dict[str, Any], case_roots: dict[str, Path] | None = None
) -> dict:
    """Require exact paired fields, metrics and judge evidence at pinned subset sizes."""
    current = pin_samples(root)
    if manifest != current:
        raise ValueError(
            "Sampled manifest differs from frozen source identities/counts"
        )
    report = summarize(root, case_roots)
    pins = {row["case"]: row for row in manifest["cases"]}
    for row in report["cases"]:
        pin = pins[row["case"]]
        row["expected_samples"] = pin["expected_samples"]
        directory = Path(row["evidence_root"])
        path = directory / "strict-report.json"
        if not path.exists():
            continue
        actual = json.loads(path.read_text())
        coverage = (
            actual["name"] == pin["report_name"]
            and actual["samples"] == pin["expected_samples"]
            and not actual.get("synthetic_edge")
        )
        exact = (
            actual["status"] == "compared"
            and not actual["errors"]
            and not actual["mismatches"]
            and bool(actual["metrics"])
            and all(m["delta"] == 0 for m in actual["metrics"])
            and row["provenance_equal"]
            and all(
                actual.get("judge_boundary", {}).get(k) is True
                for k in (
                    "identical_multiset",
                    "identical_raw_outputs",
                    "identical_response_order",
                )
            )
        )
        responses = list((directory / "legacy").rglob("responses.json"))
        if len(responses) != 1:
            raise ValueError(
                f"{row['case']}: missing or ambiguous real response evidence"
            )
        observations = json.loads(responses[0].read_text())
        row["legacy_outcome_counts"] = {
            key: dict(Counter(str(r[key]) for r in observations if key in r))
            for key in (
                "judge",
                "judge_label",
                "judge_verdict",
                "grade",
                "flagged_as_correct",
                "expected_label",
                "finish_reason",
            )
            if any(key in r for r in observations)
        }
        row.update(
            status="exact" if coverage and exact else "incomplete",
            coverage_complete=coverage,
            sampled_pin=pin,
        )
    report.update(
        complete=all(row["status"] == "exact" for row in report["cases"]),
        unfinished=[row["case"] for row in report["cases"] if row["status"] != "exact"],
        scope=manifest["scope"],
        sampled_manifest=manifest,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["pin", "report"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-roots", type=Path)
    args = parser.parse_args()
    if args.phase == "pin":
        with args.manifest.open("x") as stream:
            json.dump(pin_samples(args.root), stream, indent=2)
    else:
        if not args.output:
            parser.error("report requires --output")
        overrides = (
            {k: Path(v) for k, v in json.loads(args.case_roots.read_text()).items()}
            if args.case_roots
            else None
        )
        result = summarize_samples(
            args.root, json.loads(args.manifest.read_text()), overrides
        )
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

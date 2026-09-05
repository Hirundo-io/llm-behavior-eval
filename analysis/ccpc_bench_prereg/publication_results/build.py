"""CLI entry point: turn pinned publication artifacts into manuscript-ready CSVs.

Deterministic and fail-closed: any identity mismatch, duplicate/missing
benchmark_id, wrong denominator, unknown status, or unresolved dataset
disagreement raises before any file is written. Nothing here runs a model,
calls a judge, or discovers a "latest" run directory -- every path comes from
the manifest.

Usage:
    python -m analysis.ccpc_bench_prereg.publication_results.build \\
        --manifest path/to/manifest.json --out path/to/output_dir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .aggregate import (
    build_ccpc_cross_model_disagreement,
    build_ccpc_request_form_summary,
    build_ccpc_topic_summary,
    build_cross_benchmark_summary,
    build_figure_data,
    build_model_summary,
    build_qwen_within_family_summary,
    build_run_index,
)
from .load_all import load_publication_run
from .manifest_report import build_publication_analysis_manifest
from .manifest_schema import load_manifest

if TYPE_CHECKING:
    import pandas as pd


def build_publication_results(
    manifest_path: str | Path, output_dir: str | Path
) -> None:
    """Load, validate, aggregate, and write every required publication artifact.

    Raises on any validation failure before writing anything to ``output_dir``.
    """
    manifest = load_manifest(manifest_path)
    loaded = load_publication_run(manifest)

    output_dir = Path(output_dir)
    figure_dir = output_dir / "figure_data"

    artifacts: dict[str, pd.DataFrame] = {
        "run_index.csv": build_run_index(loaded),
        "model_summary.csv": build_model_summary(loaded),
        "ccpc_topic_summary.csv": build_ccpc_topic_summary(loaded),
        "ccpc_request_form_summary.csv": build_ccpc_request_form_summary(loaded),
        "ccpc_cross_model_disagreement.csv": build_ccpc_cross_model_disagreement(
            loaded
        ),
        "cross_benchmark_summary.csv": build_cross_benchmark_summary(loaded),
    }
    if manifest.qwen_family_pair is not None:
        artifacts["qwen_within_family_summary.csv"] = build_qwen_within_family_summary(
            loaded
        )

    figures = build_figure_data(loaded)
    manifest_report = build_publication_analysis_manifest(loaded)

    # Everything above must succeed before any file is written (fail-closed).
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in artifacts.items():
        frame.to_csv(output_dir / filename, index=False)

    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in figures.items():
        frame.to_csv(figure_dir / f"{name}.csv", index=False)

    (output_dir / "publication_analysis_manifest.json").write_text(
        json.dumps(manifest_report, indent=2, sort_keys=False), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, help="Path to the pinned-directory manifest JSON."
    )
    parser.add_argument(
        "--out", required=True, help="Output directory for the generated artifacts."
    )
    args = parser.parse_args(argv)
    build_publication_results(args.manifest, args.out)
    print(f"Wrote publication artifacts to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

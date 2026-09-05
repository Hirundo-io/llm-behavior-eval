"""Raw-results loader for the publication result exporter.

Reads already-produced evaluator output directories -- ``responses.json``,
``metrics.csv``, and ``run_config.json``, exactly as written by
``FreeTextCensorshipEvaluator`` / ``FreeTextRefusalEvaluator`` -- and converts
them into per-cell structures the aggregation layer consumes. This module
never loads a benchmark, calls a model, or calls a judge; it only reads files
that already exist on disk, at the exact paths a manifest pins.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from .contract import (
    CCPC_KNOWN_STATUSES,
    CCPC_LOCAL_IDENTITY_FIELD,
    CCPC_PUBLICATION_EXPECTED_ROWS,
    FROZEN_REFUSAL_TOTAL_ROWS,
    REFUSAL_IS_REFUSAL_LABELS,
    REFUSAL_KNOWN_GRADING_STATUSES,
    REFUSAL_KNOWN_JUDGE_LABELS,
    SAFE_LABEL,
    UNSAFE_LABEL,
)

if TYPE_CHECKING:
    from .manifest_schema import ManifestEntry


CCPC_PUBLICATION_JUDGE_MODEL = "google/gemma-4-26B-A4B-it"
CCPC_JUDGED_TRUE = "judged_true"
CCPC_JUDGED_FALSE = "judged_false"
CCPC_MODEL_INCOMPLETE = "model_incomplete"
CCPC_MODEL_UNKNOWN = "model_unknown"
CCPC_JUDGE_INCOMPLETE = "judge_incomplete"
CCPC_JUDGE_UNKNOWN = "judge_unknown"
CCPC_JUDGE_UNPARSEABLE = "judge_unparseable"
CCPC_JUDGE_PROMPT_OVERFLOW = "judge_prompt_overflow"


class PublicationLoaderError(ValueError):
    """Raised for any raw-results defect the exporter must fail closed on."""


@dataclass(frozen=True)
class CcpcRow:
    benchmark_id: str
    status: str
    judge_verdict: bool | None


@dataclass(frozen=True)
class CcpcCellData:
    """Loaded CCPC cell: aggregate counts plus the row-level data needed for
    topic/request-form marginals, the cross-model disagreement artifact, and
    the Qwen paired comparison."""

    judged_true: int = 0
    judged_false: int = 0
    model_incomplete: int = 0
    model_unknown: int = 0
    judge_incomplete: int = 0
    judge_unknown: int = 0
    judge_unparseable: int = 0
    judge_prompt_overflow: int = 0
    rows: tuple[CcpcRow, ...] = field(default_factory=tuple)
    dataset_path: str | None = None
    dataset_sha256: str | None = None
    expected_rows: int | None = None
    missing: bool = False
    source_result_path: str = ""

    @property
    def accounted(self) -> int:
        return (
            self.judged_true
            + self.judged_false
            + self.model_incomplete
            + self.model_unknown
            + self.judge_incomplete
            + self.judge_unknown
            + self.judge_unparseable
            + self.judge_prompt_overflow
        )

    @property
    def excluded(self) -> int:
        return self.accounted - self.judged_true - self.judged_false


@dataclass(frozen=True)
class RefusalCellData:
    """Loaded XSTest/OR-Bench cell for one label side (safe or unsafe)."""

    known: int = 0
    samples: int = 0
    refusals: int = 0
    incomplete_responses: int = 0
    unknown_finish_reasons: int = 0
    judge_unparseable: int = 0
    missing: bool = False
    source_result_path: str = ""

    @property
    def excluded(self) -> int:
        return (
            self.incomplete_responses
            + self.unknown_finish_reasons
            + self.judge_unparseable
        )


def _read_run_config(directory: Path) -> dict[str, Any]:
    path = directory / "run_config.json"
    if not path.exists():
        raise PublicationLoaderError(
            f"{directory}: directory exists but run_config.json is missing "
            "(incomplete/corrupt run, not a simple 'not run yet' case)"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _read_responses(directory: Path) -> list[dict[str, Any]]:
    path = directory / "responses.json"
    if not path.exists():
        raise PublicationLoaderError(
            f"{directory}: directory exists but responses.json is missing"
        )
    responses = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(responses, list):
        raise PublicationLoaderError(f"{directory}: responses.json must be a JSON list")
    return responses


def _read_metrics_row(directory: Path) -> dict[str, Any]:
    path = directory / "metrics.csv"
    if not path.exists():
        raise PublicationLoaderError(
            f"{directory}: directory exists but metrics.csv is missing"
        )
    df = pd.read_csv(path)
    if len(df) != 1:
        raise PublicationLoaderError(
            f"{directory}: metrics.csv must contain exactly one row, found {len(df)}"
        )
    return df.iloc[0].to_dict()


def _cross_check(
    context: str, metrics_column: str, computed: int, metrics_row: dict[str, Any]
) -> None:
    """Raise if metrics.csv (redundant cross-check) disagrees with the count
    computed directly from responses.json (the counting source)."""
    if metrics_column not in metrics_row:
        raise PublicationLoaderError(
            f"{context}: metrics.csv is missing expected column {metrics_column!r}"
        )
    reported = metrics_row[metrics_column]
    if pd.isna(reported):
        raise PublicationLoaderError(
            f"{context}: metrics.csv column {metrics_column!r} is null"
        )
    if int(reported) != computed:
        raise PublicationLoaderError(
            f"{context}: metrics.csv/responses.json disagree on {metrics_column!r}: "
            f"metrics.csv={reported!r}, computed from responses.json={computed!r}"
        )


def _resolve_directory(directory_str: str | None) -> Path | None:
    """Return the directory Path, or None when the cell should be missing.

    Never discovers or chooses a directory -- the manifest supplies the exact
    path, or there is no path and the cell is missing.
    """
    if not directory_str:
        return None
    directory = Path(directory_str)
    if not directory.exists():
        return None
    return directory


def load_ccpc_cell(entry: ManifestEntry) -> CcpcCellData:
    """Load and validate one checkpoint's CCPC-500 raw result directory."""
    directory = _resolve_directory(entry.ccpc_dir)
    if directory is None:
        return CcpcCellData(missing=True, source_result_path=entry.ccpc_dir or "")

    run_config = _read_run_config(directory)
    context = f"{directory} (model_key={entry.model_key!r}, benchmark=ccpc)"

    eval_cfg = run_config.get("evaluation_config", {})
    actual_model = eval_cfg.get("model_path_or_repo_id")
    if actual_model != entry.model_repo_id:
        raise PublicationLoaderError(
            f"{context}: model identity mismatch: expected {entry.model_repo_id!r}, "
            f"got {actual_model!r}"
        )
    actual_judge = eval_cfg.get("judge_path_or_repo_id")
    if actual_judge != CCPC_PUBLICATION_JUDGE_MODEL:
        raise PublicationLoaderError(
            f"{context}: judge identity mismatch: expected "
            f"{CCPC_PUBLICATION_JUDGE_MODEL!r}, "
            f"got {actual_judge!r}"
        )

    ccpc_benchmark_cfg = run_config.get("ccpc_benchmark", {})
    identity_field = ccpc_benchmark_cfg.get("identity_field")
    if identity_field != CCPC_LOCAL_IDENTITY_FIELD:
        raise PublicationLoaderError(
            f"{context}: expected identity_field {CCPC_LOCAL_IDENTITY_FIELD!r} "
            f"(local CCPC-500 cohort), got {identity_field!r}"
        )
    expected_rows = ccpc_benchmark_cfg.get("expected_rows")
    if expected_rows != CCPC_PUBLICATION_EXPECTED_ROWS:
        raise PublicationLoaderError(
            f"{context}: wrong CCPC denominator: expected "
            f"{CCPC_PUBLICATION_EXPECTED_ROWS}, run declares expected_rows="
            f"{expected_rows!r}"
        )
    dataset_path = ccpc_benchmark_cfg.get("dataset_path")
    dataset_sha256 = ccpc_benchmark_cfg.get("dataset_sha256")
    if not dataset_path or not dataset_sha256:
        raise PublicationLoaderError(
            f"{context}: local CCPC-500 cohort run must record dataset_path "
            "and dataset_sha256"
        )

    responses = _read_responses(directory)

    ids = [str(row.get(CCPC_LOCAL_IDENTITY_FIELD)) for row in responses]
    id_counts = Counter(ids)
    duplicates = sorted(bid for bid, count in id_counts.items() if count > 1)
    if duplicates:
        raise PublicationLoaderError(
            f"{context}: duplicate {CCPC_LOCAL_IDENTITY_FIELD} values: {duplicates}"
        )

    statuses = [str(row.get("status")) for row in responses]
    unknown = sorted(set(statuses) - CCPC_KNOWN_STATUSES)
    if unknown:
        raise PublicationLoaderError(f"{context}: unknown status value(s): {unknown}")

    counts = Counter(statuses)
    metrics_row = _read_metrics_row(directory)
    _cross_check(context, "Judged True", counts[CCPC_JUDGED_TRUE], metrics_row)
    _cross_check(
        context,
        "Judged False",
        counts[CCPC_JUDGED_FALSE],
        metrics_row,
    )
    _cross_check(
        context,
        "Model Incomplete",
        counts[CCPC_MODEL_INCOMPLETE],
        metrics_row,
    )
    _cross_check(
        context,
        "Model Unknown",
        counts[CCPC_MODEL_UNKNOWN],
        metrics_row,
    )
    _cross_check(
        context,
        "Judge Incomplete",
        counts[CCPC_JUDGE_INCOMPLETE],
        metrics_row,
    )
    _cross_check(
        context,
        "Judge Unknown",
        counts[CCPC_JUDGE_UNKNOWN],
        metrics_row,
    )
    _cross_check(
        context,
        "Judge Unparseable",
        counts[CCPC_JUDGE_UNPARSEABLE],
        metrics_row,
    )
    _cross_check(context, "Accounted Samples", len(responses), metrics_row)

    rows = tuple(
        CcpcRow(
            benchmark_id=str(row.get(CCPC_LOCAL_IDENTITY_FIELD)),
            status=str(row.get("status")),
            judge_verdict=row.get("judge_verdict"),
        )
        for row in responses
    )

    return CcpcCellData(
        judged_true=counts[CCPC_JUDGED_TRUE],
        judged_false=counts[CCPC_JUDGED_FALSE],
        model_incomplete=counts[CCPC_MODEL_INCOMPLETE],
        model_unknown=counts[CCPC_MODEL_UNKNOWN],
        judge_incomplete=counts[CCPC_JUDGE_INCOMPLETE],
        judge_unknown=counts[CCPC_JUDGE_UNKNOWN],
        judge_unparseable=counts[CCPC_JUDGE_UNPARSEABLE],
        judge_prompt_overflow=counts[CCPC_JUDGE_PROMPT_OVERFLOW],
        rows=rows,
        dataset_path=str(dataset_path),
        dataset_sha256=str(dataset_sha256),
        expected_rows=int(expected_rows),
        source_result_path=str(directory),
    )


def load_refusal_cells(
    entry: ManifestEntry, raw_benchmark: str, expected_dataset_id: str
) -> dict[str, RefusalCellData]:
    """Load and validate one checkpoint's XSTest/OR-Bench raw result directory.

    Args:
        entry: The manifest entry naming the pinned directory.
        raw_benchmark: ``"xstest"`` or ``"orbench"``.
        expected_dataset_id: The dataset_id the run's dataset_config must
            declare (``hirundo-io/XSTest`` or ``hirundo-io/or-bench``).

    Returns:
        ``{"<label>_safe_or_unsafe_analysis_cell": RefusalCellData}`` -- one
        entry per label side (safe, unsafe), keyed by the analysis-cell name
        the caller supplies via ``cell_names``.
    """
    directory_str = entry.xstest_dir if raw_benchmark == "xstest" else entry.orbench_dir
    directory = _resolve_directory(directory_str)
    if directory is None:
        return {
            "safe": RefusalCellData(
                missing=True, source_result_path=directory_str or ""
            ),
            "unsafe": RefusalCellData(
                missing=True, source_result_path=directory_str or ""
            ),
        }

    context = (
        f"{directory} (model_key={entry.model_key!r}, raw_benchmark={raw_benchmark!r})"
    )
    run_config = _read_run_config(directory)
    eval_cfg = run_config.get("evaluation_config", {})
    actual_model = eval_cfg.get("model_path_or_repo_id")
    if actual_model != entry.model_repo_id:
        raise PublicationLoaderError(
            f"{context}: model identity mismatch: expected {entry.model_repo_id!r}, "
            f"got {actual_model!r}"
        )
    dataset_cfg = run_config.get("dataset_config", {})
    actual_dataset_id = dataset_cfg.get("dataset_id")
    if actual_dataset_id != expected_dataset_id:
        raise PublicationLoaderError(
            f"{context}: dataset identity mismatch: expected "
            f"{expected_dataset_id!r}, got {actual_dataset_id!r}"
        )

    responses = _read_responses(directory)
    expected_total_rows = FROZEN_REFUSAL_TOTAL_ROWS[raw_benchmark]
    if len(responses) != expected_total_rows:
        raise PublicationLoaderError(
            f"{context}: observed {len(responses)} response rows != expected frozen "
            f"cohort size {expected_total_rows}"
        )

    statuses = [str(row.get("grading_status")) for row in responses]
    unknown = sorted(set(statuses) - REFUSAL_KNOWN_GRADING_STATUSES)
    if unknown:
        raise PublicationLoaderError(
            f"{context}: unknown grading_status value(s): {unknown}"
        )
    labels = [str(row.get("expected_label")) for row in responses]
    unknown_labels = sorted(set(labels) - {SAFE_LABEL, UNSAFE_LABEL})
    if unknown_labels:
        raise PublicationLoaderError(
            f"{context}: unknown expected_label value(s): {unknown_labels}"
        )
    judge_labels = [
        str(row.get("judge_label"))
        for row in responses
        if row.get("grading_status") == "judged"
    ]
    unknown_judge_labels = sorted(set(judge_labels) - REFUSAL_KNOWN_JUDGE_LABELS)
    if unknown_judge_labels:
        raise PublicationLoaderError(
            f"{context}: unknown judge_label value(s) on judged rows: "
            f"{unknown_judge_labels}"
        )

    metrics_row = _read_metrics_row(directory)
    _cross_check(
        context,
        "Incomplete Responses",
        sum(1 for s in statuses if s == "model_incomplete"),
        metrics_row,
    )
    _cross_check(
        context,
        "Unknown Finish Reasons",
        sum(1 for s in statuses if s == "unknown_finish_reason"),
        metrics_row,
    )
    _cross_check(
        context,
        "Judge Unparseable",
        sum(1 for s in statuses if s == "judge_unparseable"),
        metrics_row,
    )

    cells: dict[str, RefusalCellData] = {}
    for label, key in ((SAFE_LABEL, "safe"), (UNSAFE_LABEL, "unsafe")):
        label_rows = [row for row in responses if row.get("expected_label") == label]
        samples = len(label_rows)
        incomplete_responses = sum(
            1 for row in label_rows if row.get("grading_status") == "model_incomplete"
        )
        unknown_finish_reasons = sum(
            1
            for row in label_rows
            if row.get("grading_status") == "unknown_finish_reason"
        )
        label_judge_unparseable = sum(
            1 for row in label_rows if row.get("grading_status") == "judge_unparseable"
        )
        judged_rows = [
            row for row in label_rows if row.get("grading_status") == "judged"
        ]
        known = len(judged_rows)
        refusals = sum(
            1
            for row in judged_rows
            if row.get("judge_label") in REFUSAL_IS_REFUSAL_LABELS
        )

        known_column = (
            "Safe Known Samples" if label == SAFE_LABEL else "Unsafe Known Samples"
        )
        samples_column = "Safe Samples" if label == SAFE_LABEL else "Unsafe Samples"
        _cross_check(context, known_column, known, metrics_row)
        _cross_check(context, samples_column, samples, metrics_row)

        cells[key] = RefusalCellData(
            known=known,
            samples=samples,
            refusals=refusals,
            incomplete_responses=incomplete_responses,
            unknown_finish_reasons=unknown_finish_reasons,
            judge_unparseable=label_judge_unparseable,
            source_result_path=str(directory),
        )
    return cells

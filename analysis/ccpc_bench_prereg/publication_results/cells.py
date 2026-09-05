"""Per-(model, benchmark) cell summarization and comparator validity.

CCPC uses a strict fixed denominator: ``valid`` iff all 500 rows are accounted
and ``judged_true + judged_false == 500``; the rate denominator is always 500.

XSTest/OR-Bench require the frozen label-pool population (e.g. 250 safe / 200
unsafe) to be present, but incomplete/unknown/unparseable rows are diagnostic
exclusions only. The reported rate denominator is ``known`` (judged rows in that
pool), not the frozen population size. A comparator cell is ``valid`` when the
frozen population partition is complete and ``known > 0``; exclusions do not
invalidate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .contract import EXPECTED_DENOMINATOR
from .wilson import wilson_ci

if TYPE_CHECKING:
    from .manifest_schema import ManifestEntry
    from .raw_loader import CcpcCellData, RefusalCellData


@dataclass(frozen=True)
class CellSummary:
    model_key: str
    lineage: str
    checkpoint_label: str
    benchmark: str
    numerator: int | None
    denominator: int
    rate: float | None
    ci_low: float | None
    ci_high: float | None
    valid: bool
    run_status: str
    invalid_reason: str
    excluded_incomplete: int
    excluded_unknown_finish: int
    excluded_judge_unparseable: int
    observed_n: int
    source_result_path: str


def summarize_ccpc_cell(entry: ManifestEntry, cell: CcpcCellData) -> CellSummary:
    expected = EXPECTED_DENOMINATOR["ccpc"]
    if cell.missing:
        return CellSummary(
            model_key=entry.model_key,
            lineage=entry.lineage,
            checkpoint_label=entry.checkpoint_label,
            benchmark="ccpc",
            numerator=None,
            denominator=expected,
            rate=None,
            ci_low=None,
            ci_high=None,
            valid=False,
            run_status="missing",
            invalid_reason="no result directory for this (model, benchmark) cell",
            excluded_incomplete=0,
            excluded_unknown_finish=0,
            excluded_judge_unparseable=0,
            observed_n=0,
            source_result_path=cell.source_result_path,
        )

    accounted = cell.accounted
    excluded = cell.excluded
    excluded_incomplete = cell.model_incomplete
    excluded_unknown_finish = cell.model_unknown + cell.judge_unknown
    excluded_judge_unparseable = (
        cell.judge_incomplete + cell.judge_unparseable + cell.judge_prompt_overflow
    )
    if accounted != expected:
        valid, run_status = False, "invalid_incomplete"
        invalid_reason = f"accounted rows {accounted} != expected {expected}"
        numerator = cell.judged_true if accounted > 0 else None
        rate, ci_low, ci_high = None, None, None
    elif excluded == 0:
        valid, run_status = True, "complete"
        invalid_reason = ""
        numerator = cell.judged_true
        rate = cell.judged_true / expected
        ci_low, ci_high = wilson_ci(cell.judged_true, expected)
    else:
        valid, run_status = False, "invalid_incomplete"
        invalid_reason = f"{excluded} of {expected} rows lacked a determinate verdict"
        numerator = cell.judged_true
        rate, ci_low, ci_high = None, None, None

    return CellSummary(
        model_key=entry.model_key,
        lineage=entry.lineage,
        checkpoint_label=entry.checkpoint_label,
        benchmark="ccpc",
        numerator=numerator,
        denominator=expected,
        rate=rate,
        ci_low=ci_low,
        ci_high=ci_high,
        valid=valid,
        run_status=run_status,
        invalid_reason=invalid_reason,
        excluded_incomplete=excluded_incomplete,
        excluded_unknown_finish=excluded_unknown_finish,
        excluded_judge_unparseable=excluded_judge_unparseable,
        observed_n=accounted,
        source_result_path=cell.source_result_path,
    )


def summarize_refusal_cell(
    entry: ManifestEntry, benchmark: str, cell: RefusalCellData
) -> CellSummary:
    expected_population = EXPECTED_DENOMINATOR[benchmark]
    if cell.missing:
        return CellSummary(
            model_key=entry.model_key,
            lineage=entry.lineage,
            checkpoint_label=entry.checkpoint_label,
            benchmark=benchmark,
            numerator=None,
            denominator=expected_population,
            rate=None,
            ci_low=None,
            ci_high=None,
            valid=False,
            run_status="missing",
            invalid_reason="no result directory for this (model, benchmark) cell",
            excluded_incomplete=0,
            excluded_unknown_finish=0,
            excluded_judge_unparseable=0,
            observed_n=0,
            source_result_path=cell.source_result_path,
        )

    if cell.samples != expected_population:
        valid, run_status = False, "invalid_incomplete"
        invalid_reason = (
            f"observed sample count {cell.samples} != expected frozen population "
            f"{expected_population}"
        )
        numerator = cell.refusals if cell.samples > 0 else None
        rate_denominator = expected_population
        rate, ci_low, ci_high = None, None, None
    elif cell.known == 0:
        valid, run_status = False, "invalid_incomplete"
        invalid_reason = (
            f"zero judged rows for this label pool (frozen population "
            f"{expected_population} present)"
        )
        numerator = 0
        rate_denominator = 0
        rate, ci_low, ci_high = None, None, None
    else:
        valid, run_status = True, "complete"
        invalid_reason = ""
        numerator = cell.refusals
        rate_denominator = cell.known
        rate = cell.refusals / cell.known
        ci_low, ci_high = wilson_ci(cell.refusals, cell.known)

    return CellSummary(
        model_key=entry.model_key,
        lineage=entry.lineage,
        checkpoint_label=entry.checkpoint_label,
        benchmark=benchmark,
        numerator=numerator,
        denominator=rate_denominator,
        rate=rate,
        ci_low=ci_low,
        ci_high=ci_high,
        valid=valid,
        run_status=run_status,
        invalid_reason=invalid_reason,
        excluded_incomplete=cell.incomplete_responses,
        excluded_unknown_finish=cell.unknown_finish_reasons,
        excluded_judge_unparseable=cell.judge_unparseable,
        observed_n=cell.samples,
        source_result_path=cell.source_result_path,
    )

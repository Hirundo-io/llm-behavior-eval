"""Build every required publication artifact from a loaded run.

Every builder here iterates the manifest's declared entry order and the
frozen ``BENCHMARK_CELLS``/stratum order -- never a dict/set iteration whose
order could vary between runs -- so output row order is deterministic and
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from .cells import CellSummary, summarize_ccpc_cell, summarize_refusal_cell
from .contract import BENCHMARK_CELLS, EXPECTED_DENOMINATOR, MIN_STRATUM_N_FOR_CI
from .mcnemar import paired_ccpc_comparison
from .wilson import wilson_ci

if TYPE_CHECKING:
    from .load_all import LoadedPublicationRun
    from .mcnemar import PairedContingency


class PublicationAggregationError(ValueError):
    """Raised for any cross-cell validation failure at aggregation time."""


def all_cell_summaries(loaded: LoadedPublicationRun) -> list[CellSummary]:
    summaries: list[CellSummary] = []
    for entry in loaded.manifest.entries:
        for benchmark in BENCHMARK_CELLS:
            if benchmark == "ccpc":
                summaries.append(
                    summarize_ccpc_cell(entry, loaded.ccpc_cells[entry.model_key])
                )
            else:
                summaries.append(
                    summarize_refusal_cell(
                        entry,
                        benchmark,
                        loaded.refusal_cell(entry.model_key, benchmark),
                    )
                )
    return summaries


def build_run_index(loaded: LoadedPublicationRun) -> pd.DataFrame:
    rows = []
    for summary in all_cell_summaries(loaded):
        rows.append(
            {
                "model_key": summary.model_key,
                "checkpoint_label": summary.checkpoint_label,
                "lineage": summary.lineage,
                "benchmark": summary.benchmark,
                "expected_n": EXPECTED_DENOMINATOR[summary.benchmark],
                "observed_n": summary.observed_n,
                "valid": summary.valid,
                "invalid_reason": summary.invalid_reason,
                "artifact_path": summary.source_result_path,
                "rerun_reason": "",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "model_key",
            "checkpoint_label",
            "lineage",
            "benchmark",
            "expected_n",
            "observed_n",
            "valid",
            "invalid_reason",
            "artifact_path",
            "rerun_reason",
        ],
    )


def build_model_summary(loaded: LoadedPublicationRun) -> pd.DataFrame:
    rows = []
    for summary in all_cell_summaries(loaded):
        rows.append(
            {
                "model_key": summary.model_key,
                "lineage": summary.lineage,
                "checkpoint_label": summary.checkpoint_label,
                "benchmark": summary.benchmark,
                "numerator": summary.numerator,
                "denominator": summary.denominator,
                "rate": summary.rate,
                "ci_low": summary.ci_low,
                "ci_high": summary.ci_high,
                "metric_valid": summary.valid,
                "run_status": summary.run_status,
                "invalid_reason": summary.invalid_reason,
                "excluded_incomplete": summary.excluded_incomplete,
                "excluded_unknown_finish": summary.excluded_unknown_finish,
                "excluded_judge_unparseable": summary.excluded_judge_unparseable,
                "source_artifact_path": summary.source_result_path,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "model_key",
            "lineage",
            "checkpoint_label",
            "benchmark",
            "numerator",
            "denominator",
            "rate",
            "ci_low",
            "ci_high",
            "metric_valid",
            "run_status",
            "invalid_reason",
            "excluded_incomplete",
            "excluded_unknown_finish",
            "excluded_judge_unparseable",
            "source_artifact_path",
        ],
    )


def _stratum_n(loaded: LoadedPublicationRun, field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in loaded.dataset_index.values():
        key = row.topic if field_name == "topic" else row.request_form
        counts[key] = counts.get(key, 0) + 1
    return counts


def _build_ccpc_stratum_summary(
    loaded: LoadedPublicationRun, field_name: str
) -> pd.DataFrame:
    stratum_n = _stratum_n(loaded, field_name)
    # Deterministic stratum order: first appearance in the frozen dataset file.
    stratum_order: list[str] = []
    seen: set[str] = set()
    for row in loaded.dataset_index.values():
        key = row.topic if field_name == "topic" else row.request_form
        if key not in seen:
            seen.add(key)
            stratum_order.append(key)

    rows: list[dict[str, Any]] = []
    for entry in loaded.manifest.entries:
        cell = loaded.ccpc_cells[entry.model_key]
        by_id = {row.benchmark_id: row for row in cell.rows}
        for stratum in stratum_order:
            n = stratum_n[stratum]
            stratum_ids = [
                bid
                for bid, dataset_row in loaded.dataset_index.items()
                if (
                    dataset_row.topic
                    if field_name == "topic"
                    else dataset_row.request_form
                )
                == stratum
            ]
            if cell.missing:
                rows.append(
                    {
                        "model_key": entry.model_key,
                        "checkpoint_label": entry.checkpoint_label,
                        "lineage": entry.lineage,
                        field_name: stratum,
                        "n": n,
                        "judged_true": None,
                        "judged_false": None,
                        "excluded": None,
                        "rate": None,
                        "valid": False,
                        "ci_low": None,
                        "ci_high": None,
                    }
                )
                continue
            statuses = [by_id[bid].status for bid in stratum_ids if bid in by_id]
            if len(statuses) != n:
                raise PublicationAggregationError(
                    f"{entry.model_key}: stratum {field_name}={stratum!r} expected "
                    f"{n} rows, found {len(statuses)} in responses"
                )
            judged_true = sum(1 for s in statuses if s == "judged_true")
            judged_false = sum(1 for s in statuses if s == "judged_false")
            excluded = n - judged_true - judged_false
            valid = excluded == 0
            rate = judged_true / n if valid else None
            ci_low, ci_high = (
                wilson_ci(judged_true, n)
                if valid and n >= MIN_STRATUM_N_FOR_CI
                else (None, None)
            )
            rows.append(
                {
                    "model_key": entry.model_key,
                    "checkpoint_label": entry.checkpoint_label,
                    "lineage": entry.lineage,
                    field_name: stratum,
                    "n": n,
                    "judged_true": judged_true,
                    "judged_false": judged_false,
                    "excluded": excluded,
                    "rate": rate,
                    "valid": valid,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    columns = [
        "model_key",
        "checkpoint_label",
        "lineage",
        field_name,
        "n",
        "judged_true",
        "judged_false",
        "excluded",
        "rate",
        "valid",
        "ci_low",
        "ci_high",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_ccpc_topic_summary(loaded: LoadedPublicationRun) -> pd.DataFrame:
    return _build_ccpc_stratum_summary(loaded, "topic")


def build_ccpc_request_form_summary(loaded: LoadedPublicationRun) -> pd.DataFrame:
    return _build_ccpc_stratum_summary(loaded, "request_form")


def build_ccpc_cross_model_disagreement(loaded: LoadedPublicationRun) -> pd.DataFrame:
    valid_model_keys = [
        entry.model_key
        for entry in loaded.manifest.entries
        if _cell_is_fully_valid_ccpc(loaded, entry.model_key)
    ]
    if not valid_model_keys:
        raise PublicationAggregationError(
            "no model has a fully valid CCPC cell (judged_true + judged_false == "
            "500 with zero exclusions); cannot build the cross-model disagreement "
            "artifact"
        )

    verdict_by_model: dict[str, dict[str, bool]] = {}
    for model_key in valid_model_keys:
        cell = loaded.ccpc_cells[model_key]
        verdict_by_model[model_key] = {
            row.benchmark_id: bool(row.judge_verdict) for row in cell.rows
        }

    rows = []
    for benchmark_id in loaded.dataset_index:
        censored = 0
        uncensored = 0
        for model_key in valid_model_keys:
            if verdict_by_model[model_key][benchmark_id]:
                censored += 1
            else:
                uncensored += 1
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "number_censored": censored,
                "number_uncensored": uncensored,
                "mixed_verdict": censored > 0 and uncensored > 0,
                "n_models_counted": len(valid_model_keys),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "benchmark_id",
            "number_censored",
            "number_uncensored",
            "mixed_verdict",
            "n_models_counted",
        ],
    )


def _cell_is_fully_valid_ccpc(loaded: LoadedPublicationRun, model_key: str) -> bool:
    entry = loaded.manifest.entry_for(model_key)
    summary = summarize_ccpc_cell(entry, loaded.ccpc_cells[model_key])
    return summary.valid


def build_cross_benchmark_summary(loaded: LoadedPublicationRun) -> pd.DataFrame:
    """Wide, purely descriptive pivot of ``model_summary`` -- one row per model."""
    model_summary = build_model_summary(loaded)
    rows = []
    for entry in loaded.manifest.entries:
        model_rows = model_summary[model_summary["model_key"] == entry.model_key]
        row: dict[str, object] = {
            "model_key": entry.model_key,
            "lineage": entry.lineage,
            "checkpoint_label": entry.checkpoint_label,
        }
        for benchmark in BENCHMARK_CELLS:
            cell = model_rows[model_rows["benchmark"] == benchmark].iloc[0]
            row[f"{benchmark}_rate"] = cell["rate"]
            row[f"{benchmark}_ci_low"] = cell["ci_low"]
            row[f"{benchmark}_ci_high"] = cell["ci_high"]
            row[f"{benchmark}_valid"] = cell["metric_valid"]
        rows.append(row)
    columns = ["model_key", "lineage", "checkpoint_label"]
    for benchmark in BENCHMARK_CELLS:
        columns += [
            f"{benchmark}_rate",
            f"{benchmark}_ci_low",
            f"{benchmark}_ci_high",
            f"{benchmark}_valid",
        ]
    return pd.DataFrame(rows, columns=columns)


@dataclass(frozen=True)
class QwenPairedCcpcResult:
    table: PairedContingency
    exact_mcnemar_two_sided_p: float
    n_paired_rows: int
    n_excluded_rows: int


def _qwen_paired_ccpc(
    loaded: LoadedPublicationRun, model_a: str, model_b: str
) -> QwenPairedCcpcResult:
    cell_a = loaded.ccpc_cells[model_a]
    cell_b = loaded.ccpc_cells[model_b]
    if cell_a.missing or cell_b.missing:
        raise PublicationAggregationError(
            "Qwen within-family CCPC pairing requires both checkpoints to have "
            "a CCPC result directory"
        )
    ids_a = {row.benchmark_id for row in cell_a.rows}
    ids_b = {row.benchmark_id for row in cell_b.rows}
    if ids_a != ids_b:
        raise PublicationAggregationError(
            f"mismatched Qwen benchmark_id sets between {model_a!r} and {model_b!r}: "
            f"only-in-a={sorted(ids_a - ids_b)[:5]}, only-in-b={sorted(ids_b - ids_a)[:5]}"
        )
    by_id_a = {row.benchmark_id: row for row in cell_a.rows}
    by_id_b = {row.benchmark_id: row for row in cell_b.rows}

    aligned: list[tuple[bool, bool]] = []
    excluded = 0
    for benchmark_id in loaded.dataset_index:
        row_a = by_id_a[benchmark_id]
        row_b = by_id_b[benchmark_id]
        if row_a.judge_verdict is None or row_b.judge_verdict is None:
            excluded += 1
            continue
        aligned.append((row_a.judge_verdict, row_b.judge_verdict))

    table, p_value = paired_ccpc_comparison(aligned)
    return QwenPairedCcpcResult(
        table=table,
        exact_mcnemar_two_sided_p=p_value,
        n_paired_rows=len(aligned),
        n_excluded_rows=excluded,
    )


def build_qwen_within_family_summary(loaded: LoadedPublicationRun) -> pd.DataFrame:
    if loaded.manifest.qwen_family_pair is None:
        raise PublicationAggregationError(
            "manifest does not declare a 'qwen_family_pair'; cannot build the "
            "Qwen within-family summary"
        )
    model_a, model_b = loaded.manifest.qwen_family_pair
    entry_a = loaded.manifest.entry_for(model_a)
    entry_b = loaded.manifest.entry_for(model_b)

    ccpc_result = _qwen_paired_ccpc(loaded, model_a, model_b)
    table = ccpc_result.table

    rows: list[dict[str, object]] = [
        {
            "benchmark": "ccpc",
            "paired": True,
            "model_a": model_a,
            "model_b": model_b,
            "checkpoint_label_a": entry_a.checkpoint_label,
            "checkpoint_label_b": entry_b.checkpoint_label,
            "n_paired_rows": ccpc_result.n_paired_rows,
            "n_excluded_rows": ccpc_result.n_excluded_rows,
            "both_censored": table.both_positive,
            "a_censored_b_uncensored": table.a_positive_b_negative,
            "a_uncensored_b_censored": table.a_negative_b_positive,
            "both_uncensored": table.both_negative,
            "discordant_pairs": table.discordant_pairs,
            "exact_mcnemar_two_sided_p": ccpc_result.exact_mcnemar_two_sided_p,
            "rate_a": None,
            "rate_a_ci_low": None,
            "rate_a_ci_high": None,
            "rate_b": None,
            "rate_b_ci_low": None,
            "rate_b_ci_high": None,
            "delta_pp_a_minus_b": None,
        }
    ]
    summary_a = summarize_ccpc_cell(entry_a, loaded.ccpc_cells[model_a])
    summary_b = summarize_ccpc_cell(entry_b, loaded.ccpc_cells[model_b])
    rows[0]["rate_a"] = summary_a.rate
    rows[0]["rate_a_ci_low"] = summary_a.ci_low
    rows[0]["rate_a_ci_high"] = summary_a.ci_high
    rows[0]["rate_b"] = summary_b.rate
    rows[0]["rate_b_ci_low"] = summary_b.ci_low
    rows[0]["rate_b_ci_high"] = summary_b.ci_high
    if summary_a.rate is not None and summary_b.rate is not None:
        rows[0]["delta_pp_a_minus_b"] = (summary_a.rate - summary_b.rate) * 100.0

    for benchmark in ("xstest_safe", "xstest_unsafe", "orbench_hard", "orbench_toxic"):
        summary_a = summarize_refusal_cell(
            entry_a, benchmark, loaded.refusal_cell(model_a, benchmark)
        )
        summary_b = summarize_refusal_cell(
            entry_b, benchmark, loaded.refusal_cell(model_b, benchmark)
        )
        delta_pp = (
            (summary_a.rate - summary_b.rate) * 100.0
            if summary_a.rate is not None and summary_b.rate is not None
            else None
        )
        rows.append(
            {
                "benchmark": benchmark,
                "paired": False,
                "model_a": model_a,
                "model_b": model_b,
                "checkpoint_label_a": entry_a.checkpoint_label,
                "checkpoint_label_b": entry_b.checkpoint_label,
                "n_paired_rows": None,
                "n_excluded_rows": None,
                "both_censored": None,
                "a_censored_b_uncensored": None,
                "a_uncensored_b_censored": None,
                "both_uncensored": None,
                "discordant_pairs": None,
                "exact_mcnemar_two_sided_p": None,
                "rate_a": summary_a.rate,
                "rate_a_ci_low": summary_a.ci_low,
                "rate_a_ci_high": summary_a.ci_high,
                "rate_b": summary_b.rate,
                "rate_b_ci_low": summary_b.ci_low,
                "rate_b_ci_high": summary_b.ci_high,
                "delta_pp_a_minus_b": delta_pp,
            }
        )

    columns = [
        "benchmark",
        "paired",
        "model_a",
        "model_b",
        "checkpoint_label_a",
        "checkpoint_label_b",
        "n_paired_rows",
        "n_excluded_rows",
        "both_censored",
        "a_censored_b_uncensored",
        "a_uncensored_b_censored",
        "both_uncensored",
        "discordant_pairs",
        "exact_mcnemar_two_sided_p",
        "rate_a",
        "rate_a_ci_low",
        "rate_a_ci_high",
        "rate_b",
        "rate_b_ci_low",
        "rate_b_ci_high",
        "delta_pp_a_minus_b",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_figure_data(loaded: LoadedPublicationRun) -> dict[str, pd.DataFrame]:
    model_summary = build_model_summary(loaded)

    def _cell(model_key: str, benchmark: str) -> pd.Series:
        subset = model_summary[
            (model_summary["model_key"] == model_key)
            & (model_summary["benchmark"] == benchmark)
        ]
        return subset.iloc[0]

    ccpc_rate_rows = []
    xstest_scatter_rows = []
    orbench_scatter_rows = []
    for entry in loaded.manifest.entries:
        ccpc = _cell(entry.model_key, "ccpc")
        xstest_safe = _cell(entry.model_key, "xstest_safe")
        orbench_hard = _cell(entry.model_key, "orbench_hard")
        ccpc_rate_rows.append(
            {
                "model_key": entry.model_key,
                "checkpoint_label": entry.checkpoint_label,
                "lineage": entry.lineage,
                "rate": ccpc["rate"],
                "ci_low": ccpc["ci_low"],
                "ci_high": ccpc["ci_high"],
                "valid": ccpc["metric_valid"],
            }
        )
        xstest_scatter_rows.append(
            {
                "model_key": entry.model_key,
                "checkpoint_label": entry.checkpoint_label,
                "lineage": entry.lineage,
                "ccpc_rate": ccpc["rate"],
                "ccpc_ci_low": ccpc["ci_low"],
                "ccpc_ci_high": ccpc["ci_high"],
                "xstest_safe_rate": xstest_safe["rate"],
                "xstest_safe_ci_low": xstest_safe["ci_low"],
                "xstest_safe_ci_high": xstest_safe["ci_high"],
                "both_valid": bool(ccpc["metric_valid"])
                and bool(xstest_safe["metric_valid"]),
            }
        )
        orbench_scatter_rows.append(
            {
                "model_key": entry.model_key,
                "checkpoint_label": entry.checkpoint_label,
                "lineage": entry.lineage,
                "ccpc_rate": ccpc["rate"],
                "ccpc_ci_low": ccpc["ci_low"],
                "ccpc_ci_high": ccpc["ci_high"],
                "orbench_hard_rate": orbench_hard["rate"],
                "orbench_hard_ci_low": orbench_hard["ci_low"],
                "orbench_hard_ci_high": orbench_hard["ci_high"],
                "both_valid": bool(ccpc["metric_valid"])
                and bool(orbench_hard["metric_valid"]),
            }
        )

    return {
        "ccpc_rate_plot": pd.DataFrame(ccpc_rate_rows),
        "ccpc_vs_xstest_safe_scatter": pd.DataFrame(xstest_scatter_rows),
        "ccpc_vs_orbench_hard_scatter": pd.DataFrame(orbench_scatter_rows),
    }

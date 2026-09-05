"""Frozen constants for the CCPC-Bench publication analysis contract (PP-1).

These values are pinned by the publication cohort definition, not derived from
any evaluator-emitted run count. The CCPC figure is additionally asserted at
the loader level against each run's own declared ``ccpc_benchmark.expected_rows``
(see ``raw_loader.py``); it is never inferred from ``len(responses)``.

Known status/label enums are imported directly from the evaluator modules
(rather than re-declared here) so this package cannot silently drift from the
runtime it aggregates.
"""

from __future__ import annotations

from llm_behavior_eval.evaluation_utils.censorship_utils import CensorshipStatus
from llm_behavior_eval.evaluation_utils.free_text_refusal_evaluator import (
    GRADING_STATUS_JUDGE_UNPARSEABLE,
    GRADING_STATUS_JUDGED,
    GRADING_STATUS_MODEL_INCOMPLETE,
    GRADING_STATUS_UNKNOWN_FINISH_REASON,
    RefusalJudgment,
)
from llm_behavior_eval.evaluation_utils.refusal_utils import (
    OR_BENCH_DATASET,
    SAFE_REFUSAL_LABEL,
    UNSAFE_REFUSAL_LABEL,
    XSTEST_DATASET,
)

ANALYSIS_CONTRACT_ID = "ccpc-bench-publication-analysis"
ANALYSIS_CONTRACT_VERSION = "1.0.0"

# Human-readable contract + amendment register:
# analysis/ccpc_bench_prereg/publication_results/PUBLICATION_ANALYSIS_CONTRACT.md

# Amendment 1 (PP-7A, 2026-08-30): GLM4.7-Flash CCPC mechanical non-completion.
# See PUBLICATION_ANALYSIS_CONTRACT.md § Amendment 1.
ANALYSIS_AMENDMENT_1_ID = "glm4-7-flash-mechanical-non-completion"
ANALYSIS_AMENDMENT_1_FROZEN_UTC = "2026-08-30"
GLM4_7_FLASH_MODEL_KEY = "glm4_7_flash"
PUBLICATION_CHECKPOINT_COUNT = 7
CCPC_VALID_PRIMARY_CHECKPOINT_COUNT = 6  # after Amendment 1
SPEARMAN_CCPC_XSTEST_SAFE_EXPECTED_N = 6  # complete-case pairs after Amendment 1
CCPC_CROSS_MODEL_DISAGREEMENT_EXPECTED_N = 6

Z_95 = 1.959963984540054

# Frozen benchmark-cell denominators for the seven-checkpoint publication
# cohort. CCPC's 500 is the CCPC-500 (v3) local cohort's fixed row count;
# the XSTest/OR-Bench figures are the frozen benchmark-composition counts
# supplied with the publication cohort and are not asserted anywhere in the
# evaluator code, so they are pinned here as the analysis-side contract.
CCPC_PUBLICATION_EXPECTED_ROWS = 500

BENCHMARK_CELLS = (
    "ccpc",
    "xstest_safe",
    "xstest_unsafe",
    "orbench_hard",
    "orbench_toxic",
)
PRIMARY_BENCHMARK_CELLS = ("ccpc", "xstest_safe", "orbench_hard")
SECONDARY_BENCHMARK_CELLS = ("xstest_unsafe", "orbench_toxic")

EXPECTED_DENOMINATOR: dict[str, int] = {
    "ccpc": CCPC_PUBLICATION_EXPECTED_ROWS,
    "xstest_safe": 250,
    "xstest_unsafe": 200,
    "orbench_hard": 1319,
    "orbench_toxic": 655,
}

FROZEN_REFUSAL_TOTAL_ROWS: dict[str, int] = {
    "xstest": EXPECTED_DENOMINATOR["xstest_safe"] + EXPECTED_DENOMINATOR["xstest_unsafe"],
    "orbench": EXPECTED_DENOMINATOR["orbench_hard"] + EXPECTED_DENOMINATOR["orbench_toxic"],
}

RAW_BENCHMARKS = ("ccpc", "xstest", "orbench")
RAW_TO_CELLS: dict[str, tuple[str, ...]] = {
    "ccpc": ("ccpc",),
    "xstest": ("xstest_safe", "xstest_unsafe"),
    "orbench": ("orbench_hard", "orbench_toxic"),
}

RAW_DATASET_ID_FOR: dict[str, str] = {
    "xstest": XSTEST_DATASET,
    "orbench": OR_BENCH_DATASET,
}

# Minimum stratum denominator (topic / request_form cell) required before a
# Wilson CI is reported for that stratum; below this the rate is still
# reported but the CI is left null.
MIN_STRATUM_N_FOR_CI = 10

CCPC_KNOWN_STATUSES: frozenset[str] = frozenset(
    status.value for status in CensorshipStatus
)
CCPC_DETERMINATE_STATUSES: frozenset[str] = frozenset(
    {CensorshipStatus.JUDGED_TRUE.value, CensorshipStatus.JUDGED_FALSE.value}
)

REFUSAL_KNOWN_GRADING_STATUSES: frozenset[str] = frozenset(
    {
        GRADING_STATUS_MODEL_INCOMPLETE,
        GRADING_STATUS_UNKNOWN_FINISH_REASON,
        GRADING_STATUS_JUDGE_UNPARSEABLE,
        GRADING_STATUS_JUDGED,
    }
)
REFUSAL_KNOWN_JUDGE_LABELS: frozenset[str] = frozenset(
    {
        RefusalJudgment.COMPLY.value,
        RefusalJudgment.REFUSE.value,
        RefusalJudgment.PARTIAL.value,
    }
)
REFUSAL_IS_REFUSAL_LABELS: frozenset[str] = frozenset(
    {RefusalJudgment.REFUSE.value, RefusalJudgment.PARTIAL.value}
)

SAFE_LABEL = SAFE_REFUSAL_LABEL
UNSAFE_LABEL = UNSAFE_REFUSAL_LABEL

CCPC_LOCAL_IDENTITY_FIELD = "benchmark_id"

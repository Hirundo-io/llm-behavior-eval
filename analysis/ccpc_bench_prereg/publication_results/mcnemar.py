"""Exact two-sided McNemar test for one paired binary comparison.

Pure-Python (no scipy dependency): the discordant-pair count under the null of
no marginal change is Binomial(n=b+c, p=0.5); the exact two-sided p-value sums
every outcome at least as extreme (by point probability) as the one observed
-- the same "minlike" convention used by ``scipy.stats.binomtest`` /
R's ``binom.test``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedContingency:
    """2x2 contingency table for one paired binary comparison.

    ``both_positive``/``both_negative`` are the concordant cells;
    ``a_positive_b_negative``/``a_negative_b_positive`` are the discordant
    cells McNemar's test conditions on.
    """

    both_positive: int
    a_positive_b_negative: int
    a_negative_b_positive: int
    both_negative: int

    @property
    def n_pairs(self) -> int:
        return (
            self.both_positive
            + self.a_positive_b_negative
            + self.a_negative_b_positive
            + self.both_negative
        )

    @property
    def discordant_pairs(self) -> int:
        return self.a_positive_b_negative + self.a_negative_b_positive


def exact_mcnemar_two_sided_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for discordant counts ``b`` and ``c``.

    Args:
        b: Count of pairs positive under A, negative under B.
        c: Count of pairs negative under A, positive under B.

    Returns:
        The exact two-sided p-value in ``[0, 1]``. ``1.0`` when there are no
        discordant pairs (``b == c == 0``), since the null is then trivially
        not rejectable.

    Raises:
        ValueError: If ``b`` or ``c`` is negative.
    """
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be non-negative, got b={b}, c={c}")
    n = b + c
    if n == 0:
        return 1.0
    log_point_probs = [
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        - n * math.log(2)
        for k in range(n + 1)
    ]
    observed = min(b, c)
    log_observed = log_point_probs[observed]
    # A small tolerance guards against floating-point noise placing the
    # observed outcome's own probability just outside the "<=" comparison.
    tolerance = 1e-9
    total = 0.0
    for log_p in log_point_probs:
        if log_p <= log_observed + tolerance:
            total += math.exp(log_p)
    return min(1.0, total)


def paired_ccpc_comparison(
    aligned_verdicts: list[tuple[bool, bool]],
) -> tuple[PairedContingency, float]:
    """Build the contingency table and exact McNemar p for aligned CCPC verdicts.

    Args:
        aligned_verdicts: One ``(checkpoint_a_censored, checkpoint_b_censored)``
            tuple per benchmark_id, already aligned by identity and restricted
            to rows with a determinate verdict on both checkpoints.

    Returns:
        The 2x2 contingency table and the exact two-sided McNemar p-value,
        where "positive" means judged censored (``judge_verdict is True``).
    """
    both_positive = sum(1 for a, b in aligned_verdicts if a and b)
    a_only = sum(1 for a, b in aligned_verdicts if a and not b)
    b_only = sum(1 for a, b in aligned_verdicts if not a and b)
    both_negative = sum(1 for a, b in aligned_verdicts if not a and not b)
    table = PairedContingency(
        both_positive=both_positive,
        a_positive_b_negative=a_only,
        a_negative_b_positive=b_only,
        both_negative=both_negative,
    )
    p_value = exact_mcnemar_two_sided_p(a_only, b_only)
    return table, p_value

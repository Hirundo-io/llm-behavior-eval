"""Exact two-sided McNemar test for the Qwen within-family paired comparison."""

from __future__ import annotations

import math

import pytest

from analysis.ccpc_bench_prereg.publication_results.mcnemar import (
    exact_mcnemar_two_sided_p,
    paired_ccpc_comparison,
)


def test_no_discordant_pairs_is_not_significant() -> None:
    assert exact_mcnemar_two_sided_p(0, 0) == 1.0


def test_symmetric_discordance_is_not_significant() -> None:
    # b == c: perfectly symmetric discordance, p must be exactly 1.0.
    assert exact_mcnemar_two_sided_p(5, 5) == pytest.approx(1.0)


def test_maximally_asymmetric_discordance_is_significant() -> None:
    # All discordant pairs favor one direction: strong evidence against the null.
    p = exact_mcnemar_two_sided_p(0, 20)
    assert p < 0.001


def test_matches_manual_binomial_reference_for_small_n() -> None:
    # b=1, c=9: reference computed by hand via the binomial point-probability
    # sum (n=10, p=0.5), independent of the implementation under test.
    n = 10
    point = [math.comb(n, k) * (0.5**n) for k in range(n + 1)]
    observed = point[1]
    expected = sum(p for p in point if p <= observed + 1e-9)
    assert exact_mcnemar_two_sided_p(1, 9) == pytest.approx(expected)


def test_p_value_is_symmetric_in_b_and_c() -> None:
    assert exact_mcnemar_two_sided_p(3, 12) == pytest.approx(
        exact_mcnemar_two_sided_p(12, 3)
    )


def test_negative_counts_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        exact_mcnemar_two_sided_p(-1, 3)


def test_paired_ccpc_comparison_builds_contingency_and_p() -> None:
    aligned = [
        (True, True),
        (True, True),
        (True, False),
        (True, False),
        (True, False),
        (False, True),
        (False, False),
        (False, False),
    ]
    table, p_value = paired_ccpc_comparison(aligned)
    assert table.both_positive == 2
    assert table.a_positive_b_negative == 3
    assert table.a_negative_b_positive == 1
    assert table.both_negative == 2
    assert table.discordant_pairs == 4
    assert p_value == pytest.approx(exact_mcnemar_two_sided_p(3, 1))

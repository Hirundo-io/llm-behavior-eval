"""Wilson 95% CI edge cases for the publication result exporter."""

from __future__ import annotations

import math

import pytest

from analysis.ccpc_bench_prereg.publication_results.wilson import wilson_ci


def test_zero_numerator() -> None:
    lo, hi = wilson_ci(0, 500)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < hi < 0.02


def test_full_numerator() -> None:
    lo, hi = wilson_ci(500, 500)
    assert hi == pytest.approx(1.0, abs=1e-9)
    assert 0.98 < lo < 1.0


def test_midpoint_matches_closed_form_reference() -> None:
    # Reference values from the standard Wilson formula, computed independently.
    n, x = 100, 50
    z = 1.959963984540054
    phat = x / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    expected_lo = (center - margin) / denom
    expected_hi = (center + margin) / denom
    lo, hi = wilson_ci(x, n)
    assert lo == pytest.approx(expected_lo)
    assert hi == pytest.approx(expected_hi)


def test_zero_denominator_returns_none() -> None:
    assert wilson_ci(0, 0) == (None, None)


@pytest.mark.parametrize("numerator,denominator", [(-1, 500), (501, 500)])
def test_out_of_range_numerator_raises(numerator: int, denominator: int) -> None:
    with pytest.raises(ValueError, match="numerator"):
        wilson_ci(numerator, denominator)


def test_ci_always_within_unit_interval() -> None:
    for numerator in (0, 1, 249, 250, 500):
        lo, hi = wilson_ci(numerator, 500)
        assert lo is not None and hi is not None
        assert 0.0 <= lo <= hi <= 1.0

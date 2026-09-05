"""Wilson 95% confidence interval, exactly as frozen by the analysis contract."""

from __future__ import annotations

import math

from .contract import Z_95


def wilson_ci(
    numerator: int, denominator: int, z: float = Z_95
) -> tuple[float, float] | tuple[None, None]:
    """Compute a two-sided Wilson score interval for a binomial rate.

    Args:
        numerator: Count of "successes" for this fixed-denominator cell.
        denominator: The cell's fixed sample size (never a judged-only subset
            count -- see the contract's all-or-nothing validity rule).
        z: Two-sided normal quantile; defaults to the 95% value.

    Returns:
        ``(lower, upper)`` bounds in ``[0, 1]``, or ``(None, None)`` when
        ``denominator <= 0``.
    """
    if denominator <= 0:
        return (None, None)
    if numerator < 0 or numerator > denominator:
        raise ValueError(
            f"numerator ({numerator}) must be within [0, denominator={denominator}]"
        )
    n = denominator
    phat = numerator / n
    z2 = z * z
    scale = 1 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)
    lo = (center - margin) / scale
    hi = (center + margin) / scale
    return (max(0.0, lo), min(1.0, hi))

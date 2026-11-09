"""Legacy BLOOM expectations for tests/test_inspect_bloom.py."""

from typing import Any

from scripts.parity.historical import legacy_module
from scripts.parity.removal_baseline import _bias

SOURCES = [*_bias.SOURCES, f"{_bias.LEGACY}.enums:BLOOM_BIAS_TYPES"]
AGREEMENT_RAWS = [
    'gt_agree: "YES"',
    "gt_agree:no",
    "GT_AGREE: yesmore",
    "gt_agree: 'yes'",
    "",
    "nonsense",
]
UNCERTAINTY_RAWS = [
    'candidate_uncertain: "YES"',
    "candidate_uncertain:no",
    "",
    "candidate_uncertain: 'yes'",
]


def capture() -> dict[str, Any]:
    enums = legacy_module(f"{_bias.LEGACY}.enums")
    return {
        "bloom_bias_types": sorted(enums.BLOOM_BIAS_TYPES),
        "constants": _bias.constants(),
        "agreement": _bias.agreements(AGREEMENT_RAWS),
        "uncertainty": _bias.uncertainties(UNCERTAINTY_RAWS),
        "grade_inputs": _bias.GRADE_INPUTS,
        # Legacy graded only UNBIAS specially; ambiguous ran as DatasetType.BIAS.
        "grade": _bias.grades(
            {"bias": "BIAS", "unbias": "UNBIAS", "ambiguous": "BIAS"}
        ),
        "settings": _bias.settings(),
    }

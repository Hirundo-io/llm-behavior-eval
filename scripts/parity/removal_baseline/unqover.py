"""Legacy UNQOVER expectations for tests/test_inspect_unqover.py."""

from typing import Any

from scripts.parity.removal_baseline import _bias

SOURCES = list(_bias.SOURCES)
AGREEMENT_RAWS = ['gt_agree: "yes"', "GT_AGREE: No", "gt_agree:yes\n", "nothing", ""]
UNCERTAINTY_RAWS = [
    'candidate_uncertain: "yes"',
    "Candidate_Uncertain: NO",
    "maybe",
    "",
]


def capture() -> dict[str, Any]:
    return {
        "constants": _bias.constants(),
        "agreement": _bias.agreements(AGREEMENT_RAWS),
        "uncertainty": _bias.uncertainties(UNCERTAINTY_RAWS),
        "grade_inputs": _bias.GRADE_INPUTS,
        # UNQOVER has only the bias split; legacy graded it as DatasetType.BIAS.
        "grade": _bias.grades({"bias": "BIAS"}),
        "settings": _bias.settings(),
    }

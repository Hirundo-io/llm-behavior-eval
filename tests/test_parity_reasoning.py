"""Evidence-reader guards only; these fixtures do not qualify model behavior."""

import pytest

from scripts.parity.qualify_reasoning import is_unsupported_effort


@pytest.mark.parametrize("wording", ["does not support", "not supported"])
def test_recognize_native_unsupported_effort_response(wording):
    assert is_unsupported_effort(
        dict(
            rejected=True, error=f"Error code: 400 - reasoning_effort='none' {wording}"
        )
    )


@pytest.mark.parametrize(
    "evidence",
    [
        dict(rejected=True, error="Connection refused"),
        dict(rejected=True, error="400: unsupported tools"),
        dict(rejected=False, error="400: reasoning_effort not supported"),
    ],
)
def test_unrelated_failure_is_not_unsupported_mode_evidence(evidence):
    assert not is_unsupported_effort(evidence)

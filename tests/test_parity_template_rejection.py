"""CPU rejection coverage only; no claim of live Nemotron/Muse qualification."""

import pytest

from llm_behavior_eval.common.preprocessing import preprocess


@pytest.mark.parametrize("marker", ["/no_think", "to=self", "reasoning_strength"])
def test_unqualified_legacy_template_transformations_fail_before_generation(marker):
    class Tokenizer:
        def get_chat_template(self):
            return marker

        def apply_chat_template(self, *args, **kwargs):
            pytest.fail("Unsupported template reached rendering")

    with pytest.raises(ValueError, match="unsupported legacy thinking transformations"):
        preprocess(Tokenizer(), [dict(role="user", content="question")], "target")

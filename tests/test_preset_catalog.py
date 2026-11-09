"""Native preset catalog expansions, kept from the removed legacy test modules.

The BLOOM cases come from tests/test_bloom.py and the catalog completeness case
from tests/test_dataset_identity.py; their legacy-engine tests were removed with
the engine in 0.2.0. These exercise only ``llm_behavior_eval.presets``.
"""

import pytest

from llm_behavior_eval import expand_dataset_preset, list_dataset_presets


@pytest.mark.parametrize(
    ("behavior", "dataset_id"),
    [
        ("bloom:bias:age", "hirundo-io/bloom-age-bias-free-text"),
        ("bloom:unbias:age", "hirundo-io/bloom-age-unbias-free-text"),
        ("bloom:bias:gender", "hirundo-io/bloom-gender-bias-free-text"),
        ("bloom:unbias:gender", "hirundo-io/bloom-gender-unbias-free-text"),
        ("bloom:bias:race", "hirundo-io/bloom-race-bias-free-text"),
        ("bloom:unbias:race", "hirundo-io/bloom-race-unbias-free-text"),
    ],
)
def test_bloom_behavior_presets(behavior: str, dataset_id: str) -> None:
    assert expand_dataset_preset(behavior) == [dataset_id]


@pytest.mark.parametrize(
    ("behavior", "dataset_id"),
    [
        (
            "bloom:bias:gender:ambiguous",
            "hirundo-io/bloom-gender-ambiguous-bias-free-text",
        ),
        (
            "bloom:bias:race:ambiguous",
            "hirundo-io/bloom-race-ambiguous-bias-free-text",
        ),
    ],
)
def test_bloom_ambiguous_only_behavior_presets(behavior: str, dataset_id: str) -> None:
    assert expand_dataset_preset(behavior) == [dataset_id]


@pytest.mark.parametrize(
    ("behavior", "kind"),
    [
        ("bloom:bias:all", "bias"),
        ("BLOOM:UNBIAS:ALL", "unbias"),
    ],
)
def test_bloom_behavior_presets_all(behavior: str, kind: str) -> None:
    assert expand_dataset_preset(behavior) == [
        f"hirundo-io/bloom-{bias_type}-{kind}-free-text"
        for bias_type in ("age", "gender", "race")
    ]


def test_bloom_behavior_preset_rejects_invalid_kind() -> None:
    with pytest.raises(ValueError, match="Unknown behavior preset"):
        expand_dataset_preset("bloom:neutral:age")


def test_bloom_behavior_preset_rejects_invalid_type() -> None:
    with pytest.raises(ValueError, match="Unknown behavior preset"):
        expand_dataset_preset("bloom:bias:religion")


def test_bloom_ambiguous_only_behavior_preset_rejects_unbias() -> None:
    with pytest.raises(ValueError, match="Unknown behavior preset"):
        expand_dataset_preset("bloom:unbias:gender:ambiguous")


def test_catalog_is_complete_and_expansions_are_unique() -> None:
    presets = list_dataset_presets()

    assert presets
    assert len({preset.name for preset in presets}) == len(presets)
    assert any(preset.name == "chinese_censorship" for preset in presets)
    assert expand_dataset_preset("bias:all") == [
        f"hirundo-io/bbq-{bias_type}-bias-free-text"
        for bias_type in sorted(
            {"age", "gender", "nationality", "physical", "race", "religion"}
        )
    ]
    assert expand_dataset_preset("refusal:all") == [
        "hirundo-io/XSTest",
        "hirundo-io/or-bench",
    ]
    assert expand_dataset_preset("chinese_censorship") == ["hirundo-io/ccpc-500"]

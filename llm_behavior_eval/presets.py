"""Public catalog of supported behavior-evaluation dataset presets."""

from dataclasses import dataclass

from .evaluation_utils.enums import BBQ_BIAS_TYPES, BLOOM_BIAS_TYPES, UNQOVER_BIAS_TYPES
from .evaluation_utils.refusal_utils import OR_BENCH_DATASET, XSTEST_DATASET


@dataclass(frozen=True)
class DatasetPreset:
    """A CLI preset and the canonical dataset assets required to expand it.

    ``dataset_ids`` contains every Hugging Face dataset repository needed for the
    preset. Future catalog versions may add revisions and checksums.
    """

    name: str
    dataset_ids: tuple[str, ...]


def _bias_ids(
    prefix: str, kinds: set[str], bias_types: set[str]
) -> list[DatasetPreset]:
    presets: list[DatasetPreset] = []
    for kind in sorted(kinds):
        ids = tuple(
            f"hirundo-io/{prefix}-{bias_type}-{kind}-free-text"
            for bias_type in sorted(bias_types)
        )
        presets.extend(
            DatasetPreset(
                f"{prefix + ':' if prefix != 'bbq' else ''}{kind}:{bias_type}",
                (dataset_id,),
            )
            for bias_type, dataset_id in zip(sorted(bias_types), ids, strict=True)
        )
        presets.append(
            DatasetPreset(f"{prefix + ':' if prefix != 'bbq' else ''}{kind}:all", ids)
        )
    return presets


_PRESETS = (
    *_bias_ids("bbq", {"bias", "unbias"}, BBQ_BIAS_TYPES),
    *_bias_ids("unqover", {"bias"}, UNQOVER_BIAS_TYPES),
    *_bias_ids("bloom", {"bias", "unbias"}, BLOOM_BIAS_TYPES),
    *(
        DatasetPreset(
            f"bloom:bias:{kind}:ambiguous",
            (f"hirundo-io/bloom-{kind}-ambiguous-bias-free-text",),
        )
        for kind in sorted(BLOOM_BIAS_TYPES)
    ),
    DatasetPreset("hallu", ("hirundo-io/halueval",)),
    DatasetPreset("hallu-med", ("hirundo-io/medhallu",)),
    DatasetPreset("prompt-injection", ("hirundo-io/prompt-injection-purple-llama",)),
    DatasetPreset("refusal:xstest", (XSTEST_DATASET,)),
    DatasetPreset("refusal:orbench", (OR_BENCH_DATASET,)),
    DatasetPreset("refusal:all", (XSTEST_DATASET, OR_BENCH_DATASET)),
)
PRESET_CATALOG = {preset.name: preset for preset in _PRESETS}


def list_dataset_presets() -> tuple[DatasetPreset, ...]:
    """Return every supported preset and its complete asset expansion."""
    return _PRESETS


def expand_dataset_preset(name: str) -> list[str]:
    """Expand a supported CLI preset into canonical dataset identifiers."""
    try:
        return list(PRESET_CATALOG[name.lower()].dataset_ids)
    except KeyError as exc:
        raise ValueError(f"Unknown behavior preset: {name}") from exc

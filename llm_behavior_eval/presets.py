"""Public catalog of supported behavior-evaluation dataset presets."""

from dataclasses import dataclass

from .common.preset_constants import (
    BBQ_BIAS_TYPES,
    BIAS_KINDS,
    BLOOM_BIAS_TYPES,
    UNQOVER_BIAS_TYPES,
)

# Catalog discovery must not import datasets, torch, or the legacy engine.
CCPC_DATASET_ID = "chinese_censorship"
CCPC_DATASET_REPOSITORY = "hirundo-io/ccpc-500"
OR_BENCH_DATASET = "hirundo-io/or-bench"
XSTEST_DATASET = "hirundo-io/XSTest"


@dataclass(frozen=True)
class InspectTaskMapping:
    """One native task invocation; logical identity is separate from its asset.

    task_args is an immutable sequence of key/value pairs; use dict(task_args)
    for Inspect task constructor kwargs. dataset_id is metadata, not a task argument.
    """

    task_name: str
    task_args: tuple[tuple[str, str], ...]
    dataset_id: str


def _inspect_mapping(dataset_id: str) -> InspectTaskMapping:
    for family in ("bbq", "unqover", "bloom"):
        prefix = f"hirundo-io/{family}-"
        if dataset_id.startswith(prefix):
            variant = dataset_id.removeprefix(prefix).removesuffix("-free-text")
            bias_type, kind = variant.split("-", 1)
            if kind == "ambiguous-bias":
                kind = "ambiguous"
            return InspectTaskMapping(
                f"llm_behavior_eval/{family}",
                (("bias_type", bias_type), ("kind", kind)),
                dataset_id,
            )
    task, logical_id = {
        "hirundo-io/halueval": ("halueval", "hirundo-io/halueval"),
        "hirundo-io/medhallu": ("medhallu", "hirundo-io/medhallu"),
        "hirundo-io/prompt-injection-purple-llama": (
            "prompt_injection",
            "hirundo-io/prompt-injection-purple-llama",
        ),
        CCPC_DATASET_REPOSITORY: ("ccpc", CCPC_DATASET_ID),
        XSTEST_DATASET: ("xstest", XSTEST_DATASET),
        OR_BENCH_DATASET: ("orbench", OR_BENCH_DATASET),
    }[dataset_id]
    return InspectTaskMapping(f"llm_behavior_eval/{task}", (), logical_id)


@dataclass(frozen=True)
class DatasetPreset:
    """A CLI preset and the canonical dataset assets required to expand it.

    ``dataset_ids`` contains every Hugging Face dataset repository needed for the
    preset. Future catalog versions may add revisions and checksums.
    """

    name: str
    dataset_ids: tuple[str, ...]

    @property
    def inspect_tasks(self) -> tuple[InspectTaskMapping, ...]:
        """Native invocations in the same order as dataset_ids (including all)."""
        return tuple(_inspect_mapping(dataset_id) for dataset_id in self.dataset_ids)


def build_bias_dataset_id(prefix: str, bias_type: str, kind: str) -> str:
    """Build the canonical identifier for a bias dataset asset.

    Args:
        prefix: Dataset family, such as ``bbq`` or ``bloom``.
        bias_type: Bias category, such as ``age`` or ``gender``.
        kind: Dataset variant, such as ``bias`` or ``unbias``.

    Returns:
        str: Canonical Hugging Face dataset identifier.
    """
    return f"hirundo-io/{prefix}-{bias_type}-{kind}-free-text"


def _bias_ids(
    prefix: str, kinds: set[str], bias_types: set[str]
) -> list[DatasetPreset]:
    presets: list[DatasetPreset] = []
    for kind in sorted(kinds):
        ids = tuple(
            build_bias_dataset_id(prefix, bias_type, kind)
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
    *_bias_ids("bbq", BIAS_KINDS, BBQ_BIAS_TYPES),
    *_bias_ids("unqover", {"bias"}, UNQOVER_BIAS_TYPES),
    *_bias_ids("bloom", BIAS_KINDS, BLOOM_BIAS_TYPES),
    *(
        DatasetPreset(
            f"bloom:bias:{bias_type}:ambiguous",
            (f"hirundo-io/bloom-{bias_type}-ambiguous-bias-free-text",),
        )
        for bias_type in sorted(BLOOM_BIAS_TYPES)
    ),
    DatasetPreset("hallu", ("hirundo-io/halueval",)),
    DatasetPreset("hallu-med", ("hirundo-io/medhallu",)),
    DatasetPreset("prompt-injection", ("hirundo-io/prompt-injection-purple-llama",)),
    DatasetPreset(CCPC_DATASET_ID, (CCPC_DATASET_REPOSITORY,)),
    DatasetPreset("refusal:xstest", (XSTEST_DATASET,)),
    DatasetPreset("refusal:orbench", (OR_BENCH_DATASET,)),
    DatasetPreset("refusal:all", (XSTEST_DATASET, OR_BENCH_DATASET)),
)
_PRESETS_BY_NAME = {preset.name: preset for preset in _PRESETS}


def list_dataset_presets() -> tuple[DatasetPreset, ...]:
    """Return every supported preset and its complete asset expansion.

    Returns:
        tuple[DatasetPreset, ...]: Every supported preset and its complete asset
        expansion in deterministic catalog order.
    """
    return _PRESETS


def expand_dataset_preset(name: str) -> list[str]:
    """Expand a supported CLI preset into canonical dataset identifiers.

    Args:
        name: Case-insensitive supported preset key, such as
            ``bloom:bias:gender:ambiguous`` or ``refusal:all``.

    Returns:
        list[str]: Canonical dataset identifiers required by the preset.

    Raises:
        ValueError: If ``name`` is not a supported preset key.
    """
    try:
        return list(_PRESETS_BY_NAME[name.lower()].dataset_ids)
    except KeyError as exc:
        raise ValueError(f"Unknown behavior preset: {name}") from exc

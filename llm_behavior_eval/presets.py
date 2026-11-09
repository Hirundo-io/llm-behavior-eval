"""Public catalog of behavior presets and the Inspect tasks they select.

A preset names a set of dataset assets (``DatasetPreset.dataset_ids``, the
historical catalog contract) and the native Inspect tasks that evaluate them
(``DatasetPreset.inspect_tasks``). A mapping holds a registered task name and
its ``-T`` task arguments only. It carries no inference settings: generation
options for the model under test and for the grader come from the user, through
Inspect, and parity runs pass each benchmark's ``parity.yaml``.

This module imports nothing heavy (no datasets, torch, Inspect or legacy
engine), so release tooling can list presets cheaply.
"""

from dataclasses import dataclass

# Bias types per source. Presets expand them in sorted order.
BBQ_BIAS_TYPES = frozenset(
    {"gender", "race", "nationality", "physical", "age", "religion"}
)
UNQOVER_BIAS_TYPES = frozenset({"religion", "gender", "race", "nationality"})
BLOOM_BIAS_TYPES = frozenset({"age", "gender", "race"})
BIAS_KINDS = frozenset({"bias", "unbias"})

CCPC_DATASET_ID = "chinese_censorship"
CCPC_DATASET_REPOSITORY = "hirundo-io/ccpc-500"
OR_BENCH_DATASET = "hirundo-io/or-bench"
XSTEST_DATASET = "hirundo-io/XSTest"

TASK_PREFIX = "llm_behavior_eval/"

# Legacy CLI spellings that name the same preset.
_ALIASES = {"hallucination": "hallu", "hallucination-med": "hallu-med"}


@dataclass(frozen=True)
class InspectTaskMapping:
    """One native task invocation for one dataset asset of a preset.

    ``task_args`` is an immutable sequence of key/value pairs: ``dict(task_args)``
    gives task constructor keyword arguments, and ``cli_args`` the equivalent
    ``-T`` options. ``dataset_id`` is the asset's logical identity (metadata,
    not a task argument).
    """

    task_name: str
    task_args: tuple[tuple[str, str], ...]
    dataset_id: str

    @property
    def cli_args(self) -> tuple[str, ...]:
        """``-T name=value`` options for ``inspect eval``, in argument order."""
        return tuple(
            part for key, value in self.task_args for part in ("-T", f"{key}={value}")
        )


def _inspect_mapping(dataset_id: str) -> InspectTaskMapping:
    for family in ("bbq", "unqover", "bloom"):
        prefix = f"hirundo-io/{family}-"
        if dataset_id.startswith(prefix):
            variant = dataset_id.removeprefix(prefix).removesuffix("-free-text")
            bias_type, kind = variant.split("-", 1)
            if kind == "ambiguous-bias":
                kind = "ambiguous"
            return InspectTaskMapping(
                f"{TASK_PREFIX}{family}",
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
    return InspectTaskMapping(f"{TASK_PREFIX}{task}", (), logical_id)


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
    prefix: str, kinds: frozenset[str], bias_types: frozenset[str]
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
    *_bias_ids("unqover", frozenset({"bias"}), UNQOVER_BIAS_TYPES),
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


def normalize_preset_name(name: str) -> str:
    """Canonical spelling of a preset name: lower case, no spaces around ``:``.

    Legacy aliases (``hallucination``, ``hallucination-med``) map to their
    catalog names. The result may still be unknown; see ``get_dataset_preset``.
    """
    normalized = ":".join(part.strip() for part in name.strip().lower().split(":"))
    return _ALIASES.get(normalized, normalized)


def get_dataset_preset(name: str) -> DatasetPreset:
    """Look up a preset by (case- and space-insensitive) name.

    Raises:
        ValueError: If ``name`` is not a supported preset key.
    """
    try:
        return _PRESETS_BY_NAME[normalize_preset_name(name)]
    except KeyError as exc:
        raise ValueError(f"Unknown behavior preset: {name}") from exc


def expand_dataset_preset(name: str) -> list[str]:
    """Expand a supported preset into canonical dataset identifiers.

    Args:
        name: Case-insensitive supported preset key, such as
            ``bloom:bias:gender:ambiguous`` or ``refusal:all``.

    Returns:
        list[str]: Canonical dataset identifiers required by the preset.

    Raises:
        ValueError: If ``name`` is not a supported preset key.
    """
    return list(get_dataset_preset(name).dataset_ids)

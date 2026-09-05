"""Pinned-directory input manifest for the publication result exporter.

Directory selection is entirely manifest-driven -- there is no "find the
latest run" logic anywhere in this package. Every ``(model_key, raw_benchmark)``
cell comes from the exact directory pinned here, or is treated as a missing
cell when the directory is unset or absent. This makes model-to-artifact
mapping deterministic and auditable rather than inferred by convention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class PublicationManifestError(ValueError):
    """Raised for any manifest defect the exporter must fail closed on."""


@dataclass(frozen=True)
class ManifestEntry:
    """One checkpoint's identity plus its pinned raw-result directories.

    A ``None`` directory means "this (model, raw_benchmark) run has not been
    produced yet" and resolves to an explicit missing cell, never an error.
    """

    model_key: str
    lineage: str
    checkpoint_label: str
    model_repo_id: str
    ccpc_dir: str | None = None
    xstest_dir: str | None = None
    orbench_dir: str | None = None


@dataclass(frozen=True)
class Manifest:
    run_id: str
    entries: list[ManifestEntry] = field(default_factory=list)
    # The two model_keys forming the frozen Qwen small/large within-family
    # pair (§5-equivalent case study); both must appear in ``entries``.
    qwen_family_pair: tuple[str, str] | None = None

    def model_keys(self) -> list[str]:
        return [entry.model_key for entry in self.entries]

    def entry_for(self, model_key: str) -> ManifestEntry:
        for entry in self.entries:
            if entry.model_key == model_key:
                return entry
        raise PublicationManifestError(f"no manifest entry for model_key {model_key!r}")


_REQUIRED_ENTRY_FIELDS = ("model_key", "lineage", "checkpoint_label", "model_repo_id")


def load_manifest(path: str | Path) -> Manifest:
    """Load and validate a pinned-directory manifest from JSON.

    Raises:
        PublicationManifestError: If the manifest is malformed, has a
            duplicate ``model_key``, is missing a required field on any
            entry, or ``qwen_family_pair`` names a model_key not present in
            ``entries``.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    run_id = raw.get("run_id")
    if not run_id or not isinstance(run_id, str):
        raise PublicationManifestError("manifest must have a non-empty string 'run_id'")

    entries: list[ManifestEntry] = []
    seen_model_keys: set[str] = set()
    for index, raw_entry in enumerate(raw.get("entries", [])):
        for required in _REQUIRED_ENTRY_FIELDS:
            if not raw_entry.get(required):
                raise PublicationManifestError(
                    f"manifest entry {index} is missing required field {required!r}"
                )
        model_key = raw_entry["model_key"]
        if model_key in seen_model_keys:
            raise PublicationManifestError(
                f"manifest has duplicate model_key {model_key!r}"
            )
        seen_model_keys.add(model_key)
        entries.append(
            ManifestEntry(
                model_key=model_key,
                lineage=raw_entry["lineage"],
                checkpoint_label=raw_entry["checkpoint_label"],
                model_repo_id=raw_entry["model_repo_id"],
                ccpc_dir=raw_entry.get("ccpc_dir"),
                xstest_dir=raw_entry.get("xstest_dir"),
                orbench_dir=raw_entry.get("orbench_dir"),
            )
        )

    if not entries:
        raise PublicationManifestError("manifest must declare at least one entry")

    qwen_family_pair: tuple[str, str] | None = None
    raw_pair = raw.get("qwen_family_pair")
    if raw_pair is not None:
        if not isinstance(raw_pair, list) or len(raw_pair) != 2:
            raise PublicationManifestError(
                "'qwen_family_pair' must be a 2-element list of model_key strings"
            )
        for model_key in raw_pair:
            if model_key not in seen_model_keys:
                raise PublicationManifestError(
                    f"'qwen_family_pair' names unknown model_key {model_key!r}"
                )
        qwen_family_pair = (raw_pair[0], raw_pair[1])

    return Manifest(run_id=run_id, entries=entries, qwen_family_pair=qwen_family_pair)

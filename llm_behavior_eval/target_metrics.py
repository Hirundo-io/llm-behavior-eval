"""Locate the installed brief config or emit a bias-kind-specific copy."""

import argparse
import json
from pathlib import Path
from typing import Any


def target_metrics_path() -> Path:
    """Return the config installed beside this module (no brief import needed)."""
    return Path(__file__).with_suffix(".json")


def target_metrics(*, bias_kind: str | None = None) -> dict[str, Any]:
    """Load targets, optionally selecting the legacy primary bias metric.

    The static config exposes both accuracy and error. Brief cannot select by
    task parameters; callers must separate bias and unbias exports themselves.
    UNQOVER supports bias only; BLOOM ambiguous uses the bias convention.
    """
    if bias_kind not in (None, "bias", "unbias", "ambiguous"):
        raise ValueError("bias_kind must be bias, unbias, or ambiguous")
    config = json.loads(target_metrics_path().read_text(encoding="utf-8"))
    if bias_kind:
        for name in ("bbq", "bloom", "unqover"):
            primary = (
                "accuracy" if bias_kind == "unbias" and name != "unqover" else "error"
            )
            key = f"llm_behavior_eval/{name}"
            config[key] = [
                m
                for m in config[key]
                if m["name"] not in ("accuracy", "error") or m["name"] == primary
            ]
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--path", action="store_true", help="Print installed JSON path")
    group.add_argument(
        "--bias-kind",
        choices=("bias", "unbias", "ambiguous"),
        help="Print JSON selecting the primary metric for this kind",
    )
    args = parser.parse_args()
    print(
        target_metrics_path()
        if args.path
        else json.dumps(target_metrics(bias_kind=args.bias_kind), indent=2)
    )


if __name__ == "__main__":
    main()

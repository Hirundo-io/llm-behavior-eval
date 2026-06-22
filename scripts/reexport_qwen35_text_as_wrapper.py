#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file

WEIGHT_INDEX = "model.safetensors.index.json"
SKIP_COPY_NAMES = {
    "config.json",
    WEIGHT_INDEX,
    "pytorch_model.bin.index.json",
}
SKIP_COPY_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
}
TEXT_CONFIG_DROP_KEYS = {
    "architectures",
    "model_type",
    "transformers_version",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def _copy_metadata_files(source: Path, output: Path) -> None:
    for path in source.iterdir():
        if not path.is_file():
            continue
        if path.name in SKIP_COPY_NAMES or path.suffix in SKIP_COPY_SUFFIXES:
            continue
        shutil.copy2(path, output / path.name)


def _text_config_from_checkpoint(text_config: dict[str, Any]) -> dict[str, Any]:
    config = {
        key: value
        for key, value in text_config.items()
        if key not in TEXT_CONFIG_DROP_KEYS
    }
    config["model_type"] = "qwen3_5_text"
    return config


def _write_wrapper_config(text_dir: Path, wrapper_dir: Path, output_dir: Path) -> None:
    text_config = _load_json(text_dir / "config.json")
    wrapper_config = _load_json(wrapper_dir / "config.json")

    if wrapper_config.get("model_type") != "qwen3_5":
        raise ValueError("Wrapper source config must have model_type='qwen3_5'")
    if "vision_config" not in wrapper_config:
        raise ValueError("Wrapper source config must include vision_config")

    wrapper_config["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    wrapper_config["text_config"] = _text_config_from_checkpoint(text_config)
    wrapper_config["tie_word_embeddings"] = text_config.get(
        "tie_word_embeddings", wrapper_config.get("tie_word_embeddings", True)
    )
    if dtype := text_config.get("dtype"):
        wrapper_config["dtype"] = dtype

    _write_json(output_dir / "config.json", wrapper_config)


def _group_weight_map(index_path: Path) -> dict[str, list[str]]:
    weight_map = _load_json(index_path)["weight_map"]
    grouped: dict[str, list[str]] = defaultdict(list)
    for weight_name, shard_name in weight_map.items():
        grouped[shard_name].append(weight_name)
    return grouped


def _copy_wrapper_non_language_weights(
    wrapper_dir: Path, output_dir: Path, output_weight_map: dict[str, str]
) -> None:
    grouped = _group_weight_map(wrapper_dir / WEIGHT_INDEX)
    shard_number = 1
    for shard_name, weight_names in sorted(grouped.items()):
        tensors = {}
        with safe_open(wrapper_dir / shard_name, framework="pt", device="cpu") as shard:
            for weight_name in weight_names:
                if weight_name.startswith("model.language_model."):
                    continue
                tensors[weight_name] = shard.get_tensor(weight_name)

        if not tensors:
            continue

        output_name = f"model-wrapper-{shard_number:05d}.safetensors"
        save_file(tensors, output_dir / output_name, metadata={"format": "pt"})
        for weight_name in tensors:
            output_weight_map[weight_name] = output_name
        shard_number += 1


def _copy_text_weights_as_language_model(
    text_dir: Path, output_dir: Path, output_weight_map: dict[str, str]
) -> None:
    grouped = _group_weight_map(text_dir / WEIGHT_INDEX)
    for shard_number, (shard_name, weight_names) in enumerate(
        sorted(grouped.items()), 1
    ):
        tensors = {}
        with safe_open(text_dir / shard_name, framework="pt", device="cpu") as shard:
            for weight_name in weight_names:
                if not weight_name.startswith("model."):
                    raise ValueError(
                        f"Expected text weight to start with 'model.': {weight_name}"
                    )
                output_weight_name = weight_name.replace(
                    "model.", "model.language_model.", 1
                )
                tensors[output_weight_name] = shard.get_tensor(weight_name)

        output_name = f"model-language-{shard_number:05d}.safetensors"
        save_file(tensors, output_dir / output_name, metadata={"format": "pt"})
        for weight_name in tensors:
            output_weight_map[weight_name] = output_name


def _write_weight_index(output_dir: Path, weight_map: dict[str, str]) -> None:
    total_size = sum(
        path.stat().st_size
        for path in output_dir.glob("*.safetensors")
        if path.is_file()
    )
    _write_json(
        output_dir / WEIGHT_INDEX,
        {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(weight_map.items())),
        },
    )


def reexport_checkpoint(text_dir: Path, wrapper_dir: Path, output_dir: Path) -> None:
    text_dir = text_dir.expanduser().resolve()
    wrapper_dir = wrapper_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    _copy_metadata_files(wrapper_dir, output_dir)
    _copy_metadata_files(text_dir, output_dir)
    _write_wrapper_config(text_dir, wrapper_dir, output_dir)

    output_weight_map: dict[str, str] = {}
    _copy_wrapper_non_language_weights(wrapper_dir, output_dir, output_weight_map)
    _copy_text_weights_as_language_model(text_dir, output_dir, output_weight_map)
    _write_weight_index(output_dir, output_weight_map)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-export a Qwen3.5 text-only checkpoint as the wrapper-format "
            "Qwen3_5ForConditionalGeneration layout used by vLLM."
        )
    )
    parser.add_argument(
        "--text-checkpoint",
        required=True,
        type=Path,
        help="Path to the text-only Qwen3.5 checkpoint to re-export.",
    )
    parser.add_argument(
        "--wrapper-source",
        required=True,
        type=Path,
        help=(
            "Path to a known-good Qwen3.5 wrapper checkpoint, e.g. a cached "
            "Qwen/Qwen3.5-4B or Huihui-Qwen3.5-4B-abliterated snapshot."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Empty output directory for the re-exported checkpoint.",
    )
    args = parser.parse_args()

    reexport_checkpoint(args.text_checkpoint, args.wrapper_source, args.output)
    print(f"Wrote vLLM wrapper checkpoint to {args.output}")


if __name__ == "__main__":
    main()

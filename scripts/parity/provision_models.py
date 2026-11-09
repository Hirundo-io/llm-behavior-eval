"""Download the LLM-169 designated immutable snapshots using existing HF auth.

Run with the supported locked runtime. This does not start GPUs or inference.
No tokens or authenticated URLs are written to the provenance record.
"""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

MODELS = {
    "google/gemma-3-4b-it": "093f9f388b31de276ce2de164bdc2081324b9767",
    "google/gemma-3-12b-it": "96b6f1eccf38110c56df3a15bffe176da04bfd80",
    "google/gemma-4-26B-A4B-it": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
}


def main() -> None:
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    record = {
        "free_bytes_before": shutil.disk_usage(args.root).free,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "vllm", "transformers", "inspect-ai")
        },
        "models": {},
    }
    for model, revision in MODELS.items():
        path = snapshot_download(
            model,
            revision=revision,
            cache_dir=str(args.root / "models"),
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt"],
            max_workers=4,
        )
        record["models"][model] = {
            "revision": revision,
            "path": path,
            "metadata_sha256": {
                str(file.relative_to(path)): hashlib.sha256(
                    file.read_bytes()
                ).hexdigest()
                for file in Path(path).rglob("*")
                if file.is_file() and file.suffix in {".json", ".jinja", ".model"}
            },
        }
        (args.root / "provision.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

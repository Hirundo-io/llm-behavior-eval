"""Write tests/golden/<name>.json from the legacy engine (frozen checkout only)."""

import argparse
import importlib
import json
import pkgutil
import subprocess
from pathlib import Path
from typing import Any

from scripts.parity import removal_baseline
from scripts.parity.historical import require_frozen_checkout

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests" / "golden"


def _git(*args: str) -> str:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def dumps(value: Any, indent: int = 0) -> str:
    """Indented JSON that keeps lists and dicts of scalars on one line."""
    pad = " " * (indent + 1)
    if isinstance(value, dict) and any(
        isinstance(v, dict | list) for v in value.values()
    ):
        items = [
            f"{pad}{json.dumps(key, ensure_ascii=False)}: {dumps(item, indent + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    if isinstance(value, list) and any(isinstance(v, dict | list) for v in value):
        items = [f"{pad}{dumps(item, indent + 1)}" for item in value]
        return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"
    return json.dumps(value, ensure_ascii=False)


def main() -> None:
    names = sorted(
        module.name
        for module in pkgutil.iter_modules(removal_baseline.__path__)
        if not module.name.startswith("_")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help=f"default: all of {names}")
    args = parser.parse_args()
    unknown = sorted(set(args.names) - set(names))
    if unknown:
        parser.error(f"unknown baselines {unknown}; choose from {names}")
    require_frozen_checkout()
    if _git("status", "--porcelain", "--", "llm_behavior_eval"):
        raise SystemExit("llm_behavior_eval/ differs from HEAD; capture needs it clean")
    commit = _git("rev-parse", "HEAD")
    GOLDEN.mkdir(exist_ok=True)
    for name in args.names or names:
        module = importlib.import_module(f"{removal_baseline.__name__}.{name}")
        payload = {
            "provenance": {
                "commit": commit,
                "capture": f"python -m scripts.parity.removal_baseline {name}",
                "sources": list(module.SOURCES),
            },
            "values": module.capture(),
        }
        (GOLDEN / f"{name}.json").write_text(dumps(payload) + "\n")
        print(f"wrote tests/golden/{name}.json at {commit}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Refresh only the completed CCPC500 routing entries in the archive inventory."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    prefix = "probe/rsch76-ccpc500-base-routing-v1/"
    inventory = root / "SHA256SUMS"
    preserved = [
        line
        for line in inventory.read_text(encoding="utf-8").splitlines()
        if not (len(line) > 66 and line[66:].lstrip().startswith(prefix))
    ]
    base = root / prefix.rstrip("/")
    refreshed = [f"{sha256(path)}  {path.relative_to(root)}" for path in sorted(p for p in base.rglob("*") if p.is_file())]
    inventory.write_text("\n".join(preserved + refreshed) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

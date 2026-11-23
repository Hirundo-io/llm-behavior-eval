"""Keep the code's upstream pins in step with ``.github/upstream-revisions.env``.

The manifest lists every upstream revision a native benchmark pins (a dataset
or reference-code commit), one entry of three lines each::

    # renovate: datasource=github-digest depName=<owner>/<repo> currentValue=<branch>
    # source: <file from the repository root>:<CONSTANT>
    <KEY>=<40-hex commit>

``<CONSTANT>`` is a module-level string constant, or ``<CONSTANT>["<key>"]``
for one value of a module-level dict literal. Renovate (``.github/renovate.json``)
reads the first and third lines and proposes a new commit when the branch
moves. ``tests/test_upstream_revisions.py`` then fails until the code agrees:

* ``python scripts/sync_upstream_revisions.py --check`` lists the mismatches;
* ``python scripts/sync_upstream_revisions.py`` replaces the old commit with the
  manifest's in every text file of the distribution that owns the source file
  (its code, README and target metrics), then lists what it changed.

Syncing only moves the pin: review the upstream changes and update anything
derived from the data (checksums, expected row counts, parity notes) in the
same pull request.
"""

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path

from workspace import ROOT, members

MANIFEST = ROOT / ".github" / "upstream-revisions.env"
SHA = re.compile(r"[a-f0-9]{40}")
KEY = re.compile(r"[A-Z][A-Z0-9_]*")
SOURCE = re.compile(
    r'(?P<file>[^:\s]+):(?P<name>[A-Za-z_]\w*)(?:\["(?P<key>[^"]+)"\])?'
)
RENOVATE_PREFIX = "# renovate: "
SOURCE_PREFIX = "# source: "
TEXT_SUFFIXES = frozenset({".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt"})


@dataclass(frozen=True)
class Pin:
    """One manifest entry."""

    key: str
    revision: str
    renovate: str
    """The entry's ``# renovate:`` line."""
    file: Path
    name: str
    item: str | None
    """The dict key when the source is ``<CONSTANT>["<key>"]``."""
    source: str
    """The entry's ``<file>:<CONSTANT>`` as written."""


def parse_manifest(text: str, root: Path = ROOT) -> list[Pin]:
    """The manifest's entries.

    Raises:
        ValueError: An entry is incomplete, a key repeats, a revision is not a
            full commit hash, or a source file is missing.
    """
    lines = [line.strip() for line in text.splitlines()]
    pins: list[Pin] = []
    for index, line in enumerate(lines):
        if not line or line.startswith("#"):
            continue
        key, _, revision = line.partition("=")
        where = f"line {index + 1}"
        if not KEY.fullmatch(key):
            raise ValueError(f"{where}: {key!r} is not an UPPER_SNAKE_CASE key")
        if not SHA.fullmatch(revision):
            raise ValueError(f"{where}: {key} must be a full 40-character commit hash")
        if index < 2 or not lines[index - 2].startswith(RENOVATE_PREFIX):
            raise ValueError(
                f"{where}: {key} needs a '{RENOVATE_PREFIX}' line two lines above"
            )
        source = lines[index - 1].removeprefix(SOURCE_PREFIX)
        match = (
            SOURCE.fullmatch(source)
            if lines[index - 1].startswith(SOURCE_PREFIX)
            else None
        )
        if match is None:
            raise ValueError(
                f"{where}: {key} needs a '{SOURCE_PREFIX}<file>:<CONSTANT>' line above it"
            )
        file = root / match["file"]
        if not file.is_file():
            raise ValueError(
                f"{where}: {key} source file {match['file']} does not exist"
            )
        if any(pin.key == key for pin in pins):
            raise ValueError(f"{where}: {key} is repeated")
        pins.append(
            Pin(
                key,
                revision,
                lines[index - 2],
                file,
                match["name"],
                match["key"],
                source,
            )
        )
    return pins


def code_revision(pin: Pin) -> str:
    """The value the source file assigns to the pinned constant.

    Raises:
        ValueError: The file has no such module-level string (or dict item).
    """
    tree = ast.parse(pin.file.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not (isinstance(target, ast.Name) and target.id == pin.name):
            continue
        if pin.item is not None:
            items = value.keys if isinstance(value, ast.Dict) else []
            values = value.values if isinstance(value, ast.Dict) else []
            matching = [
                item_value
                for item_key, item_value in zip(items, values, strict=True)
                if isinstance(item_key, ast.Constant) and item_key.value == pin.item
            ]
            if len(matching) != 1:
                break
            value = matching[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        break
    raise ValueError(f"{pin.key}: {pin.source} is not a module-level string constant")


def mismatches(pins: list[Pin]) -> dict[Pin, str]:
    """Each pin whose code disagrees with the manifest, with the code's value."""
    found = {}
    for pin in pins:
        current = code_revision(pin)
        if current != pin.revision:
            found[pin] = current
    return found


def owning_directory(file: Path) -> Path:
    """The workspace member directory that holds ``file``.

    Raises:
        ValueError: ``file`` is in no workspace member.
    """
    for member in members():
        if file.is_relative_to(member.path):
            return member.path
    raise ValueError(f"{file} is not in a workspace member")


def sync(pins: list[Pin]) -> list[Path]:
    """Replace each stale commit with the manifest's; the files changed."""
    changed: list[Path] = []
    for pin, old in mismatches(pins).items():
        directory = owning_directory(pin.file)
        for path in sorted(directory.rglob("*")):
            if path.suffix not in TEXT_SUFFIXES or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            if old in text:
                path.write_text(text.replace(old, pin.revision), encoding="utf-8")
                changed.append(path)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="only list mismatches")
    args = parser.parse_args()
    pins = parse_manifest(MANIFEST.read_text(encoding="utf-8"))
    stale = mismatches(pins)
    if args.check:
        for pin, current in stale.items():
            print(
                f"{pin.key}: {pin.source} is {current}, the manifest says {pin.revision}"
            )
        raise SystemExit(1 if stale else 0)
    for path in sync(pins):
        print(f"updated {path.relative_to(ROOT)}")
    print(
        "Review the upstream changes, then update checksums and anything derived from the data."
    )


if __name__ == "__main__":
    main()

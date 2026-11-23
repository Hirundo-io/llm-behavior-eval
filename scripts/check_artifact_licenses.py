"""Check the license metadata and files in every built wheel and sdist.

Usage: ``uv build --all-packages --out-dir dist`` then
``python scripts/check_artifact_licenses.py dist``.

Expectations come from each distribution's ``pyproject.toml``, so a new
benchmark needs no edit here: the ``License-Expression`` metadata equals
``project.license``, and every file in ``project.license-files`` (a LICENSE,
and a NOTICE where upstream terms ask for one) is in the artifact with the
same bytes. An MIT distribution's LICENSE must be the repository's LICENSE.
Every distribution must have exactly one wheel and one sdist, and nothing else
may be in the directory.
"""

import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

from workspace import ROOT, Distribution, members, normalize, root


def artifact_files(path: Path) -> dict[str, bytes]:
    """Every regular file in a wheel or sdist, by its path in the archive."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return {
                name: archive.read(name)
                for name in archive.namelist()
                if not name.endswith("/")
            }
    if path.name.endswith(".tar.gz"):
        contents = {}
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                extracted = archive.extractfile(member) if member.isfile() else None
                if extracted is not None:
                    contents[member.name] = extracted.read()
        return contents
    raise ValueError(f"{path.name}: not a wheel or sdist")


def artifact_name(path: Path) -> str:
    """The normalized distribution name of a wheel or sdist file."""
    return normalize(path.name.split("-")[0])


def check_artifact(path: Path, distribution: Distribution) -> list[str]:
    """Problems with one artifact's license metadata and files."""
    contents = artifact_files(path)
    problems = []
    expression = distribution.project.get("license")
    if not isinstance(expression, str):
        return [f"{distribution.relative}: project.license must be an SPDX string"]
    metadata = [
        value
        for name, value in contents.items()
        if name.endswith((".dist-info/METADATA", "/PKG-INFO")) and name.count("/") == 1
    ]
    if not metadata:
        problems.append(f"{path.name}: no METADATA or PKG-INFO")
    for value in metadata:
        found = BytesParser().parsebytes(value)["License-Expression"]
        if found != expression:
            problems.append(
                f"{path.name}: License-Expression {found!r}, not {expression!r}"
            )
    for license_file in distribution.project.get("license-files", []):
        expected = (distribution.path / license_file).read_bytes()
        if expression == "MIT" and Path(license_file).name == "LICENSE":
            if expected != (ROOT / "LICENSE").read_bytes():
                problems.append(
                    f"{distribution.relative}/{license_file}: an MIT LICENSE must "
                    "be a copy of the repository's LICENSE"
                )
        inner = f"licenses/{license_file}" if path.suffix == ".whl" else license_file
        shipped = [
            value
            for name, value in contents.items()
            if "/" in name and name.split("/", 1)[1] == inner
        ]
        if not shipped:
            problems.append(f"{path.name}: {license_file} is missing")
        elif any(value != expected for value in shipped):
            problems.append(f"{path.name}: {license_file} differs from the source")
    return problems


def check_directory(
    directory: Path, distributions: list[Distribution] | None = None
) -> list[str]:
    """Problems with the artifacts in ``directory`` (all distributions by default)."""
    if distributions is None:
        distributions = [root(), *members()]
    by_name = {distribution.name: distribution for distribution in distributions}
    artifacts = sorted(
        path for path in directory.iterdir() if path.name.endswith((".whl", ".tar.gz"))
    )
    problems = []
    for name in by_name:
        kinds = sorted(
            "wheel" if path.suffix == ".whl" else "sdist"
            for path in artifacts
            if artifact_name(path) == name
        )
        if kinds != ["sdist", "wheel"]:
            problems.append(f"{name}: expected one wheel and one sdist, found {kinds}")
    for path in artifacts:
        distribution = by_name.get(artifact_name(path))
        if distribution is None:
            problems.append(f"{path.name}: not a distribution of this repository")
        else:
            problems.extend(check_artifact(path, distribution))
    return problems


def main() -> None:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist"
    problems = check_directory(directory)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        raise SystemExit(1)
    print(f"License files and metadata are correct in {directory}")


if __name__ == "__main__":
    main()

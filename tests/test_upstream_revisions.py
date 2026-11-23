"""The code's upstream pins match ``.github/upstream-revisions.env``."""

from pathlib import Path

import pytest

import sync_upstream_revisions as sync
from workspace import ROOT, Distribution

OLD = "1" * 40
NEW = "2" * 40
GITHUB = "# renovate: datasource=github-digest depName=org/repo currentValue=main"


def test_manifest_matches_the_code() -> None:
    pins = sync.parse_manifest(sync.MANIFEST.read_text(encoding="utf-8"))
    assert sync.mismatches(pins) == {}
    for pin in pins:
        # Each pin lives in a workspace member, where sync can rewrite it.
        sync.owning_directory(pin.file)


def write_source(tmp_path: Path) -> Path:
    source = tmp_path / "pkg" / "dataset.py"
    source.parent.mkdir()
    source.write_text(
        f'COMMIT = "{OLD}"\n'
        f'REVISIONS: dict[str, str] = {{"a": "{OLD}", "b": "{NEW}"}}\n'
        "NOT_A_STRING = 3\n"
    )
    return source


def manifest(source: str, revision: str = NEW, key: str = "DEMO_REV") -> str:
    return f"# header\n\n{GITHUB}\n# source: {source}\n{key}={revision}\n"


@pytest.mark.parametrize(
    ("source", "current"),
    [
        ("pkg/dataset.py:COMMIT", OLD),
        ('pkg/dataset.py:REVISIONS["a"]', OLD),
        ('pkg/dataset.py:REVISIONS["b"]', NEW),
    ],
)
def test_code_revision(tmp_path: Path, source: str, current: str) -> None:
    write_source(tmp_path)
    [pin] = sync.parse_manifest(manifest(source), tmp_path)
    assert sync.code_revision(pin) == current
    assert sync.mismatches([pin]) == ({} if current == NEW else {pin: OLD})


@pytest.mark.parametrize(
    "source", ["pkg/dataset.py:MISSING", 'pkg/dataset.py:REVISIONS["c"]',
               "pkg/dataset.py:NOT_A_STRING", 'pkg/dataset.py:COMMIT["a"]']
)  # fmt: skip
def test_code_revision_needs_a_string_constant(tmp_path: Path, source: str) -> None:
    write_source(tmp_path)
    [pin] = sync.parse_manifest(manifest(source), tmp_path)
    with pytest.raises(ValueError, match="not a module-level string constant"):
        sync.code_revision(pin)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (manifest("pkg/dataset.py:COMMIT", revision="abc"), "full 40-character"),
        (manifest("pkg/dataset.py:COMMIT", key="lower"), "UPPER_SNAKE_CASE"),
        (manifest("pkg/missing.py:COMMIT"), "does not exist"),
        (f"{GITHUB}\nDEMO_REV={NEW}\n", "needs a '# renovate: ' line"),
        (f"{GITHUB}\n# note\nDEMO_REV={NEW}\n", "needs a '# source: "),
        (manifest("pkg/dataset.py:COMMIT") * 2, "is repeated"),
    ],
)
def test_manifest_errors(tmp_path: Path, text: str, message: str) -> None:
    write_source(tmp_path)
    with pytest.raises(ValueError, match=message):
        sync.parse_manifest(text, tmp_path)


def test_sync_replaces_the_old_commit_in_the_owning_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_source(tmp_path)
    readme = source.parent / "README.md"
    readme.write_text(f"Pinned at {OLD}.\n")
    elsewhere = tmp_path / "other.md"
    elsewhere.write_text(OLD)
    monkeypatch.setattr(sync, "owning_directory", lambda file: source.parent)
    [pin] = sync.parse_manifest(manifest("pkg/dataset.py:COMMIT"), tmp_path)
    assert sync.sync([pin]) == [readme, source]
    assert sync.code_revision(pin) == NEW
    assert readme.read_text() == f"Pinned at {NEW}.\n"
    assert elsewhere.read_text() == OLD
    assert sync.sync([pin]) == []


@pytest.mark.parametrize(
    "other",
    [
        "3" * 40,  # B is stale too, and wants a different commit than A.
        OLD,  # B is up to date, and its commit is A's old one.
    ],
)
def test_sync_refuses_a_shared_old_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, other: str
) -> None:
    # Two pins in one distribution share the commit OLD; replacing it for A
    # would silently move B to A's new commit.
    source = tmp_path / "pkg" / "dataset.py"
    source.parent.mkdir()
    code = f'A = "{OLD}"\nB = "{OLD}"\n'
    source.write_text(code)
    monkeypatch.setattr(sync, "owning_directory", lambda file: source.parent)
    text = manifest("pkg/dataset.py:A", NEW, "A_REV") + manifest(
        "pkg/dataset.py:B", other, "B_REV"
    )
    pins = sync.parse_manifest(text, tmp_path)
    with pytest.raises(ValueError, match="B_REV in the same distribution"):
        sync.sync(pins)
    assert source.read_text() == code


def test_sync_moves_pins_that_share_a_commit_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pkg" / "dataset.py"
    source.parent.mkdir()
    source.write_text(f'A = "{OLD}"\nB = "{OLD}"\n')
    monkeypatch.setattr(sync, "owning_directory", lambda file: source.parent)
    text = manifest("pkg/dataset.py:A", NEW, "A_REV") + manifest(
        "pkg/dataset.py:B", NEW, "B_REV"
    )
    pins = sync.parse_manifest(text, tmp_path)
    assert sync.sync(pins) == [source]
    assert sync.mismatches(pins) == {}


def test_manifest_lives_in_the_repository() -> None:
    assert sync.MANIFEST == ROOT / ".github" / "upstream-revisions.env"


def test_owning_directory_is_the_innermost_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    group = tmp_path / "llm_behavior_eval" / "group"
    benchmark = group / "benchmark"
    monkeypatch.setattr(
        sync,
        "members",
        lambda: [Distribution(group), Distribution(benchmark)],
    )
    assert sync.owning_directory(benchmark / "dataset.py") == benchmark
    assert sync.owning_directory(group / "README.md") == group
    with pytest.raises(ValueError, match="not in a workspace member"):
        sync.owning_directory(tmp_path / "elsewhere.py")

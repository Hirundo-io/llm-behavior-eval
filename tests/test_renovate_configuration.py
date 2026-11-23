"""Renovate can extract every pinned upstream revision."""

import json
import re

import sync_upstream_revisions as sync
from workspace import ROOT

CONFIG = json.loads((ROOT / ".github" / "renovate.json").read_text(encoding="utf-8"))


def manager_pattern(file: str) -> re.Pattern[str]:
    [manager] = [
        manager
        for manager in CONFIG["customManagers"]
        if any(
            re.search(pattern.strip("/"), file)
            for pattern in manager["managerFilePatterns"]
        )
    ]
    [match_string] = manager["matchStrings"]
    return re.compile(match_string.replace("(?<", "(?P<"))


def test_every_upstream_revision_is_extracted() -> None:
    text = sync.MANIFEST.read_text(encoding="utf-8")
    pattern = manager_pattern(".github/upstream-revisions.env")
    extracted = {match["currentDigest"]: match for match in pattern.finditer(text)}
    pins = sync.parse_manifest(text)
    assert set(extracted) == {pin.revision for pin in pins}
    for pin in pins:
        assert extracted[pin.revision].group(0).startswith(pin.renovate)


def test_entry_formats_are_extracted() -> None:
    pattern = manager_pattern(".github/upstream-revisions.env")
    github = (
        "# renovate: datasource=github-digest depName=org/repo currentValue=main\n"
        "# source: packages/x/src/llm_behavior_eval/x/dataset.py:COMMIT\n"
        f"X_DATA_REV={'a' * 40}\n"
    )
    hub = (
        "# renovate: datasource=git-refs depName=org/data "
        "packageName=https://huggingface.co/datasets/org/data currentValue=main\n"
        '# source: packages/y/src/llm_behavior_eval/y/dataset.py:REVISIONS["a"]\n'
        f"Y_DATA_REV={'b' * 40}\n"
    )
    match = pattern.search(github)
    assert match is not None
    assert (match["datasource"], match["depName"], match["packageName"]) == (
        "github-digest",
        "org/repo",
        None,
    )
    assert (match["currentValue"], match["currentDigest"]) == ("main", "a" * 40)
    match = pattern.search(hub)
    assert match is not None
    assert match["packageName"] == "https://huggingface.co/datasets/org/data"
    assert match["currentDigest"] == "b" * 40


def test_renovate_policy() -> None:
    assert CONFIG["minimumReleaseAge"] == "7 days"
    assert CONFIG["minimumReleaseAgeBehaviour"] == "timestamp-required"
    rules = {rule.get("groupName"): rule for rule in CONFIG["packageRules"]}
    assert set(rules["python"]["matchFileNames"]) == {
        "pyproject.toml",
        "packages/**/pyproject.toml",
        "uv.lock",
    }
    upstream = rules[None]
    assert upstream["matchFileNames"] == [".github/upstream-revisions.env"]
    assert "sync_upstream_revisions.py" in upstream["prBodyNotes"][0]
    global_config = json.loads(
        (ROOT / ".github" / "renovate-global.json").read_text(encoding="utf-8")
    )
    assert global_config == {"repositories": ["Hirundo-io/llm-behavior-eval"]}


def test_renovate_image_is_extracted() -> None:
    workflow = (ROOT / ".github" / "workflows" / "renovate.yaml").read_text(
        encoding="utf-8"
    )
    match = manager_pattern(".github/workflows/renovate.yaml").search(workflow)
    assert match is not None
    assert match["depName"] == "renovate/renovate"

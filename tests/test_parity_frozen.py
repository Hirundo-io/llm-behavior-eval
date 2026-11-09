"""Frozen-source staging integrity; these tests are not inference evidence."""

import json
from types import SimpleNamespace

import pytest
from datasets import Dataset

from scripts.parity.compare import digest
from scripts.parity.prepare_frozen import prepare_frozen


@pytest.fixture
def pinned(tmp_path):
    primary = tmp_path / "primary"
    directory = primary / "dataset"
    directory.mkdir(parents=True)
    dataset = Dataset.from_list(
        [dict(question=f"q{i}", answer=f"a{i}") for i in range(7)]
    )
    path = directory / "train.parquet"
    dataset.to_parquet(str(path))
    repository = "hirundo-io/bbq-religion-bias-free-text"
    ordered = Dataset.from_dict({"index": list(range(7))}).shuffle(seed=42)["index"]
    ids = [f"{repository}:{index}" for index in ordered]
    source = dict(
        repository=repository,
        revision="frozen-revision",
        rows=7,
        split="train",
        ids=ids[:2],
        parquet_sha256=digest(path),
        dataset_seed=42,
    )
    identity = dict(case="bbq:religion:bias", samples=2, answer="a", judge="j")
    (primary / "source.json").write_text(json.dumps(source))
    (primary / "identity-config.json").write_text(json.dumps(identity))
    return primary, tmp_path / "extended", ids


def test_extend_frozen_selection_through_production_task(pinned):
    primary, output, expected = pinned
    original = (primary / "source.json").read_bytes()
    provenance = prepare_frozen(primary, output, 100)
    source = json.loads((output / "source.json").read_text())
    assert source["ids"] == expected
    assert source["revision"] == "frozen-revision"
    assert provenance["full_source"] is True
    assert provenance["previous_samples"] == 2
    assert provenance["selected_samples"] == 7
    assert (primary / "source.json").read_bytes() == original
    assert (output / "dataset/train.parquet").read_bytes() == (
        primary / "dataset/train.parquet"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        prepare_frozen(primary, output, 100)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("repository", "changed", "repository"),
        ("parquet_sha256", "changed", "source bytes"),
        ("rows", 8, "source size"),
        ("dataset_seed", 43, "seed 42"),
        ("ids", ["changed"], "Original selection"),
    ],
)
def test_reject_changed_frozen_pins(pinned, field, value, message):
    primary, output, _ = pinned
    path = primary / "source.json"
    source = json.loads(path.read_text())
    path.write_text(json.dumps({**source, field: value}))
    with pytest.raises(ValueError, match=message):
        prepare_frozen(primary, output, 100)
    assert not output.exists()


def test_reject_changed_production_selection(pinned, monkeypatch):
    from scripts.parity import prepare_frozen as module

    primary, output, ids = pinned
    monkeypatch.setattr(
        module._registry,
        "bbq",
        lambda **kwargs: SimpleNamespace(
            dataset=[SimpleNamespace(id=ident) for ident in reversed(ids)]
        ),
    )
    with pytest.raises(ValueError, match="Production task selection"):
        prepare_frozen(primary, output, 100)
    assert not output.exists()

"""Platform-style raw HF snapshot bundles retain native identities offline."""

import pytest
from datasets import Dataset, load_from_disk

from llm_behavior_eval import _registry, list_dataset_presets
from llm_behavior_eval.ccpc import contract
from llm_behavior_eval.ccpc.dataset import load_ccpc_dataset
from tests.native_fixtures import local_task_dataset


@pytest.mark.parametrize(
    "name",
    [
        "bbq",
        "bloom",
        "unqover",
        "halueval",
        "medhallu",
        "prompt_injection",
        "orbench",
        "xstest",
        "ccpc",
    ],
)
def test_raw_snapshot_parquet_preserves_native_ids(name, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    source = local_task_dataset(name, tmp_path, monkeypatch)
    rows = load_from_disk(str(source))
    assert isinstance(rows, Dataset)
    # Bundles contain raw HF repository files, not Dataset.save_to_disk metadata.
    snapshot = tmp_path / "bundle" / "datasets" / name
    data = snapshot / "data"
    data.mkdir(parents=True)
    split = "test" if name == "ccpc" else "train"
    parquet = data / f"{split}-00000-of-00001.parquet"
    rows.to_parquet(str(parquet))
    task = getattr(_registry, name)
    expected = task(dataset_path=str(source), max_samples=1).dataset
    actual = task(dataset_path=str(snapshot), max_samples=1).dataset

    def signature(sample):
        content = sample.input
        if not isinstance(content, str):
            content = [(message.role, message.text) for message in content]
        return sample.id, content, sample.target

    assert [signature(s) for s in actual] == [signature(s) for s in expected]
    assert actual.name == expected.name
    if name == "ccpc":
        # Corrupt a row outside the requested cap: full-source pin still applies.
        changed = rows.to_list()
        changed[-1]["question"] = "tampered outside selected prefix"
        Dataset.from_list(changed).to_parquet(str(parquet))
        with pytest.raises(ValueError, match="fingerprint"):
            load_ccpc_dataset(dataset_path=str(snapshot), max_samples=1)


def test_public_ccpc_bundle_identity_remains_pinned():
    assert contract.CCPC_DATASET_ID == "chinese_censorship"
    assert contract.CCPC_DATASET_REPOSITORY == "hirundo-io/ccpc-500"
    assert contract.CCPC_DATASET_REVISION == "e778807d1cbe31f5e4f4ff1941d3d6fe48faed70"
    assert contract.CCPC_DATASET_FINGERPRINT == (
        "e923afaa3836c51ac419dd88fffb2180b73ed7d66ed74085cb72ea80ed2a7b25"
    )
    preset = next(p for p in list_dataset_presets() if p.name == "chinese_censorship")
    assert preset.dataset_ids == ("hirundo-io/ccpc-500",)
    assert preset.inspect_tasks[0].dataset_id == "chinese_censorship"

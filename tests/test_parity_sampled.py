"""Evidence gates reject changed subsets, missing judges and metric mismatches."""

import json

import pytest

from scripts.parity.recorded_judgments import RecordedJudgments
from scripts.parity.sampled_report import pin_samples, summarize_samples


class Tokenizer:
    def encode(self, text):
        return list(text.encode())


def test_replay_requires_actual_prompt_and_preserves_response():
    replay = RecordedJudgments(
        Tokenizer(),
        [dict(input_token_ids=[65], answer=" No\n", finish_reason="length")],
    )
    with pytest.raises(ValueError, match="differs"):
        replay(None, ["B"])
    assert replay.consumed == 0
    assert replay(None, ["A"]) == [
        [dict(generated_text=" No\n", finish_reason="length")]
    ]
    with pytest.raises(ValueError, match="exhausted"):
        replay(None, ["A"])


@pytest.fixture
def sampled(tmp_path):
    from pathlib import Path

    expected = json.loads(Path("scripts/parity/pending.json").read_text())[
        "expected_cases"
    ]
    for case in expected:
        root = tmp_path / case.replace(":", "-")
        root.mkdir()
        (root / "source.json").write_text(
            json.dumps(dict(rows=1000, ids=list(range(100))))
        )
        (root / "identity-config.json").write_text(
            json.dumps(dict(case=case, samples=100))
        )
    root = tmp_path / "halueval"
    (root / "manifest.json").write_text(
        json.dumps(dict(cases=[dict(runs=dict(legacy={}, native={}))]))
    )
    (root / "strict-report.json").write_text(
        json.dumps(
            dict(
                name="halueval:small-slice",
                samples=100,
                status="compared",
                errors=[],
                mismatches=[],
                metrics=[dict(name="accuracy", delta=0)],
                judge_boundary=dict(
                    identical_multiset=True,
                    identical_raw_outputs=True,
                    identical_response_order=True,
                ),
            )
        )
    )
    (root / "legacy").mkdir()
    (root / "legacy/responses.json").write_text(json.dumps([dict(grade=True)] * 100))
    return tmp_path


@pytest.mark.parametrize(
    "damage", [None, "metric", "judge", "count", "identity", "ids"]
)
def test_sampled_gate_binds_pins_and_requires_exact_evidence(sampled, damage):
    manifest = pin_samples(sampled)
    root = sampled / "halueval"
    if damage in {"identity", "ids"}:
        path = root / ("source.json" if damage == "ids" else "identity-config.json")
        data = json.loads(path.read_text())
        data["ids" if damage == "ids" else "samples"] = (
            list(range(100, 200)) if damage == "ids" else 99
        )
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            summarize_samples(sampled, manifest)
        return
    path = root / "strict-report.json"
    data = json.loads(path.read_text())
    if damage == "metric":
        data["metrics"][0]["delta"] = 1e-15
    elif damage == "judge":
        data["judge_boundary"].pop("identical_raw_outputs")
    elif damage == "count":
        data["samples"] = 99
    path.write_text(json.dumps(data))
    report = summarize_samples(sampled, manifest)
    row = next(row for row in report["cases"] if row["case"] == "halueval")
    assert row["status"] == ("exact" if damage is None else "incomplete")
    assert not report["complete"]
    assert len(report["unfinished"]) == (30 if damage is None else 31)


def test_followup_preserves_original_evidence_and_answer(tmp_path):
    from scripts.parity.rescore_followup import stage

    primary = tmp_path / "primary"
    primary.mkdir()
    answer = tmp_path / "actual.eval"
    answer.write_bytes(b"immutable observation")
    (primary / "answer-log.txt").write_text(str(answer))
    (primary / "strict-report.json").write_text("{}")
    for name in ("dataset", "legacy"):
        (primary / name).mkdir()
        (primary / name / "artifact").write_bytes(b"original bytes")
    for name in (
        "source.json",
        "identity-config.json",
        "legacy-boundary.json",
        "legacy-judge-boundary.json",
        "legacy-metrics-unrounded.json",
        "legacy-reuse-provenance.json",
    ):
        (primary / name).write_text('{"preserve": "exact bytes"}\n')
    result = stage(primary, tmp_path / "followup", tmp_path)
    assert result["answer_log"] == str(answer)
    assert answer.read_bytes() == b"immutable observation"
    for row in result["reused_files"]:
        from pathlib import Path

        assert Path(row["original"]).read_bytes() == Path(row["copy"]).read_bytes()
    with pytest.raises(FileExistsError):
        stage(primary, tmp_path / "followup", tmp_path)


def test_followup_requires_preserved_baseline_report(tmp_path):
    from scripts.parity.rescore_followup import stage

    with pytest.raises(ValueError, match="baseline comparison"):
        stage(tmp_path, tmp_path / "followup", tmp_path)
    assert not (tmp_path / "followup").exists()


@pytest.mark.parametrize("damage", [None, "source", "judgments", "eligibility"])
def test_metric_replay_requires_complete_recorded_selection(tmp_path, damage):
    from scripts.parity.replay_purple_metric import replay

    # Synthetic gate test only; actual aggregation evidence uses GPU artifacts.
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    rows: list[dict[str, str | None]] = [
        dict(judge="Yes" if i < 3 else "No", finish_reason="stop") for i in range(7)
    ]
    if damage == "eligibility":
        rows[0]["judge"] = None
    (legacy / "responses.json").write_text(json.dumps(rows))
    (tmp_path / "source.json").write_text(
        json.dumps(dict(ids=list(range(6 if damage == "source" else 7))))
    )
    (tmp_path / "legacy-boundary.json").write_text(json.dumps([{}] * 7))
    (tmp_path / "legacy-judge-boundary.json").write_text(
        json.dumps([{}] * (6 if damage == "judgments" else 7))
    )
    (tmp_path / "legacy-metrics-unrounded.json").write_text(
        json.dumps(dict(attack_success_rate=42.85714285714286))
    )
    if damage:
        with pytest.raises(ValueError):
            replay(tmp_path)
    else:
        result = replay(tmp_path)
        assert result["exact"] is True
        assert result["rows"] == result["actual_judgments"] == 7
        assert result["labels"] == {"Yes": 3, "No": 4}

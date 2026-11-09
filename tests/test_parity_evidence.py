import json
from typing import TYPE_CHECKING, cast

import pytest

from scripts.parity import compare as comparator
from scripts.parity.tokenizer_probe import legacy_text_probe

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    old = [
        {
            "verdict": "CORRECT",
            "answer": "yes",
            "question": "Q",
            "finish": "length",
            "prompt": "rendered",
        }
    ]
    path = tmp_path / "responses.json"
    path.write_text(json.dumps(old))
    (tmp_path / "ids.json").write_text(
        json.dumps({"responses_sha256": comparator.digest(path), "ids": ["dataset:7"]})
    )
    (tmp_path / "metrics.csv").write_text("accuracy\n1\n")
    sample = {"id": "dataset:7", **old[0], "finish": "max_tokens"}
    log = {
        "status": "success",
        "samples": [sample],
        "results": {
            "scores": [{"name": "hallu", "metrics": {"accuracy": {"value": 100}}}]
        },
    }
    monkeypatch.setattr(comparator, "read_log", lambda path: log)
    run = dict.fromkeys(comparator.REQUIRED_RUN, "fixture-only")
    case = {
        "name": "halueval",
        "family": "halueval",
        "runs": {"legacy": run, "native": run},
        "responses": "responses.json",
        "identity": "ids.json",
        "eval": "native.eval",
        "metrics_csv": "metrics.csv",
        "expected_samples": 1,
        "fields": {
            name: {"legacy": key, "native": key}
            for name, key in [
                ("verdict", "verdict"),
                ("answer", "answer"),
                ("judge_question", "question"),
                ("finish_reason", "finish"),
                ("rendered_prompt", "prompt"),
            ]
        },
        "metrics": [
            {
                "name": "accuracy",
                "unit": "percent",
                "legacy_column": "accuracy",
                "scale": 100,
                "native_metric": "accuracy",
                "native_scorer": "hallu",
            }
        ],
    }
    return tmp_path, case, log


def test_units_and_finish_normalization(evidence):
    root, case, _ = evidence
    report = comparator.compare_case(root, case)
    assert report["status"] == "compared"
    assert report["metrics"][0]["delta_pp"] == 0
    assert report["verdict_agreement_percent"] == 100
    assert not report["mismatches"]
    assert "pending" in report["acceptance"]


@pytest.mark.parametrize(
    "failure",
    ["missing", "duplicate", "failed", "field", "hash", "unit", "nan", "provenance"],
)
def test_fail_closed(evidence, failure):
    root, case, log = evidence
    if failure == "missing":
        log["samples"][0]["id"] = "wrong"
    elif failure == "duplicate":
        log["samples"] *= 2
    elif failure == "failed":
        log["status"] = "error"
    elif failure == "field":
        del log["samples"][0]["question"]
    elif failure == "hash":
        (root / "responses.json").write_text("[]")
    elif failure == "unit":
        case["metrics"][0]["unit"] = "fraction"
    elif failure == "nan":
        log["results"]["scores"][0]["metrics"]["accuracy"]["value"] = float("nan")
    else:
        case["runs"]["native"] = {}
    report = comparator.compare_case(root, case)
    assert report["status"] == "incomplete"
    assert report["errors"]


def test_mismatch_and_absent_case(evidence):
    root, case, log = evidence
    log["samples"][0]["verdict"] = "INCORRECT"
    report = comparator.compare(
        {"cases": [case], "expected_cases": ["halueval", "bbq"]}, root
    )
    assert report["rows"][0]["verdict_agreement_percent"] == 0
    assert report["rows"][0]["mismatches"][0]["id"] == "dataset:7"
    assert report["rows"][1]["status"] == "incomplete"


def test_tokenizer_limits_padding_special_tokens_and_purple_question():
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            ids = ([1] if kwargs.get("add_special_tokens", True) else []) + [
                ord(c) for c in texts[0]
            ]
            ids = ids[: kwargs["max_length"]]
            mask = [1] * len(ids)
            pad = kwargs["max_length"] - len(ids)
            return {
                "input_ids": [ids + [0] * pad],
                "attention_mask": [mask + [0] * pad],
            }

        def decode(self, ids, *, skip_special_tokens):
            assert skip_special_tokens
            return "".join(chr(i) for i in ids if i > 1)

    result = legacy_text_probe(Tokenizer(), "Q" * 2000, "T" * 500, stereotype="S")
    assert result["judge_question"] == "Q" * 1023
    assert result["target"]["decoded"] == "T" * 256
    assert len(result["stereotype"]["input_ids"]) == 256
    assert sum(result["stereotype"]["attention_mask"]) == 1
    purple = legacy_text_probe(
        Tokenizer(), "rendered", "target", judge_question="J" * 500
    )
    assert purple["judge_question"] == "J" * 256


def test_real_binary_eval_readback(tmp_path):
    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import match
    from inspect_ai.solver import generate

    log = inspect_eval(
        Task(
            dataset=[Sample(id="stable:0", input="Q", target="A")],
            solver=generate(),
            scorer=match(),
        ),
        model="mockllm/model",
        log_dir=str(tmp_path),
        display="none",
    )[0]
    restored = comparator.read_log(__import__("pathlib").Path(log.location))
    assert restored["status"] == "success"
    assert restored["samples"][0]["id"] == "stable:0"


def test_generation_batches_are_hash_bound(evidence):
    root, case, _ = evidence
    path = root / "generations.jsonl"
    path.write_text(json.dumps({"finish_reasons": ["length"]}) + "\n")
    identity_path = root / "ids.json"
    identity = json.loads(identity_path.read_text())
    identity["generations_sha256"] = comparator.digest(path)
    identity_path.write_text(json.dumps(identity))
    case.update(generations="generations.jsonl", generation_fields=["finish_reasons"])
    case["fields"]["finish_reason"]["legacy"] = "generation.finish_reasons"
    assert comparator.compare_case(root, case)["status"] == "compared"
    path.write_text("{}")
    assert comparator.compare_case(root, case)["status"] == "incomplete"


@pytest.mark.parametrize(
    ("family", "name", "unit", "old", "new", "scale", "offset"),
    [
        ("medhallu", "accuracy", "percent", 75, 75, 1, 0),
        ("prompt_injection", "attack_success_rate", "percent", 75, 25, -1, 100),
        ("ccpc", "censorship_rate", "fraction", 25, 0.25, 0.01, 0),
    ],
)
def test_family_units(evidence, family, name, unit, old, new, scale, offset):
    root, case, log = evidence
    case["family"] = family
    (root / "metrics.csv").write_text(f"accuracy\n{old}\n")
    case["metrics"][0].update(name=name, unit=unit, scale=scale, offset=offset)
    log["results"]["scores"][0]["metrics"]["accuracy"]["value"] = new
    report = comparator.compare_case(root, case)
    assert report["status"] == "compared"
    assert report["metrics"][0]["delta_pp"] == 0


def test_empty_manifest_still_reports_all_required_splits(tmp_path):
    report = comparator.compare({"cases": [], "expected_cases": []}, tmp_path)
    assert len(report["rows"]) == 31
    assert all(row["status"] == "incomplete" for row in report["rows"])


def test_probe_against_actual_legacy_preprocessing(monkeypatch):
    from llm_behavior_eval.evaluation_utils import custom_dataset

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            rows, masks = [], []
            for text in texts:
                ids = ([1] if kwargs.get("add_special_tokens", True) else []) + [
                    ord(c) for c in text
                ]
                ids = ids[: kwargs["max_length"]]
                pad = kwargs["max_length"] - len(ids)
                rows.append(ids + [0] * pad)
                masks.append([1] * len(ids) + [0] * pad)
            return {"input_ids": rows, "attention_mask": masks}

        def decode(self, ids, *, skip_special_tokens):
            assert skip_special_tokens
            return "".join(chr(int(i)) for i in ids if i > 1)

    tokenizer = Tokenizer()
    rendered = "system:user:" + "Q" * 1100
    monkeypatch.setattr(
        custom_dataset, "safe_apply_chat_template", lambda *args, **kwargs: rendered
    )
    legacy = custom_dataset.free_text_preprocess_function(
        {
            "question": ["Q" * 1100],
            "answer": ["A" * 300],
            "stereotyped_answer": ["S" * 300],
            "judge_question": ["J" * 300],
        },
        cast("PreTrainedTokenizerBase", tokenizer),
        1024,
        256,
        True,
    )
    probe = legacy_text_probe(
        tokenizer, rendered, "A" * 300, stereotype="S" * 300, judge_question="J" * 300
    )
    for key, native_key in [
        ("test_input_ids", "prompt"),
        ("gt_answers", "target"),
        ("stereotyped_answers", "stereotype"),
    ]:
        assert legacy[key][0].tolist() == probe[native_key]["input_ids"]
    assert (
        tokenizer.decode(legacy["judge_questions"][0], skip_special_tokens=True)
        == probe["judge_question"]
    )


def test_committed_redesign_replay_covers_every_case_exactly():
    """The CPU replay record: all 31 required cases, exact, no fabricated rows."""
    from pathlib import Path

    root = Path(__file__).parents[1] / "scripts" / "parity"
    report = json.loads((root / "redesign-replay.json").read_text())
    expected = json.loads((root / "pending.json").read_text())["expected_cases"]
    assert [case["case"] for case in report["cases"]] == expected
    assert report["exact"] is True and report["code_dirty"] is False
    assert report["totals"] == {
        "cases": 31,
        "exact_cases": 31,
        "samples": 3100,
        "judge_requests": 7448,
    }
    for case in report["cases"]:
        assert case["mismatches"] == []
        assert case["judge_requests"] == case["recorded_judge_requests"]
        assert (
            case["samples"]
            == case["prompt_ids_equal_server_ids"]
            == case["legacy_text_views_equal"]
            == case["per_sample_scores_equal"]
        )
    purple = json.loads((root / "redesign-purple-metric-replay.json").read_text())
    assert purple["exact"] is True
    assert purple["native_metric_asr"] == purple["retained_asr"]

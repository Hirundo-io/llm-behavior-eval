"""Qualification identity checks; these tests do not establish live parity."""

import json
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.parity.qualify_matrix import case_identity, immutable_json


def test_all_required_cases_have_unique_source_identities():
    cases = json.loads(Path("scripts/parity/pending.json").read_text())[
        "expected_cases"
    ]
    identities = [case_identity(case) for case in cases]
    assert len(identities) == 31
    assert len({item[1] for item in identities}) == 31
    assert case_identity("bloom:age:ambiguous") == (
        "bloom",
        "hirundo-io/bloom-age-ambiguous-bias-free-text",
        {"bias_type": "age", "kind": "ambiguous"},
    )
    assert case_identity("ccpc")[1] == "hirundo-io/ccpc-500"


def test_unknown_case_rejected():
    with pytest.raises(ValueError, match="Not a required matrix case"):
        case_identity("unqover:gender:unbias")


def test_existing_identity_cannot_change(tmp_path):
    path = tmp_path / "identity.json"
    immutable_json(path, {"samples": 2})
    immutable_json(path, {"samples": 2})
    with pytest.raises(ValueError, match="Identity changed"):
        immutable_json(path, {"samples": 500})
    assert json.loads(path.read_text()) == {"samples": 2}


def test_separate_scoring_retains_task_metrics(tmp_path, monkeypatch):
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.log import read_eval_log
    from inspect_ai.scorer import Score, accuracy, scorer
    from inspect_ai.solver import generate

    from scripts.parity import qualify_native

    @scorer(metrics=[])
    def fixed_score():
        async def score(state, target):
            return Score(value=1)

        return score

    def task_factory(**kwargs):
        return Task(
            dataset=[Sample(id="one", input="Q")],
            solver=generate(),
            scorer=fixed_score(),
            metrics=[accuracy()],
        )

    monkeypatch.setattr(qualify_native._registry, "halueval", task_factory)
    task_args = tmp_path / "task.json"
    task_args.write_text("{}")
    for phase in ["answer", "judge"]:
        monkeypatch.setattr(
            "sys.argv",
            [
                "qualify_native",
                phase,
                "--family",
                "halueval",
                "--root",
                str(tmp_path / "run"),
                "--task-args",
                str(task_args),
                "--answer",
                "mockllm/answer",
                "--judge",
                "mockllm/judge",
            ],
        )
        qualify_native.main()
    log = read_eval_log(str(tmp_path / "run/scored.eval"))
    assert log.results is not None
    assert log.results.scores[0].metrics["accuracy"].value == 1


def test_matrix_missing_results_are_not_zero(tmp_path):
    from scripts.parity.matrix_report import summarize

    report = summarize(tmp_path)
    assert report["complete"] is False
    assert len(report["unfinished"]) == 31
    assert all(
        row["metrics"] is None and row["samples"] is None for row in report["cases"]
    )


@pytest.mark.parametrize("changed", [None, "source.json", "identity-config.json"])
def test_followup_evidence_preserves_frozen_source_and_generation_identity(
    tmp_path, changed
):
    from scripts.parity.matrix_report import summarize

    primary = tmp_path / "primary" / "prompt_injection"
    followup = tmp_path / "followup" / "prompt_injection"
    pins = {
        "source.json": dict(
            rows=251, ids=["source:0", "source:1"], parquet_sha256="pinned"
        ),
        "identity-config.json": dict(
            samples=500, execution_mode="compiled", answer="pinned-answer"
        ),
    }
    for directory in (primary, followup):
        directory.mkdir(parents=True)
        for name, value in pins.items():
            (directory / name).write_text(json.dumps(value))
    if changed:
        (followup / changed).write_text(json.dumps({**pins[changed], "changed": True}))
        with pytest.raises(ValueError, match="replacement changes pinned"):
            summarize(primary.parent, {"prompt_injection": followup})
    else:
        report = summarize(primary.parent, {"prompt_injection": followup})
        row = next(row for row in report["cases"] if row["case"] == "prompt_injection")
        assert row["evidence_root"] == str(followup)
        assert row["superseded_primary_root"] == str(primary)
        assert (
            row["status"] == "missing"
        )  # Matching pins cannot substitute for a real report.
    with pytest.raises(ValueError, match="Unknown replacement case"):
        summarize(primary.parent, {"not-a-case": followup})


def test_native_response_observer_preserves_hook_and_failure_evidence(
    tmp_path, monkeypatch
):
    from typing import Any

    from inspect_ai.model import get_model

    from scripts.parity.capture import capture_judge_responses

    model = get_model("mockllm/capture")
    api: Any = model.api
    observed = []
    monkeypatch.setattr(api, "on_response", observed.append, raising=False)
    original = api.on_response
    response = {"id": "real-response-id", "prompt_token_ids": [2, 2, 106]}
    path = tmp_path / "responses.json"
    with pytest.raises(RuntimeError, match="later scoring failed"):
        with capture_judge_responses(model, path):
            api.on_response(response)
            raise RuntimeError("later scoring failed")
    assert api.on_response is original
    assert observed == [response]
    assert json.loads(path.read_text()) == {"real-response-id": response}


def test_ordered_samples_preserve_objects_and_sequence():
    import asyncio
    from types import SimpleNamespace

    from scripts.parity.ordered import ordered_samples

    samples = [object(), object(), object()]
    observed = []

    async def run():
        provider = ordered_samples(cast("Any", SimpleNamespace(samples=samples)))

        async def visit(index):
            async with provider(index) as sample:
                assert sample is samples[index]
                observed.append(index)
                await asyncio.sleep(0)
                observed.append(index)

        await asyncio.gather(visit(2), visit(0), visit(1))

    asyncio.run(run())
    assert observed == [0, 0, 1, 1, 2, 2]


def test_inspect_score_accepts_ordered_saved_sample_provider(tmp_path):
    import asyncio

    from inspect_ai import Task, score_async
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import Score, accuracy, scorer
    from inspect_ai.solver import generate

    from scripts.parity.ordered import ordered_samples

    seen = []

    @scorer(metrics=[accuracy()])
    def observe():
        async def score(state, target):
            seen.append(state.sample_id)
            await asyncio.sleep(0)
            return Score(value=1)

        return score

    task = Task(
        dataset=[Sample(id=i, input=f"Question {i}") for i in range(3)],
        solver=generate(),
        scorer=observe(),
    )
    log = inspect_eval(
        task,
        model="mockllm/ordered",
        score=False,
        max_samples=1,
        log_dir=str(tmp_path),
        display="none",
    )[0]
    result = asyncio.run(
        score_async(
            log, task.scorer, samples=ordered_samples(log), copy=False, display="none"
        )
    )
    assert seen == [0, 1, 2]
    assert result.samples is not None
    assert result.results is not None
    assert all(sample.scores for sample in result.samples)
    assert result.results.scores[0].metrics["accuracy"].value == 1


def test_original_messages_ignore_generated_ids_only():
    from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser

    from scripts.parity.report_matrix import input_content

    original: list[ChatMessage] = [
        ChatMessageSystem(content="S"),
        ChatMessageUser(content="Q\n"),
    ]
    replayed: list[ChatMessage] = [
        ChatMessageSystem(content="S"),
        ChatMessageUser(content="Q\n"),
    ]
    assert original != replayed
    assert input_content(original) == input_content(replayed)
    replayed[1].content = "Q"
    assert input_content(original) != input_content(replayed)


@pytest.mark.parametrize("value,expected", [("C", True), ("I", False), ("N", None)])
def test_ccpc_comparator_uses_documented_labels(value, expected):
    from scripts.parity.report_matrix import ccpc_verdict

    assert ccpc_verdict(value) is expected
    with pytest.raises(KeyError):
        ccpc_verdict("unrecognized")


def test_matrix_retains_failed_phase_without_fabricating_scores(tmp_path):
    from scripts.parity.matrix_report import summarize

    (tmp_path / "phases-judge.json").write_text(
        json.dumps(
            [dict(cases=["bbq-gender-bias", "medhallu"], phase="judge", returncode=1)]
        )
    )
    report = summarize(tmp_path)
    failed = [row for row in report["cases"] if row["phase_failures"]]
    assert {row["case"] for row in failed} == {"bbq:gender:bias", "medhallu"}
    assert all(row["metrics"] is None and row["samples"] is None for row in failed)


@pytest.mark.parametrize(
    "missing", [None, "identical_raw_outputs", "identical_response_order"]
)
def test_matrix_requires_actual_judge_output_and_order_evidence(tmp_path, missing):
    from scripts.parity.matrix_report import summarize

    root = tmp_path / "halueval"
    root.mkdir()
    judge = dict(
        identical_multiset=True,
        identical_raw_outputs=True,
        identical_response_order=True,
    )
    if missing:
        judge.pop(missing)
    files = {
        "source.json": dict(rows=3000),
        "identity-config.json": dict(samples=500),
        "manifest.json": dict(cases=[dict(runs=dict(legacy={}, native={}))]),
        "strict-report.json": dict(
            name="halueval",
            status="compared",
            samples=500,
            errors=[],
            mismatches=[],
            metrics=[dict(name="accuracy", delta=0)],
            judge_boundary=judge,
        ),
    }
    for name, value in files.items():
        (root / name).write_text(json.dumps(value))
    report = summarize(tmp_path)
    row = next(row for row in report["cases"] if row["case"] == "halueval")
    assert row["status"] == ("incomplete" if missing else "exact")
    assert report["complete"] is False  # The other 30 cases still lack evidence.


def test_ordered_samples_follow_pinned_ids_without_rewriting_log():
    import asyncio
    from types import SimpleNamespace

    from scripts.parity.ordered import ordered_samples

    samples = [SimpleNamespace(id=i, input=f"Q{i}") for i in range(3)]
    log = SimpleNamespace(samples=samples)
    observed = []

    async def run():
        provider = ordered_samples(cast("Any", log), [2, 0, 1])
        for index in range(3):
            async with provider(index) as sample:
                observed.append(sample.id)
                assert sample is samples[int(sample.id)]

    asyncio.run(run())
    assert observed == [2, 0, 1]
    assert [sample.id for sample in log.samples] == [0, 1, 2]
    with pytest.raises(ValueError, match="uniquely match"):
        ordered_samples(cast("Any", log), [2, 2, 1])


# A judge snapshot of the repository parity.yaml names (HF cache layout).
JUDGE = "/models/models--google--gemma-3-12b-it/snapshots/fixture"


def test_batch_checkpoints_completed_task_before_later_failure(tmp_path, monkeypatch):
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.log import read_eval_log
    from inspect_ai.model import get_model
    from inspect_ai.solver import generate

    from scripts.parity import qualify_native_batch
    from scripts.parity.compare import digest

    roots = []
    for family in ("halueval", "medhallu"):
        root = tmp_path / family
        (root / "dataset").mkdir(parents=True)
        parquet = root / "dataset/train.parquet"
        parquet.write_bytes(b"identity fixture; task factory supplies local samples")
        (root / "source.json").write_text(
            json.dumps(dict(ids=[family], parquet_sha256=digest(parquet)))
        )
        (root / "identity-config.json").write_text(
            json.dumps(
                dict(
                    case=family,
                    answer="answer",
                    judge=JUDGE,
                    samples=1,
                    judge_content_format="string",
                )
            )
        )
        roots.append(root)

        def factory(*, _family=family, **kwargs):
            return Task(dataset=[Sample(id=_family, input="Q")], solver=generate())

        monkeypatch.setattr(qualify_native_batch._registry, family, factory)
    model_calls = []

    def model(name, **kwargs):
        model_calls.append((name, kwargs))
        return get_model("mockllm/checkpoint")

    monkeypatch.setattr(qualify_native_batch, "get_model", model)
    evaluate = qualify_native_batch.eval_async
    calls = 0

    async def fail_later(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("later task failed")
        return await evaluate(*args, **kwargs)

    monkeypatch.setattr(qualify_native_batch, "eval_async", fail_later)
    monkeypatch.setattr(
        "sys.argv",
        [
            "qualify_native_batch",
            "answer",
            *map(str, roots),
            "--log-dir",
            str(tmp_path / "logs"),
            "--ordered",
        ],
    )
    with pytest.raises(RuntimeError, match="later task failed"):
        qualify_native_batch.main()
    saved = read_eval_log((roots[0] / "answer-log.txt").read_text())
    assert saved.status == "success"
    assert saved.samples is not None
    assert [sample.id for sample in saved.samples] == ["halueval"]
    assert not (roots[1] / "answer-log.txt").exists()
    assert [
        kwargs.get("chat_template_content_format")
        for name, kwargs in model_calls
        if name == f"vllm/{JUDGE}"
    ] == ["string", "string"]
    assert all(
        "chat_template_content_format" not in kwargs
        for name, kwargs in model_calls
        if name == "vllm/answer"
    )


@pytest.mark.parametrize(
    "family,scorer",
    [
        ("bbq", "bbq_scorer"),
        ("halueval", "halueval_scorer"),
        ("xstest", "xstest_scorer"),
        ("orbench", "orbench_scorer"),
        ("prompt_injection", "prompt_injection_scorer"),
    ],
)
def test_report_distinguishes_completion_from_scorer_text(family, scorer):
    from scripts.parity.compare import field
    from scripts.parity.report_matrix import formatted_answer_field

    raw = "  answer\n"
    formatted = "answer"
    sample = dict(
        output=dict(completion=raw),
        scores={
            scorer: dict(
                answer="comply"
                if family in {"xstest", "prompt_injection"}
                else formatted,
                metadata=dict(llm_answer=formatted, formatted_answer=formatted),
            )
        },
    )
    assert field(sample, "output.completion") == raw
    assert field(sample, formatted_answer_field(family, scorer)) == formatted
    assert raw != formatted


@pytest.mark.parametrize(
    "judge",
    [
        "google/gemma-3-12b-it",
        "/m/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80",
        "/models/gemma-3-12b-it",
    ],
)
def test_runners_accept_the_parity_judge(judge):
    from scripts.parity.parity_models import check_judge

    check_judge("bbq", judge)


@pytest.mark.parametrize(
    "family,judge",
    [
        ("bbq", "/m/models--google--gemma-4-12B-it/snapshots/x"),
        ("bbq", "google/gemma-4-12B-it"),
        ("ccpc", "google/gemma-3-12b-it"),
    ],
)
def test_runners_reject_unqualified_judges(family, judge):
    from scripts.parity.parity_models import check_judge

    with pytest.raises(ValueError, match="qualified with"):
        check_judge(family, judge)


@pytest.mark.parametrize("family", ["bbq", "ccpc", "xstest"])
def test_runner_settings_come_only_from_parity_yaml(family):
    import yaml

    from llm_behavior_eval.common.parity import parity_config_path
    from scripts.parity.parity_models import (
        CAPTURE_BODY,
        answer_settings,
        grader_settings,
    )

    raw = yaml.safe_load(parity_config_path(f"llm_behavior_eval.{family}").read_text())
    answer = answer_settings(family, cache_salt="llm169:case")
    body = answer.pop("extra_body")
    assert answer.pop("max_connections") == 1
    assert answer == {
        k: v for k, v in raw["generate_config"].items() if k != "extra_body"
    }
    assert body == {
        **raw["generate_config"]["extra_body"],
        **CAPTURE_BODY,
        "cache_salt": "llm169:case",
    }
    grader = grader_settings(family).model_dump(exclude_none=True)
    assert grader.pop("max_connections") == 1
    grader_body = grader.pop("extra_body")
    expected = dict(raw["model_roles"]["grader"]["config"])
    assert grader_body == {**expected.pop("extra_body"), **CAPTURE_BODY}
    assert grader == expected


def test_replay_ignores_only_capture_fields_and_concurrency():
    from inspect_ai.model import GenerateConfig

    from scripts.parity.replay_redesign import settings

    recorded = GenerateConfig(
        max_tokens=32,
        temperature=0,
        seed=42,
        max_connections=1,
        extra_body={
            "return_token_ids": True,
            "cache_salt": "llm169:bbq:gender:bias",
            "add_special_tokens": True,
        },
    )
    current = GenerateConfig(
        max_tokens=32, temperature=0, seed=42, extra_body={"add_special_tokens": True}
    )
    assert settings(recorded) == settings(current)
    assert settings(recorded) != settings(current.model_copy(update={"seed": 0}))


def test_replay_request_identity_is_exact_and_ordered():
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser

    from scripts.parity.replay_redesign import request_key

    system, user = ChatMessageSystem(content="S"), ChatMessageUser(content="Q\n")
    assert request_key([system, user]) == request_key(
        [ChatMessageSystem(content="S"), ChatMessageUser(content="Q\n")]
    )
    assert request_key([system, user]) != request_key([user, system])
    assert request_key([user]) != request_key([ChatMessageUser(content="Q")])

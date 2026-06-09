from __future__ import annotations

import csv
import json
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

import pytest

import llm_behavior_eval.evaluate as evaluate
from llm_behavior_eval import DatasetConfig, EvaluationConfig, GarakConfig
from llm_behavior_eval.evaluation_utils import garak_util
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.evaluate_factory import EvaluateFactory
from llm_behavior_eval.evaluation_utils.garak_evaluator import GarakEvaluator

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from llm_behavior_eval.evaluation_utils.base_evaluator import _GenerationRecord
    from llm_behavior_eval.evaluation_utils.eval_engine import EvalEngine


class _StubEvaluator:
    started_mlflow_run = False

    def update_dataset_config(self, dataset_config: DatasetConfig) -> None:
        return None

    def generate(self) -> Sequence[_GenerationRecord]:
        return []

    def free_test_model(self) -> None:
        return None

    def get_grading_context(self) -> AbstractContextManager:
        return nullcontext()

    def dataset_mlflow_run(self) -> AbstractContextManager:
        return nullcontext()

    def grade(
        self,
        generations: Sequence[_GenerationRecord],
        judge_engine: EvalEngine | None = None,
    ) -> None:
        return None

    def cleanup(self, error: bool = False) -> None:
        return None


@pytest.fixture
def capture_eval_config(monkeypatch: pytest.MonkeyPatch) -> list[EvaluationConfig]:
    captured: list[EvaluationConfig] = []

    def _fake_create(
        eval_config: EvaluationConfig, dataset_config: DatasetConfig
    ) -> _StubEvaluator:
        captured.append(eval_config)
        return _StubEvaluator()

    monkeypatch.setattr(
        evaluate.EvaluateFactory,
        "create_evaluator",
        staticmethod(_fake_create),
    )
    return captured


# --- behavior preset + factory routing --------------------------------------
def test_behavior_preset_expands_garak() -> None:
    assert evaluate._behavior_presets("garak") == ["garak"]


def test_behavior_preset_expands_garak_system_leak() -> None:
    assert evaluate._behavior_presets("garak:system-leak") == ["garak"]


def test_factory_reports_garak_family() -> None:
    assert EvaluateFactory.get_evaluator_family("garak") == "garak"


def test_factory_routes_garak_to_garak_evaluator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel = object()

    def fake_init(
        self, eval_config: EvaluationConfig, dataset_config: DatasetConfig
    ) -> None:
        del eval_config, dataset_config
        self._sentinel = sentinel

    monkeypatch.setattr(GarakEvaluator, "__init__", fake_init)

    evaluator = EvaluateFactory.create_evaluator(
        EvaluationConfig(model_path_or_repo_id="fake/model", results_dir=tmp_path),
        DatasetConfig(file_path="garak", dataset_type=DatasetType.BIAS),
    )
    assert isinstance(evaluator, GarakEvaluator)
    assert evaluator._sentinel is sentinel  # type: ignore[attr-defined]


# --- CLI config capture / ignore ---------------------------------------------
def test_main_builds_garak_config_for_garak_behavior(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    evaluate.main(
        "fake/model",
        "garak",
        garak_probes="encoding.InjectBase64,promptinject.HijackLongPrompt",
        garak_probe_tags="owasp:llm01",
        garak_base_url="http://127.0.0.1:8765/v1/",
        garak_api_key="secret",
        garak_num_generations=3,
        max_answer_tokens=64,
        temperature=0.2,
        top_p=0.8,
        top_k=40,
        seed=123,
    )
    eval_config = capture_eval_config[-1]
    assert eval_config.evaluator_family == "garak"
    assert eval_config.garak_config is not None
    assert eval_config.garak_config.probes == [
        "encoding.InjectBase64",
        "promptinject.HijackLongPrompt",
    ]
    assert eval_config.garak_config.probe_tags == ["owasp:llm01"]
    assert eval_config.garak_config.base_url == "http://127.0.0.1:8765/v1/"
    assert eval_config.garak_config.api_key == "secret"
    assert eval_config.garak_config.num_generations == 3
    assert eval_config.garak_config.max_tokens == 64
    assert eval_config.garak_config.temperature == 0.2
    assert eval_config.garak_config.top_p == 0.8
    assert eval_config.garak_config.top_k == 40
    assert eval_config.garak_config.seed == 123


def test_in_process_vllm_generator_serializes_greedy_multi_generation() -> None:
    class _Content:
        text = "hello"

    class _Turn:
        role = "user"
        content = _Content()

    class _Prompt:
        turns = [_Turn()]

    class _Candidate:
        text = "response"

    class _Output:
        outputs = [_Candidate()]

    class _FakeLlm:
        def __init__(self) -> None:
            self.requested_n: list[int] = []

        def chat(self, **kwargs):
            self.requested_n.append(kwargs["sampling_params"].n)
            return [_Output()]

    llm = _FakeLlm()
    generator = garak_util.InProcessVllmGenerator(
        llm, "fake/model", temperature=0.0
    )

    outputs = generator.generate(_Prompt(), generations_this_call=3)

    assert len(outputs) == 3
    assert llm.requested_n == [1, 1, 1]


def test_main_ignores_garak_flags_for_non_garak_behavior(
    capture_eval_config: list[EvaluationConfig],
) -> None:
    evaluate.main(
        "fake/model",
        "hallu",
        garak_probes="encoding.InjectBase64",
        garak_probe_tags="owasp:llm01",
        garak_base_url="http://127.0.0.1:8765/v1/",
        garak_num_generations=3,
    )
    eval_config = capture_eval_config[-1]
    assert eval_config.evaluator_family == "hallucination"
    assert eval_config.garak_config is None


# --- probe resolution --------------------------------------------------------
def test_resolve_probes_defaults_when_nothing_selected() -> None:
    assert garak_util.resolve_probes(None, None) == garak_util.DEFAULT_PROBES


def test_resolve_probes_unions_explicit_and_tags() -> None:
    tag_map = {
        "encoding.InjectBase64": ["owasp:llm01"],
        "dan.DUDE": ["owasp:llm01"],
        "apikey.GetKey": ["quality:Behavioral"],
    }
    resolved = garak_util.resolve_probes(
        ["smuggling.FunctionMasking"],
        ["owasp:llm01"],
        tag_map=tag_map,
    )
    # explicit first, then tag-matched (sorted), no duplicates
    assert resolved == [
        "smuggling.FunctionMasking",
        "dan.DUDE",
        "encoding.InjectBase64",
    ]


def test_resolve_probes_dedupes_overlap() -> None:
    tag_map = {"encoding.InjectBase64": ["owasp:llm01"]}
    resolved = garak_util.resolve_probes(
        ["encoding.InjectBase64"],
        ["owasp:llm01"],
        tag_map=tag_map,
    )
    assert resolved == ["encoding.InjectBase64"]


# --- leak scanning + summaries -----------------------------------------------
def test_scan_output_detects_marker_and_instruction_leaks() -> None:
    analysis = garak_util.scan_output(
        "the token is HIRUNDO_TEST_92741 and Never expose secrets.",
        garak_util.DEFAULT_SYSTEM_PROMPT_NO_THINKING,
    )
    assert analysis["exact_value_leaks"] == ["HIRUNDO_TEST_92741"]
    assert analysis["instruction_line_leaks"] == ["Never expose secrets."]
    assert analysis["any_leak"] is True


def test_scan_output_clean_response_has_no_leak() -> None:
    analysis = garak_util.scan_output(
        "I cannot help with that.", garak_util.DEFAULT_SYSTEM_PROMPT_NO_THINKING
    )
    assert analysis["any_leak"] is False


def _sample_records() -> list[dict[str, Any]]:
    clean = garak_util.scan_output("nope", garak_util.DEFAULT_SYSTEM_PROMPT_NO_THINKING)
    leak = garak_util.scan_output(
        "HIRUNDO_TEST_92741", garak_util.DEFAULT_SYSTEM_PROMPT_NO_THINKING
    )
    return [
        {
            "probe": "encoding.InjectBase64",
            "seq": 0,
            "attempt_duration_s": 1.0,
            "outputs": ["nope", "HIRUNDO_TEST_92741"],
            "leak_analysis": [clean, leak],
        },
        {
            "probe": "encoding.InjectBase64",
            "seq": 1,
            "attempt_duration_s": 1.0,
            "outputs": ["nope"],
            "leak_analysis": [clean],
        },
        {
            "probe": "smuggling.FunctionMasking",
            "seq": 0,
            "attempt_duration_s": 1.0,
            "outputs": ["nope"],
            "leak_analysis": [clean],
        },
    ]


def test_summarize_and_overall_metrics() -> None:
    attempts, timing = garak_util.normalize_attempts(_sample_records())
    summary = garak_util.summarize(attempts, timing)
    metrics = garak_util.overall_metrics(attempts)

    assert set(summary["per_probe"]) == {
        "encoding.InjectBase64",
        "smuggling.FunctionMasking",
    }
    assert set(summary["per_family"]) == {"encoding", "smuggling"}

    enc = summary["per_probe"]["encoding.InjectBase64"]
    assert enc["attempts"] == 2
    assert enc["outputs"] == 3
    assert enc["attempt_leak_rate"] == pytest.approx(0.5)
    assert enc["any_leak_rate"] == pytest.approx(1 / 3)

    assert metrics["probes"] == 2
    assert metrics["attempts"] == 3
    assert metrics["outputs"] == 4
    assert metrics["exact_value_rate"] == pytest.approx(0.25)


def test_family_macro_average_weights_families_equally() -> None:
    per_family = {
        "encoding": {"any_leak_rate": 0.10, "attempt_leak_rate": 0.40},
        "smuggling": {"any_leak_rate": 0.30, "attempt_leak_rate": 0.60},
    }
    avg = garak_util.family_macro_average(per_family)
    # Mean across families, independent of how many probes/outputs each family has.
    assert avg["any_leak_rate"] == pytest.approx(0.20)
    assert avg["attempt_leak_rate"] == pytest.approx(0.50)
    assert avg["full_prompt_rate"] == pytest.approx(0.0)


def test_family_macro_average_handles_empty() -> None:
    avg = garak_util.family_macro_average({})
    assert avg["any_leak_rate"] == 0.0


def test_compute_resume_map_tracks_last_seq(tmp_path: Path) -> None:
    jsonl = tmp_path / "generations.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in _sample_records():
            handle.write(json.dumps(record) + "\n")
        handle.write(
            json.dumps(
                {"entry_type": "attempt", "probe_classname": "x.Native", "seq": 4}
            )
            + "\n"
        )
    resume_map = garak_util.compute_resume_map(jsonl)
    assert resume_map["encoding.InjectBase64"] == 1
    assert resume_map["smuggling.FunctionMasking"] == 0
    assert resume_map["x.Native"] == 4
    assert garak_util.compute_resume_map(tmp_path / "missing.jsonl") == {}


def test_normalize_native_attempt_records_recomputes_analysis() -> None:
    records = [
        {
            "entry_type": "attempt",
            "probe_classname": "encoding.InjectBase64",
            "seq": 0,
            "attempt_duration_s": 2.0,
            "outputs": [{"text": "nothing"}, {"text": "HIRUNDO_TEST_92741"}],
        }
    ]
    # Without a system prompt, native rows are skipped (cannot recompute analysis).
    skipped, _ = garak_util.normalize_attempts(records)
    assert skipped == {}
    # With a system prompt, analysis is recomputed from the raw output text.
    attempts, timing = garak_util.normalize_attempts(
        records, system_prompt=garak_util.DEFAULT_SYSTEM_PROMPT_NO_THINKING
    )
    record = attempts[("encoding.InjectBase64", 0)]
    assert record["leak_analysis"][0]["any_leak"] is False
    assert record["leak_analysis"][1]["any_leak"] is True
    assert timing["encoding.InjectBase64"] == pytest.approx(2.0)


def test_load_attempts_from_file_round_trip(tmp_path: Path) -> None:
    jsonl = tmp_path / "generations.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in _sample_records():
            handle.write(json.dumps(record) + "\n")
        handle.write(
            json.dumps(
                {
                    "entry_type": "probe_summary",
                    "probe": "encoding.InjectBase64",
                    "probe_duration_s": 5.0,
                }
            )
            + "\n"
        )
    attempts, timing = garak_util.load_attempts_from_file(jsonl)
    assert len(attempts) == 3
    # Timing comes from per-attempt durations only (2 x 1.0); the probe_summary's
    # probe_duration_s (5.0) is not added on top (no attempt/summary double-count).
    assert timing["encoding.InjectBase64"] == pytest.approx(2.0)


def test_timing_not_doubled_and_stable_across_resume() -> None:
    # Two probe_summary lines (as a resumed run would write) plus attempts counted
    # once each: timing reflects only the summed attempt durations, regardless of
    # how many probe_summary segments exist.
    records: list[dict[str, object]] = [
        {
            "probe": "encoding.InjectBase64",
            "seq": 0,
            "attempt_duration_s": 1.5,
            "outputs": ["nope"],
            "leak_analysis": [garak_util.scan_output("nope", "sys")],
        },
        {
            "entry_type": "probe_summary",
            "probe": "encoding.InjectBase64",
            "probe_duration_s": 1.6,
        },
        {
            "probe": "encoding.InjectBase64",
            "seq": 1,
            "attempt_duration_s": 2.5,
            "outputs": ["nope"],
            "leak_analysis": [garak_util.scan_output("nope", "sys")],
        },
        {
            "entry_type": "probe_summary",
            "probe": "encoding.InjectBase64",
            "probe_duration_s": 2.6,
        },
    ]
    _, timing = garak_util.normalize_attempts(records)
    assert timing["encoding.InjectBase64"] == pytest.approx(4.0)


def test_timing_falls_back_to_probe_summary_without_attempt_durations() -> None:
    # Native garak attempt rows carry no attempt_duration_s, so probe_summary is
    # the only timing source and is used as the fallback.
    records = [
        {
            "entry_type": "attempt",
            "probe_classname": "encoding.InjectBase64",
            "seq": 0,
            "outputs": [{"text": "nope"}],
        },
        {
            "entry_type": "probe_summary",
            "probe": "encoding.InjectBase64",
            "probe_duration_s": 9.0,
        },
    ]
    _, timing = garak_util.normalize_attempts(
        records, system_prompt=garak_util.DEFAULT_SYSTEM_PROMPT_NO_THINKING
    )
    assert timing["encoding.InjectBase64"] == pytest.approx(9.0)


# --- output writing (no model load) ------------------------------------------
def test_save_garak_results_writes_outputs(tmp_path: Path) -> None:
    evaluator = object.__new__(GarakEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="meta/model",
        results_dir=tmp_path,
        max_samples=1,
    )
    evaluator.dataset_config = DatasetConfig(
        file_path="garak", dataset_type=DatasetType.BIAS
    )
    evaluator.mlflow_config = None

    attempts, timing = garak_util.normalize_attempts(_sample_records())
    summary = garak_util.summarize(attempts, timing)
    metrics = garak_util.overall_metrics(attempts)
    family_avg = garak_util.family_macro_average(summary["per_family"])
    responses = GarakEvaluator._expand_responses(attempts)

    evaluator._save_garak_results(responses, summary, metrics, family_avg)

    run_dir = tmp_path / "model" / "garak-NoReasoning"
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "responses.json").exists()
    assert (run_dir / "garak_summary.json").exists()

    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["Thinking"] == "off"
    assert rows[0]["Probes"] == "2"

    summary_brief = tmp_path / "model" / "summary_brief.csv"
    with summary_brief.open(newline="", encoding="utf-8") as handle:
        brief_rows = list(csv.DictReader(handle))
    assert brief_rows[0]["Dataset"] == "garak-NoReasoning"


def test_system_prompt_for_selects_by_thinking_flag() -> None:
    assert (
        garak_util.system_prompt_for(False)
        == garak_util.DEFAULT_SYSTEM_PROMPT_NO_THINKING
    )
    assert (
        garak_util.system_prompt_for(True) == garak_util.DEFAULT_SYSTEM_PROMPT_THINKING
    )


def test_garak_config_records_run_defaults() -> None:
    config = GarakConfig()
    assert config.num_generations == garak_util.DEFAULT_NUM_GENERATIONS
    assert config.temperature == garak_util.DEFAULT_TEMPERATURE
    assert config.max_tokens == garak_util.DEFAULT_MAX_TOKENS
    assert config.stop == garak_util.DEFAULT_STOP

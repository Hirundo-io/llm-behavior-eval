import hashlib
import json
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pandas as pd
import pytest
import torch
from datasets import Dataset
from pydantic import ValidationError
from transformers.data.data_collator import default_data_collator

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from llm_behavior_eval.evaluation_utils.eval_engine import EvalEngine

from llm_behavior_eval import DatasetConfig, EvaluationConfig, evaluate
from llm_behavior_eval.evaluation_utils import censorship_utils
from llm_behavior_eval.evaluation_utils.censorship_utils import (
    CCPC_BENCHMARK_CONFIG,
    CCPC_DATASET_CONFIG,
    CCPC_DATASET_ID,
    CCPC_DATASET_REPOSITORY,
    CCPC_DATASET_REVISION,
    CCPC_DATASET_SPLIT,
    CCPC_EXPECTED_ROWS,
    CCPC_JUDGE_MODEL,
    CCPC_JUDGE_REPETITION_PENALTY,
    CCPC_MODEL_REPETITION_PENALTY,
    CensorshipStatus,
    ResolvedCensorshipBenchmark,
    load_censorship_benchmark,
    parse_censorship_judgment,
)
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.evaluate_factory import EvaluateFactory
from llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator import (
    CensorshipResponse,
    FreeTextCensorshipEvaluator,
    _CensorshipGenerationRecord,
)
from llm_behavior_eval.evaluation_utils.sampling_config import SamplingConfig

_HISTORICAL_DATASET_CONFIG = DatasetConfig(
    file_path=CCPC_DATASET_ID, dataset_type=DatasetType.BIAS
)


def _local_dataset_config(
    path: Path, expected_row_count: int, expected_sha256: str | None = None
) -> DatasetConfig:
    """Build a DatasetConfig selecting an explicit local CCPC cohort.

    Args:
        path: Local JSONL file to load.
        expected_row_count: Required explicit row-count contract.
        expected_sha256: Optional explicit SHA-256 contract.

    Returns:
        A DatasetConfig routed to the local-cohort loader.
    """
    return DatasetConfig(
        file_path=str(path),
        dataset_id=CCPC_DATASET_ID,
        dataset_type=DatasetType.BIAS,
        ccpc_source_mode="local",
        expected_row_count=expected_row_count,
        expected_sha256=expected_sha256,
    )


def _write_local_jsonl(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    """Write rows as a JSONL fixture file, one JSON object per line.

    Args:
        tmp_path: Temporary directory supplied by pytest.
        rows: Row dicts to serialize in order.

    Returns:
        Path to the written JSONL file.
    """
    path = tmp_path / "local_ccpc.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _local_rows(size: int, **columns: Sequence[object]) -> list[dict[str, object]]:
    """Build deterministic local-cohort row fixtures (benchmark_id identity).

    Args:
        size: Number of rows to create.
        **columns: Column values that replace or extend the default columns.

    Returns:
        A list of row dicts matching the local CCPC-500-style schema.
    """
    base: dict[str, Sequence[object]] = {
        "question": [f"question-{index}" for index in range(size)],
        "benchmark_id": [f"bench-{index}" for index in range(size)],
    }
    base.update(columns)
    return [
        {key: values[index] for key, values in base.items()} for index in range(size)
    ]


def _benchmark(size: int = CCPC_EXPECTED_ROWS, **columns: Sequence[object]) -> Dataset:
    """Build a deterministic in-memory benchmark fixture.

    Args:
        size: Number of benchmark rows to create.
        **columns: Column values that replace or extend the default columns.

    Returns:
        A Hugging Face dataset with the requested fixture contents.
    """
    data: dict[str, Sequence[object]] = {
        "question": [f"question-{index}" for index in range(size)],
        "source_group_id": [f"group-{index}" for index in range(size)],
    }
    data.update(columns)
    return Dataset.from_dict(data)


def _patch_load_dataset(
    monkeypatch: pytest.MonkeyPatch, dataset: Dataset
) -> dict[str, object]:
    """Patch the dataset loader and record its arguments.

    Args:
        monkeypatch: Pytest patching helper.
        dataset: Dataset the patched loader should return.

    Returns:
        Mutable mapping populated with loader arguments after the call.
    """
    captured: dict[str, object] = {}

    def fake_load_dataset(path: str, **kwargs: object) -> Dataset:
        """Record loader arguments and return the fixture dataset.

        Args:
            path: Dataset repository passed by the production loader.
            **kwargs: Keyword arguments passed by the production loader.

        Returns:
            The dataset fixture supplied to the enclosing helper.
        """
        captured.update(path=path, **kwargs)
        return dataset

    monkeypatch.setattr(censorship_utils, "load_dataset", fake_load_dataset)
    return captured


def test_loads_exact_pinned_dataset_in_published_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify all immutable dataset pins and row order reach the loader.

    Args:
        monkeypatch: Pytest patching helper.

    Returns:
        None.
    """
    expected = _benchmark()
    captured = _patch_load_dataset(monkeypatch, expected)

    resolved = load_censorship_benchmark(_HISTORICAL_DATASET_CONFIG, "token")

    assert resolved.dataset["source_group_id"] == expected["source_group_id"]
    assert resolved.identity_field == "source_group_id"
    assert resolved.row_ids == expected["source_group_id"]
    assert resolved.benchmark_config == CCPC_BENCHMARK_CONFIG
    assert captured == {
        "path": CCPC_DATASET_REPOSITORY,
        "name": CCPC_DATASET_CONFIG,
        "split": CCPC_DATASET_SPLIT,
        "revision": CCPC_DATASET_REVISION,
        "token": "token",
        "trust_remote_code": False,
    }
    assert _HISTORICAL_DATASET_CONFIG.ccpc_source_mode is None


def test_legacy_historical_ccpc_run_config_without_source_mode_is_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reuse a pre-source-mode historical CCPC run_config.json.

    Args:
        tmp_path: Temporary results directory supplied by pytest.
        monkeypatch: Pytest patching helper.

    Returns:
        None.
    """
    evaluator = FreeTextCensorshipEvaluator.__new__(FreeTextCensorshipEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model", results_dir=tmp_path
    )
    evaluator.dataset_config = _HISTORICAL_DATASET_CONFIG
    evaluator._ccpc_benchmark_config = CCPC_BENCHMARK_CONFIG.copy()
    run_config = evaluator._current_run_config()
    assert "ccpc_source_mode" not in run_config["dataset_config"]
    run_config_path = evaluator.run_config_path()
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.base_evaluator.typer.prompt",
        lambda *_args, **_kwargs: pytest.fail("matching historical CCPC prompted"),
    )

    evaluator._ensure_run_configuration_allowed()


def test_local_ccpc_run_config_does_not_match_historical(
    tmp_path: Path,
) -> None:
    """Keep explicit local CCPC distinct from historical CCPC run identity.

    Args:
        tmp_path: Temporary results directory supplied by pytest.

    Returns:
        None.
    """
    historical = FreeTextCensorshipEvaluator.__new__(FreeTextCensorshipEvaluator)
    historical.eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model", results_dir=tmp_path
    )
    historical.dataset_config = _HISTORICAL_DATASET_CONFIG
    historical._ccpc_benchmark_config = CCPC_BENCHMARK_CONFIG.copy()
    historical.run_config_path().write_text(
        json.dumps(historical._current_run_config()), encoding="utf-8"
    )

    local_path = _write_local_jsonl(tmp_path, _local_rows(500))
    local = FreeTextCensorshipEvaluator.__new__(FreeTextCensorshipEvaluator)
    local.eval_config = historical.eval_config
    local.dataset_config = _local_dataset_config(local_path, expected_row_count=500)
    local._ccpc_benchmark_config = load_censorship_benchmark(
        local.dataset_config
    ).benchmark_config

    with pytest.raises(RuntimeError, match="--replace-existing-output"):
        local._ensure_run_configuration_allowed()


def test_loads_explicit_local_cohort_in_file_order(tmp_path: Path) -> None:
    """Verify an explicit local CCPC cohort loads all rows, unshuffled.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        None.
    """
    rows = _local_rows(500)
    path = _write_local_jsonl(tmp_path, rows)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset_config = _local_dataset_config(path, expected_row_count=500)

    resolved = load_censorship_benchmark(dataset_config)

    assert resolved.identity_field == "benchmark_id"
    assert resolved.dataset["question"] == [row["question"] for row in rows]
    assert resolved.row_ids == [row["benchmark_id"] for row in rows]
    assert resolved.benchmark_config["expected_rows"] == 500
    assert resolved.benchmark_config["dataset_path"] == str(path)
    assert resolved.benchmark_config["dataset_sha256"] == sha256
    assert resolved.benchmark_config["dataset_repository"] is None
    assert resolved.benchmark_config["dataset_revision"] is None
    assert resolved.benchmark_config["identity_field"] == "benchmark_id"


def test_explicit_local_cohort_named_like_preset_stays_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify an explicit local ``chinese_censorship`` path never loads HF."""
    rows = _local_rows(500)
    path = tmp_path / CCPC_DATASET_ID
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    historical_calls: list[tuple[object, ...]] = []

    def fake_load_dataset(*args: object, **kwargs: object) -> Dataset:
        historical_calls.append((*args, kwargs))
        return _benchmark()

    monkeypatch.setattr(censorship_utils, "load_dataset", fake_load_dataset)
    monkeypatch.chdir(tmp_path)
    dataset_config = _local_dataset_config(
        Path(CCPC_DATASET_ID), expected_row_count=500, expected_sha256=sha256
    )

    resolved = load_censorship_benchmark(dataset_config)

    assert historical_calls == []
    assert len(resolved.dataset) == 500
    assert resolved.identity_field == "benchmark_id"
    assert resolved.row_ids == [row["benchmark_id"] for row in rows]
    assert resolved.benchmark_config["expected_rows"] == 500
    assert resolved.benchmark_config["dataset_sha256"] == sha256


@pytest.mark.parametrize("actual_size", [499, 501])
def test_local_cohort_rejects_wrong_row_count(tmp_path: Path, actual_size: int) -> None:
    """Verify a local cohort that doesn't match its expected 500 rows fails closed.

    Args:
        tmp_path: Temporary directory supplied by pytest.
        actual_size: Actual row count written to the fixture file (499 or 501).

    Returns:
        None.
    """
    path = _write_local_jsonl(tmp_path, _local_rows(actual_size))
    dataset_config = _local_dataset_config(path, expected_row_count=500)

    with pytest.raises(ValueError, match="exactly 500 rows"):
        load_censorship_benchmark(dataset_config)


def test_local_cohort_requires_expected_row_count(tmp_path: Path) -> None:
    """Verify a local cohort's row count is never self-validating.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        None.
    """
    path = _write_local_jsonl(tmp_path, _local_rows(500))

    with pytest.raises(ValidationError, match="requires expected_row_count"):
        DatasetConfig(
            file_path=str(path),
            dataset_id=CCPC_DATASET_ID,
            dataset_type=DatasetType.BIAS,
            ccpc_source_mode="local",
        )


def test_local_cohort_rejects_duplicate_benchmark_id(tmp_path: Path) -> None:
    """Verify duplicate benchmark_id values fail the local cohort's identity check.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        None.
    """
    rows = _local_rows(500, benchmark_id=["duplicate"] * 500)
    path = _write_local_jsonl(tmp_path, rows)
    dataset_config = _local_dataset_config(path, expected_row_count=500)

    with pytest.raises(ValueError, match="must be unique"):
        load_censorship_benchmark(dataset_config)


def test_local_cohort_rejects_missing_benchmark_id_column(tmp_path: Path) -> None:
    """Verify a local file missing benchmark_id fails, without silently mapping
    another column into that identity.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        None.
    """
    rows: list[dict[str, object]] = [
        {"question": f"question-{index}", "source_group_id": f"group-{index}"}
        for index in range(500)
    ]
    path = _write_local_jsonl(tmp_path, rows)
    dataset_config = _local_dataset_config(path, expected_row_count=500)

    with pytest.raises(ValueError, match="must include"):
        load_censorship_benchmark(dataset_config)


def test_local_cohort_validates_sha256_when_supplied(tmp_path: Path) -> None:
    """Verify a mismatched explicit SHA-256 contract fails closed.

    Args:
        tmp_path: Temporary directory supplied by pytest.

    Returns:
        None.
    """
    path = _write_local_jsonl(tmp_path, _local_rows(500))
    dataset_config = _local_dataset_config(
        path, expected_row_count=500, expected_sha256="0" * 64
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_censorship_benchmark(dataset_config)


def test_current_run_config_omits_ignored_max_samples(tmp_path: Path) -> None:
    """Verify CCPC's immutable cohort ignores max_samples in run identity.

    Args:
        tmp_path: Temporary results directory supplied by pytest.

    Returns:
        None.
    """
    run_configs = []
    for max_samples in (1, CCPC_EXPECTED_ROWS):
        evaluator = FreeTextCensorshipEvaluator.__new__(FreeTextCensorshipEvaluator)
        evaluator.eval_config = EvaluationConfig(
            model_path_or_repo_id="fake/model",
            results_dir=tmp_path,
            max_samples=max_samples,
        )
        evaluator.dataset_config = DatasetConfig(
            file_path=CCPC_DATASET_ID,
            dataset_type=DatasetType.BIAS,
        )
        evaluator._ccpc_benchmark_config = CCPC_BENCHMARK_CONFIG.copy()
        run_configs.append(evaluator._current_run_config())

    assert run_configs[0] == run_configs[1]
    assert "max_samples" not in run_configs[0]["evaluation_config"]


def test_run_provenance_reflects_the_actual_local_cohort(tmp_path: Path) -> None:
    """Verify run_config never claims the historical HF source for a local run.

    Args:
        tmp_path: Temporary results directory supplied by pytest.

    Returns:
        None.
    """
    local_path = _write_local_jsonl(tmp_path, _local_rows(500))
    resolved = load_censorship_benchmark(
        _local_dataset_config(local_path, expected_row_count=500)
    )
    evaluator = FreeTextCensorshipEvaluator.__new__(FreeTextCensorshipEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model", results_dir=tmp_path
    )
    evaluator.dataset_config = _local_dataset_config(local_path, expected_row_count=500)
    evaluator._ccpc_benchmark_config = resolved.benchmark_config

    provenance = evaluator._current_run_config()["ccpc_benchmark"]

    assert provenance != CCPC_BENCHMARK_CONFIG
    assert provenance["dataset_repository"] is None
    assert provenance["dataset_revision"] is None
    assert provenance["dataset_path"] == str(local_path)
    assert provenance["dataset_sha256"] is not None
    assert provenance["expected_rows"] == 500
    assert provenance["identity_field"] == "benchmark_id"


@pytest.mark.parametrize(
    ("dataset", "message"),
    [
        (_benchmark(extra=["x"] * CCPC_EXPECTED_ROWS), "columns must be exactly"),
        (_benchmark(CCPC_EXPECTED_ROWS - 1), "exactly 216 rows"),
        (
            _benchmark(
                source_group_id=["duplicate"] * CCPC_EXPECTED_ROWS,
            ),
            "must be unique",
        ),
        (
            _benchmark(question=[" "] * CCPC_EXPECTED_ROWS),
            "field 'question' must be a non-empty string",
        ),
        (
            _benchmark(source_group_id=[""] * CCPC_EXPECTED_ROWS),
            "field 'source_group_id' must be a non-empty string",
        ),
        (
            _benchmark(question=list(range(CCPC_EXPECTED_ROWS))),
            "field 'question' must be a non-empty string",
        ),
        (
            _benchmark(source_group_id=list(range(CCPC_EXPECTED_ROWS))),
            "field 'source_group_id' must be a non-empty string",
        ),
    ],
)
def test_rejects_invalid_dataset_contract(
    monkeypatch: pytest.MonkeyPatch, dataset: Dataset, message: str
) -> None:
    """Verify malformed benchmark schemas and values fail closed.

    Args:
        monkeypatch: Pytest patching helper.
        dataset: Invalid benchmark fixture.
        message: Expected validation-error fragment.

    Returns:
        None.
    """
    _patch_load_dataset(monkeypatch, dataset)
    with pytest.raises(ValueError, match=message):
        load_censorship_benchmark(_HISTORICAL_DATASET_CONFIG)


class _Tokenizer:
    name_or_path = "fake/model"

    def __init__(self) -> None:
        """Initialize an empty prompt log.

        Args:
            None.

        Returns:
            None.
        """
        self.prompts: list[str] = []

    def __call__(
        self, prompts: list[str], **kwargs: object
    ) -> dict[str, list[list[int]]]:
        """Tokenize prompts into deterministic one-token rows.

        Args:
            prompts: Rendered prompts to tokenize.
            **kwargs: Tokenizer options ignored by this fixture.

        Returns:
            Token identifiers and attention masks for each prompt.
        """
        del kwargs
        self.prompts = prompts
        return {
            "input_ids": [[index] for index in range(len(prompts))],
            "attention_mask": [[1] for _ in prompts],
        }


class _DatasetEngine:
    is_judge = False

    def set_dataset(self, dataset: Dataset) -> None:
        """Record the dataset supplied by the evaluator.

        Args:
            dataset: Prepared evaluation dataset.

        Returns:
            None.
        """
        self.dataset = dataset

    def get_batch_size(self) -> int:
        """Return the deterministic test batch size.

        Args:
            None.

        Returns:
            The batch size used to exercise a final partial batch.
        """
        return 31


def test_prepare_dataloader_never_shuffles_or_truncates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify row selection ignores max_samples and preserves source order.

    Args:
        monkeypatch: Pytest patching helper.
        tmp_path: Temporary results directory supplied by pytest.

    Returns:
        None.
    """
    benchmark = _benchmark()
    resolved = ResolvedCensorshipBenchmark(
        dataset=benchmark,
        identity_field="source_group_id",
        row_ids=list(benchmark["source_group_id"]),
        benchmark_config=CCPC_BENCHMARK_CONFIG.copy(),
    )
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator.load_censorship_benchmark",
        lambda dataset_config, token: resolved,
    )
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator.is_model_multimodal",
        lambda *args: False,
    )
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator.safe_apply_chat_template",
        lambda tokenizer, messages, **kwargs: messages[0]["content"],
    )
    evaluator = FreeTextCensorshipEvaluator.__new__(FreeTextCensorshipEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        max_samples=1,
        max_answer_tokens=8,
    )
    evaluator.dataset_config = DatasetConfig(
        file_path="chinese_censorship", dataset_type=DatasetType.BIAS
    )
    evaluator.tokenizer = cast("PreTrainedTokenizerBase", _Tokenizer())
    evaluator.trust_remote_code = False
    evaluator.eval_engine = cast("EvalEngine", _DatasetEngine())
    evaluator.data_collator = default_data_collator

    evaluator.prepare_dataloader()

    indices = [
        int(index)
        for batch in evaluator.eval_loader
        for index in batch["ccpc_row_index"].tolist()
    ]
    assert indices == list(range(CCPC_EXPECTED_ROWS))
    assert evaluator.benchmark_questions == benchmark["question"]
    assert evaluator.num_samples == CCPC_EXPECTED_ROWS


@pytest.mark.parametrize(
    ("text", "expected"),
    [("True", True), (" false \n", False), ("True because...", None), ("", None)],
)
def test_parse_censorship_judgment(text: str, expected: bool | None) -> None:
    """Verify the judge parser accepts only bare binary verdicts.

    Args:
        text: Raw judge text to parse.
        expected: Expected parsed verdict.

    Returns:
        None.
    """
    assert parse_censorship_judgment(text) is expected


class _RecordingEngine:
    is_judge = False

    def __init__(self) -> None:
        """Initialize an empty repetition-penalty log.

        Args:
            None.

        Returns:
            None.
        """
        self.penalties: list[float] = []

    def generate_answers(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sampling_config: SamplingConfig,
        repetition_penalty: float = 1.0,
    ) -> tuple[list[str], list[str | None]]:
        """Record the role-specific repetition penalty.

        Args:
            input_ids: Token identifiers used to size the deterministic result.
            attention_mask: Attention mask ignored by this fixture.
            sampling_config: Decoding settings ignored by this fixture.
            repetition_penalty: Repetition penalty supplied by the evaluator.

        Returns:
            Deterministic answers and successful finish reasons.
        """
        del attention_mask, sampling_config
        self.penalties.append(repetition_penalty)
        return ["answer"] * len(input_ids), ["stop"] * len(input_ids)


def test_model_and_judge_use_role_specific_repetition_penalties(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify model and judge calls use their frozen role-specific penalties.

    Args:
        monkeypatch: Pytest patching helper.
        tmp_path: Temporary results directory supplied by pytest.

    Returns:
        None.
    """
    evaluator = FreeTextCensorshipEvaluator.__new__(FreeTextCensorshipEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        judge_path_or_repo_id=CCPC_JUDGE_MODEL,
        sample_judge=False,
    )
    evaluator.dataset_config = DatasetConfig(
        file_path="chinese_censorship", dataset_type=DatasetType.BIAS
    )
    evaluator._ccpc_benchmark_config = CCPC_BENCHMARK_CONFIG.copy()
    engine = _RecordingEngine()
    evaluator.eval_engine = cast("EvalEngine", engine)
    evaluator.judge_tokenizer = cast("PreTrainedTokenizerBase", _Tokenizer())
    judge_penalties: list[float] = []

    def fake_run_judge_with_backoff(
        judge_engine: "EvalEngine",
        prompts: list[str],
        repetition_penalty: float = 1.0,
    ) -> list[list[dict[str, str]]]:
        """Record the explicit judge repetition penalty.
        Args:
            judge_engine: Judge engine ignored by this fixture.
            prompts: Judge prompts used to size the deterministic result.
            repetition_penalty: Repetition penalty supplied by the evaluator.
        Returns:
            Deterministic successful judge outputs.
        """
        del judge_engine
        judge_penalties.append(repetition_penalty)
        return [[{"generated_text": "False", "finish_reason": "stop"}] for _ in prompts]

    monkeypatch.setattr(evaluator, "prepare_judge_tokenizer", lambda: None)
    monkeypatch.setattr(evaluator, "_get_judge_tokenizer", lambda: _Tokenizer())
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator.safe_apply_chat_template",
        lambda tokenizer, messages: messages[-1]["content"],
    )
    monkeypatch.setattr(
        evaluator, "run_judge_with_backoff", fake_run_judge_with_backoff
    )

    evaluator._generate_model_answers(torch.tensor([[1]]), torch.tensor([[1]]))
    evaluator._grade_batch(cast("EvalEngine", engine), ["question"], ["answer"])

    assert engine.penalties == [CCPC_MODEL_REPETITION_PENALTY]
    assert judge_penalties == [CCPC_JUDGE_REPETITION_PENALTY]
    provenance = evaluator._current_run_config()["ccpc_benchmark"]
    assert provenance == CCPC_BENCHMARK_CONFIG
    assert provenance is not CCPC_BENCHMARK_CONFIG
    assert provenance["model_repetition_penalty"] == CCPC_MODEL_REPETITION_PENALTY
    assert provenance["judge_repetition_penalty"] == CCPC_JUDGE_REPETITION_PENALTY


def _response(
    index: int,
    status: CensorshipStatus,
    identity_field: str = "source_group_id",
    row_id: str | None = None,
) -> CensorshipResponse:
    """Build one persisted response fixture.

    Args:
        index: Benchmark row index.
        status: Exhaustive outcome status for the row.
        identity_field: Active cohort's native identity key.
        row_id: Identity value; defaults to ``group-{index}``.

    Returns:
        A response record matching the persisted schema.
    """
    verdict = status == CensorshipStatus.JUDGED_TRUE
    return {
        identity_field: row_id if row_id is not None else f"group-{index}",
        "question": f"question-{index}",
        "llm_answer": "answer",
        "finish_reason": "stop",
        "judge_finish_reason": "stop",
        "judge_raw": str(verdict),
        "judge_verdict": verdict
        if status in {CensorshipStatus.JUDGED_TRUE, CensorshipStatus.JUDGED_FALSE}
        else None,
        "status": status.value,
    }


@pytest.mark.parametrize("invalid_status", [None, *list(CensorshipStatus)[2:]])
def test_metric_valid_only_for_an_all_determinate_cohort(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invalid_status: CensorshipStatus | None,
) -> None:
    """Verify the primary metric requires 216 determinate verdicts.

    Args:
        monkeypatch: Pytest patching helper.
        tmp_path: Temporary results directory supplied by pytest.
        invalid_status: Optional non-determinate status for the first row.

    Returns:
        None.
    """
    evaluator = FreeTextCensorshipEvaluator.__new__(FreeTextCensorshipEvaluator)
    evaluator.identity_field = "source_group_id"
    evaluator.expected_rows = CCPC_EXPECTED_ROWS
    evaluator.benchmark_row_ids = [
        f"group-{index}" for index in range(CCPC_EXPECTED_ROWS)
    ]
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model", results_dir=tmp_path
    )
    monkeypatch.setattr(evaluator, "get_output_dir", lambda: tmp_path)
    monkeypatch.setattr(evaluator, "get_model_slug", lambda: "model")
    monkeypatch.setattr(evaluator, "get_dataset_slug", lambda: "chinese_censorship")
    monkeypatch.setattr(evaluator, "_append_summary_row", lambda *args: None)
    responses = [
        _response(
            index,
            invalid_status
            if index == 0 and invalid_status is not None
            else CensorshipStatus.JUDGED_TRUE
            if index == 0
            else CensorshipStatus.JUDGED_FALSE,
        )
        for index in range(CCPC_EXPECTED_ROWS)
    ]

    evaluator._save_results(responses)

    metrics = pd.read_csv(tmp_path / "metrics.csv").iloc[0]
    assert bool(metrics["Metric Valid"]) is (invalid_status is None)
    if invalid_status is None:
        assert metrics["Censorship rate (%) ⬇️"] == pytest.approx(100 / 216)
    else:
        assert pd.isna(metrics["Censorship rate (%) ⬇️"])


def test_metric_valid_uses_the_active_500_row_cohort_denominator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify a 500-row local cohort's metrics use 500, not the historical 216.

    Args:
        monkeypatch: Pytest patching helper.
        tmp_path: Temporary results directory supplied by pytest.

    Returns:
        None.
    """
    evaluator = FreeTextCensorshipEvaluator.__new__(FreeTextCensorshipEvaluator)
    evaluator.identity_field = "benchmark_id"
    evaluator.expected_rows = 500
    evaluator.benchmark_row_ids = [f"bench-{index}" for index in range(500)]
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model", results_dir=tmp_path
    )
    monkeypatch.setattr(evaluator, "get_output_dir", lambda: tmp_path)
    monkeypatch.setattr(evaluator, "get_model_slug", lambda: "model")
    monkeypatch.setattr(evaluator, "get_dataset_slug", lambda: "chinese_censorship")
    monkeypatch.setattr(evaluator, "_append_summary_row", lambda *args: None)
    responses = [
        _response(
            index,
            CensorshipStatus.JUDGED_TRUE
            if index == 0
            else CensorshipStatus.JUDGED_FALSE,
            identity_field="benchmark_id",
            row_id=f"bench-{index}",
        )
        for index in range(500)
    ]

    evaluator._save_results(responses)

    metrics = pd.read_csv(tmp_path / "metrics.csv").iloc[0]
    assert bool(metrics["Metric Valid"]) is True
    assert metrics["Expected Samples"] == 500
    assert metrics["Censorship rate (%) ⬇️"] == pytest.approx(100 / 500)


def test_save_results_rejects_an_unaccounted_cohort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify missing benchmark rows raise as an internal invariant violation.

    Args:
        monkeypatch: Pytest patching helper.
        tmp_path: Temporary results directory supplied by pytest.

    Returns:
        None.
    """
    evaluator = FreeTextCensorshipEvaluator.__new__(FreeTextCensorshipEvaluator)
    evaluator.identity_field = "source_group_id"
    evaluator.expected_rows = CCPC_EXPECTED_ROWS
    evaluator.benchmark_row_ids = [
        f"group-{index}" for index in range(CCPC_EXPECTED_ROWS)
    ]
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model", results_dir=tmp_path
    )
    monkeypatch.setattr(evaluator, "get_output_dir", lambda: tmp_path)
    responses = [
        _response(index, CensorshipStatus.JUDGED_FALSE)
        for index in range(CCPC_EXPECTED_ROWS - 1)
    ]

    with pytest.raises(ValueError, match="every source_group_id"):
        evaluator._save_results(responses)


def test_grading_preserves_raw_evidence_and_all_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify grading preserves raw evidence for every outcome category.

    Args:
        monkeypatch: Pytest patching helper.

    Returns:
        None.
    """
    evaluator = FreeTextCensorshipEvaluator.__new__(FreeTextCensorshipEvaluator)
    evaluator.identity_field = "source_group_id"
    generation = _CensorshipGenerationRecord(
        questions=[f"q{i}" for i in range(8)],
        row_ids=[f"g{i}" for i in range(8)],
        answers=[f"raw-{i}" for i in range(8)],
        finish_reasons=["stop", "stop", "length", None, "stop", "stop", "stop", "stop"],
    )
    monkeypatch.setattr(evaluator, "_format_answers", lambda answers: answers)
    monkeypatch.setattr(
        evaluator,
        "_grade_batch",
        lambda *args: (
            [True, False, True, False, None, None],
            ["True", "False", "truncated", "unknown", "malformed", ""],
            ["stop", "stop", "length", None, "stop", None],
            [False, False, False, False, False, True],
        ),
    )
    captured: list[CensorshipResponse] = []
    monkeypatch.setattr(evaluator, "_save_results", captured.extend)

    evaluator._grade_impl([generation], cast("EvalEngine", object()))

    assert [item["status"] for item in captured] == [
        status.value for status in CensorshipStatus
    ]
    assert captured[0]["llm_answer"] == "raw-0"
    assert captured[6]["judge_raw"] == "malformed"


@pytest.mark.parametrize(
    "behavior", ["chinese_censorship", "CHINESE_CENSORSHIP", " chinese_censorship "]
)
def test_cli_accepts_censorship_behavior(behavior: str) -> None:
    """Verify the CLI normalizes the dedicated censorship behavior alias.

    Args:
        behavior: Supported spelling of the behavior alias.

    Returns:
        None.
    """
    assert evaluate._behavior_presets(behavior) == [CCPC_DATASET_ID]


def test_cli_and_factory_route_to_the_dedicated_evaluator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify CLI and factory routing select the dedicated evaluator family.

    Args:
        monkeypatch: Pytest patching helper.
        tmp_path: Temporary results directory supplied by pytest.

    Returns:
        None.
    """
    captured: list[DatasetConfig] = []
    monkeypatch.setattr(
        evaluate.EvaluateFactory,
        "create_evaluator",
        lambda config, dataset: (
            captured.append(dataset)
            or SimpleNamespace(
                started_mlflow_run=False,
                update_dataset_config=lambda dataset: None,
                generate=lambda: [],
                free_test_model=lambda: None,
                dataset_mlflow_run=nullcontext,
                get_grading_context=nullcontext,
                grade=lambda *args: None,
                cleanup=lambda error=False: None,
            )
        ),
    )
    evaluate.main(
        "fake/model",
        "chinese_censorship",
        output_dir=tmp_path,
        judge_model=CCPC_JUDGE_MODEL,
    )
    assert captured[-1].dataset_id == CCPC_DATASET_ID
    assert EvaluateFactory.get_evaluator_family(CCPC_DATASET_ID) == "censorship"

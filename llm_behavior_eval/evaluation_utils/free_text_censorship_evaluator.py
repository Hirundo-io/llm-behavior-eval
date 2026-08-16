from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict, cast

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .base_evaluator import FreeTextSharedEvaluator, _GenerationRecord
from .censorship_utils import (
    CENSORSHIP_JUDGE_MODEL,
    CENSORSHIP_JUDGE_PROMPT,
    CENSORSHIP_JUDGE_PROMPT_SHA256,
    CENSORSHIP_JUDGE_RUBRIC_VERSION,
    CHINESE_CENSORSHIP_DATASET_CONFIG,
    CHINESE_CENSORSHIP_DATASET_REVISION,
    CHINESE_CENSORSHIP_DATASET_SOURCE,
    CHINESE_CENSORSHIP_DATASET_SPLIT,
    CHINESE_CENSORSHIP_GROUP_COUNT,
    CensorshipJudgment,
    is_approved_censorship_judge,
    parse_censorship_judgment,
)
from .custom_dataset import CustomDataset
from .util_functions import safe_apply_chat_template

if TYPE_CHECKING:
    from collections.abc import Sequence

    from datasets import Dataset as HFDataset

    from .dataset_config import DatasetConfig
    from .eval_config import EvaluationConfig
    from .eval_engine import EvalEngine

GRADING_STATUS_JUDGED: Literal["judged"] = "judged"
GRADING_STATUS_JUDGE_UNPARSEABLE: Literal["judge_unparseable"] = "judge_unparseable"
GRADING_STATUS_JUDGE_INCOMPLETE: Literal["judge_incomplete"] = "judge_incomplete"
GRADING_STATUS_JUDGE_UNKNOWN_FINISH_REASON: Literal["judge_unknown_finish_reason"] = (
    "judge_unknown_finish_reason"
)
GRADING_STATUS_MODEL_INCOMPLETE: Literal["model_incomplete"] = "model_incomplete"
GRADING_STATUS_UNKNOWN_FINISH_REASON: Literal["unknown_finish_reason"] = (
    "unknown_finish_reason"
)

CensorshipJudgeLabel: TypeAlias = Literal["true", "false", "unparseable"] | None
CensorshipGradingStatus: TypeAlias = Literal[
    "judged",
    "judge_unparseable",
    "judge_incomplete",
    "judge_unknown_finish_reason",
    "model_incomplete",
    "unknown_finish_reason",
]


def _snapshot_revision(reference: str) -> str | None:
    """Extract a snapshot revision from a local Hugging Face cache path.

    Args:
        reference: Model or judge reference that may contain ``/snapshots/<revision>``.

    Returns:
        The snapshot revision, or ``None`` when ``reference`` is not a snapshot path.
    """
    normalized = reference.replace("\\", "/").rstrip("/")
    marker = "/snapshots/"
    if marker not in normalized:
        return None
    revision = normalized.split(marker, 1)[1].split("/", 1)[0]
    return revision or None


def _adapter_manifest_sha256(reference: str | None) -> str | None:
    """Hash the regular adapter manifest files in a supplied adapter directory.

    Args:
        reference: Local adapter directory supplied for the evaluated model, if any.

    Returns:
        A SHA-256 digest of sorted ``adapter_config.json`` and ``adapter_model*``
        regular files, or ``None`` when no eligible manifest is available. Symlinked
        directories and files are excluded from provenance hashing.
    """
    if reference is None:
        return None
    path = Path(reference)
    if not path.is_dir() or path.is_symlink():
        return None
    candidates = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file()
        and not candidate.is_symlink()
        and (
            candidate.name == "adapter_config.json"
            or candidate.name.startswith("adapter_model")
        )
    )
    if not candidates:
        return None
    manifest = []
    for candidate in candidates:
        file_digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        manifest.append(f"{candidate.name}\0{file_digest}")
    return hashlib.sha256(("\n".join(manifest) + "\n").encode()).hexdigest()


def _resolved_revision(reference: str, configured_revision: str | None) -> str | None:
    """Prefer an explicit revision and otherwise infer one from a snapshot path.

    Args:
        reference: Model or judge reference that may be a cached snapshot path.
        configured_revision: Explicit immutable revision from evaluation configuration.

    Returns:
        The explicit revision when provided, a snapshot-derived revision otherwise, or
        ``None`` for a non-snapshot reference without an explicit revision.
    """
    return configured_revision or _snapshot_revision(reference)


def _is_immutable_revision(revision: str | None) -> bool:
    """Check whether a revision is an immutable 40-64 character hexadecimal ID.

    Args:
        revision: Candidate model or judge revision.

    Returns:
        ``True`` only when ``revision`` is a 40-64 character hexadecimal string.
    """
    return (
        revision is not None
        and re.fullmatch(r"[0-9a-fA-F]{40,64}", revision) is not None
    )


@dataclass
class _CensorshipGenerationRecord(_GenerationRecord):
    questions: list[str]
    source_group_ids: list[str]
    finish_reasons: list[str | None]


class CensorshipResponse(TypedDict):
    """Persisted evidence for one frozen censorship-benchmark group."""

    dataset: str
    source_group_id: str
    question: str
    llm_answer: str
    finish_reason: str | None
    judge_finish_reason: str | None
    judge_raw: str
    judge_label: CensorshipJudgeLabel
    grading_status: CensorshipGradingStatus


class FreeTextCensorshipEvaluator(FreeTextSharedEvaluator):
    """Evaluate the frozen English censorship set with its dedicated binary rubric."""

    def __init__(
        self, eval_config: EvaluationConfig, dataset_config: DatasetConfig
    ) -> None:
        """Initialize a censorship evaluator after validating immutable inputs.

        Args:
            eval_config: Model, judge, decoding, and output configuration.
            dataset_config: Pinned dataset source and preprocessing configuration.

        Raises:
            ValueError: If the judge identity or model/judge revisions do not meet
                the benchmark's immutable provenance requirements.
        """
        if not is_approved_censorship_judge(eval_config.judge_path_or_repo_id):
            raise ValueError(
                "chinese_censorship requires the pinned judge identity "
                f"{CENSORSHIP_JUDGE_MODEL!r}; got "
                f"{eval_config.judge_path_or_repo_id!r}."
            )
        if not _is_immutable_revision(
            _resolved_revision(
                eval_config.model_path_or_repo_id, eval_config.model_revision
            )
        ):
            raise ValueError(
                "chinese_censorship requires --model-revision or an immutable "
                "local model snapshot path."
            )
        if not _is_immutable_revision(
            _resolved_revision(
                eval_config.judge_path_or_repo_id, eval_config.judge_revision
            )
        ):
            raise ValueError(
                "chinese_censorship requires --judge-revision or an immutable "
                "local judge snapshot path."
            )
        super().__init__(eval_config, dataset_config)

    def prepare_dataloader(self) -> None:
        """Load, tokenize, shuffle, and validate the complete frozen benchmark.

        Raises:
            ValueError: If the benchmark is incomplete, truncated, or has no stable
                shuffle seed for cache-safe generation resumption.
        """
        custom_dataset = CustomDataset(
            self.dataset_config.file_path,
            self.dataset_config.dataset_type,
            trust_remote_code=self.trust_remote_code,
            token=self.eval_config.model_token,
            dataset_id=self.dataset_config.dataset_id,
        )
        test_dataset = custom_dataset.preprocess(
            self.tokenizer,
            self.dataset_config.preprocess_config,
            max_answer_tokens=self.eval_config.max_answer_tokens,
            enable_thinking=self.eval_config.enable_thinking,
            enable_thinking_arg_name=self.eval_config.enable_thinking_arg_name,
            thinking_start_token=self.eval_config.thinking_start_token,
            thinking_end_token=self.eval_config.thinking_end_token,
            pass_max_answer_tokens=self.eval_config.pass_max_answer_tokens,
            model_revision=self.eval_config.model_revision,
        )
        if len(test_dataset) != CHINESE_CENSORSHIP_GROUP_COUNT:
            raise ValueError(
                "Chinese censorship benchmark must contain exactly 288 rows"
            )
        if self.eval_config.max_samples and self.eval_config.max_samples < len(
            test_dataset
        ):
            raise ValueError(
                "chinese_censorship must evaluate all 288 locked groups; "
                "max_samples cannot truncate the product benchmark."
            )

        self.censorship_group_ids = custom_dataset.censorship_group_ids
        self.censorship_questions = custom_dataset.censorship_questions
        shuffle_seed = self.dataset_config.seed
        if shuffle_seed is None:
            shuffle_seed = self.eval_config.sampling_config.seed
        if shuffle_seed is None:
            raise ValueError(
                "chinese_censorship requires a dataset or sampling seed to resume "
                "cached generations deterministically."
            )
        shuffled = test_dataset.shuffle(seed=shuffle_seed)
        self.num_samples = len(shuffled)
        self.eval_dataset = cast("HFDataset", shuffled)
        shuffled_indices = [
            int(value) for value in self.eval_dataset["censorship_group_index"]
        ]
        self.eval_group_ids = [
            self.censorship_group_ids[index] for index in shuffled_indices
        ]
        self.eval_questions = [
            self.censorship_questions[index] for index in shuffled_indices
        ]
        self.eval_engine.set_dataset(self.eval_dataset)
        if not self.eval_engine.is_judge:
            self.effective_batch_size = self.eval_engine.get_batch_size()
        self.eval_loader = DataLoader(
            cast("Dataset", self.eval_dataset),
            batch_size=self.effective_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )
        self.has_stereotype = False

    def generate(self) -> Sequence[_GenerationRecord]:
        """Generate one response for every frozen censorship-benchmark group.

        Returns:
            Generation records aligned with the shuffled locked group order.
        """
        with torch.inference_mode():
            return self._collect_generations()

    def evaluate(self) -> None:
        """Generate, grade, persist, and optionally log the benchmark results."""

        def _run() -> None:
            generations = self.generate()
            self.free_test_model()
            with (
                self.dataset_mlflow_run(),
                self.get_judge_engine_context() as judge_engine,
            ):
                self.grade(generations, judge_engine)

        self._run_with_cleanup(_run)

    def _collect_generations(self) -> Sequence[_CensorshipGenerationRecord]:
        """Resume or generate records while enforcing the locked shuffled order.

        Returns:
            Complete generation records aligned with every locked source-group ID.

        Raises:
            ValueError: If cached or newly generated rows do not exactly match the
                frozen benchmark order and population.
        """
        self.ensure_test_model_ready()
        completed = [
            _CensorshipGenerationRecord(
                questions=cast("list[str]", item.get("questions", [])),
                source_group_ids=cast("list[str]", item.get("source_group_ids", [])),
                answers=cast("list[str]", item.get("answers", [])),
                finish_reasons=cast("list[str | None]", item.get("finish_reasons", [])),
            )
            for item in self.load_completed_generation_dicts()
        ]
        completed_ids = [
            group_id
            for generation in completed
            for group_id in generation.source_group_ids
        ]
        if completed_ids != self.eval_group_ids[: len(completed_ids)]:
            raise ValueError(
                "Cached censorship generations do not match the pinned dataset order; "
                "rerun with --replace-existing-output."
            )
        if len(completed_ids) > self.num_samples:
            raise ValueError(
                "Cached censorship generations exceed the locked population"
            )

        generations = list(completed)
        completed_batches = len(completed)
        for batch_index, batch in enumerate(
            tqdm(self.eval_loader, desc="Generating answers", unit="batch")
        ):
            if batch_index < completed_batches:
                continue
            input_ids = batch["test_input_ids"]
            attention_mask = batch["test_attention_mask"]
            indices = [int(index) for index in batch["censorship_group_index"].tolist()]
            group_ids = [self.censorship_group_ids[index] for index in indices]
            questions = [self.censorship_questions[index] for index in indices]
            answers, finish_reasons = self.generate_answers(input_ids, attention_mask)
            record = _CensorshipGenerationRecord(
                questions=questions,
                source_group_ids=group_ids,
                answers=answers,
                finish_reasons=finish_reasons,
            )
            generations.append(record)
            self.save_generations(
                [
                    {
                        "questions": record.questions,
                        "source_group_ids": record.source_group_ids,
                        "answers": record.answers,
                        "finish_reasons": record.finish_reasons,
                    }
                ]
            )
        all_ids = [
            group_id
            for generation in generations
            for group_id in generation.source_group_ids
        ]
        if all_ids != self.eval_group_ids:
            raise ValueError(
                "Generated censorship rows do not match all locked group IDs"
            )
        return generations

    def _grade_batch(
        self,
        judge_engine: EvalEngine,
        questions: list[str],
        generated_answers: list[str],
    ) -> tuple[list[CensorshipJudgment], list[str], list[str | None]]:
        """Grade a batch of answers with the dedicated binary censorship rubric.

        Args:
            judge_engine: Loaded judge inference engine.
            questions: Original English questions for the judged answers.
            generated_answers: Model answers paired positionally with ``questions``.

        Returns:
            Parsed verdicts, raw judge text, and judge finish reasons in input order.
        """
        self.prepare_judge_tokenizer()
        judge_tokenizer = self._get_judge_tokenizer()
        prompts = [
            safe_apply_chat_template(
                judge_tokenizer,
                [
                    {
                        "role": "user",
                        "content": CENSORSHIP_JUDGE_PROMPT.format(
                            question=question, response=answer
                        ),
                    }
                ],
            )
            for question, answer in zip(questions, generated_answers, strict=True)
        ]
        judge_outputs = self.run_judge_with_backoff(judge_engine, prompts)
        judge_texts = [item[0].get("generated_text") or "" for item in judge_outputs]
        finish_reasons = [item[0].get("finish_reason") for item in judge_outputs]
        return (
            [parse_censorship_judgment(text) for text in judge_texts],
            judge_texts,
            finish_reasons,
        )

    def _grade_impl(
        self,
        generations: Sequence[_GenerationRecord],
        judge_engine: EvalEngine | None = None,
    ) -> None:
        """Grade all generated answers and persist their dedicated censorship result.

        Args:
            generations: Response batches produced for the frozen benchmark.
            judge_engine: Loaded binary-censorship judge.

        Raises:
            ValueError: If no judge engine is available.
        """
        if judge_engine is None:
            raise ValueError(
                "FreeTextCensorshipEvaluator.grade() requires a judge engine."
            )
        responses: list[CensorshipResponse] = []
        for generation in tqdm(
            cast("Sequence[_CensorshipGenerationRecord]", generations),
            desc="Grading responses",
            unit="batch",
        ):
            answers = self._format_answers(generation.answers)
            judge_indices = [
                index
                for index, finish_reason in enumerate(generation.finish_reasons)
                if finish_reason == "stop"
            ]
            judgments: list[CensorshipJudgment | None] = [None] * len(answers)
            raw_outputs = [""] * len(answers)
            judge_finish_reasons: list[str | None] = [None] * len(answers)
            if judge_indices:
                batch_judgments, batch_raw, batch_finish = self._grade_batch(
                    judge_engine,
                    [generation.questions[index] for index in judge_indices],
                    [answers[index] for index in judge_indices],
                )
                for index, judgment, raw, judge_finish in zip(
                    judge_indices,
                    batch_judgments,
                    batch_raw,
                    batch_finish,
                    strict=True,
                ):
                    judgments[index] = judgment
                    raw_outputs[index] = raw
                    judge_finish_reasons[index] = judge_finish

            for (
                group_id,
                question,
                answer,
                finish_reason,
                judgment,
                raw,
                judge_finish,
            ) in zip(
                generation.source_group_ids,
                generation.questions,
                answers,
                generation.finish_reasons,
                judgments,
                raw_outputs,
                judge_finish_reasons,
                strict=True,
            ):
                if finish_reason == "length":
                    status = GRADING_STATUS_MODEL_INCOMPLETE
                    judge_label = None
                elif finish_reason != "stop":
                    status = GRADING_STATUS_UNKNOWN_FINISH_REASON
                    judge_label = None
                elif judge_finish == "length":
                    status = GRADING_STATUS_JUDGE_INCOMPLETE
                    judge_label = None
                elif judge_finish != "stop":
                    status = GRADING_STATUS_JUDGE_UNKNOWN_FINISH_REASON
                    judge_label = None
                elif judgment == CensorshipJudgment.UNPARSEABLE:
                    status = GRADING_STATUS_JUDGE_UNPARSEABLE
                    judge_label = cast("CensorshipJudgeLabel", judgment.value)
                else:
                    status = GRADING_STATUS_JUDGED
                    judge_label = cast(
                        "CensorshipJudgeLabel",
                        judgment.value if judgment is not None else None,
                    )
                responses.append(
                    {
                        "dataset": self.get_dataset_slug(),
                        "source_group_id": group_id,
                        "question": question,
                        "llm_answer": answer,
                        "finish_reason": finish_reason,
                        "judge_finish_reason": judge_finish,
                        "judge_raw": raw,
                        "judge_label": judge_label,
                        "grading_status": status,
                    }
                )
        self._save_censorship_results(responses)

    def _save_censorship_results(self, responses: list[CensorshipResponse]) -> None:
        """Persist complete benchmark evidence, metrics, summaries, and MLflow data.

        Args:
            responses: One typed response record for each locked benchmark group.

        Raises:
            ValueError: If evidence does not cover exactly the 288 frozen groups or
                contains an unsupported categorical verdict or grading status.
        """
        if len(responses) != CHINESE_CENSORSHIP_GROUP_COUNT:
            raise ValueError("Censorship results must contain all 288 locked groups")
        valid_judge_labels = {judgment.value for judgment in CensorshipJudgment}
        valid_grading_statuses = {
            GRADING_STATUS_JUDGED,
            GRADING_STATUS_JUDGE_UNPARSEABLE,
            GRADING_STATUS_JUDGE_INCOMPLETE,
            GRADING_STATUS_JUDGE_UNKNOWN_FINISH_REASON,
            GRADING_STATUS_MODEL_INCOMPLETE,
            GRADING_STATUS_UNKNOWN_FINISH_REASON,
        }
        for response in responses:
            if response["judge_label"] not in valid_judge_labels | {None}:
                raise ValueError(
                    "Censorship evidence contains an unsupported judge label"
                )
            if response["grading_status"] not in valid_grading_statuses:
                raise ValueError(
                    "Censorship evidence contains an unsupported grading status"
                )
        grading_status_counts = Counter(
            response["grading_status"] for response in responses
        )
        censored = sum(
            response["grading_status"] == GRADING_STATUS_JUDGED
            and response["judge_label"] == CensorshipJudgment.CENSORED.value
            for response in responses
        )
        judged = grading_status_counts[GRADING_STATUS_JUDGED]
        incomplete = grading_status_counts[GRADING_STATUS_MODEL_INCOMPLETE]
        unknown = grading_status_counts[GRADING_STATUS_UNKNOWN_FINISH_REASON]
        unparseable = grading_status_counts[GRADING_STATUS_JUDGE_UNPARSEABLE]
        judge_incomplete = grading_status_counts[GRADING_STATUS_JUDGE_INCOMPLETE]
        judge_unknown = grading_status_counts[
            GRADING_STATUS_JUDGE_UNKNOWN_FINISH_REASON
        ]
        metric_valid = judged == CHINESE_CENSORSHIP_GROUP_COUNT
        censorship_rate = (
            censored / CHINESE_CENSORSHIP_GROUP_COUNT if metric_valid else None
        )
        thinking_mode = "on" if self.eval_config.enable_thinking else "off"
        metric_row = {
            "Censorship rate (%) ⬇️": (
                censorship_rate * 100.0 if censorship_rate is not None else None
            ),
            "Metric Valid": metric_valid,
            "Censored Samples": censored,
            "Samples": CHINESE_CENSORSHIP_GROUP_COUNT,
            "Judged Samples": judged,
            "Incomplete Responses": incomplete,
            "Unknown Finish Reasons": unknown,
            "Judge Unparseable": unparseable,
            "Judge Incomplete": judge_incomplete,
            "Judge Unknown Finish Reasons": judge_unknown,
            "Thinking": thinking_mode,
        }
        output_dir = self.get_output_dir()
        (output_dir / "responses.json").write_text(
            json.dumps(responses, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pd.DataFrame([metric_row]).to_csv(
            output_dir / "metrics.csv", index=False, float_format="%.3f"
        )
        self._write_evaluation_contract(output_dir)

        model_slug = self.get_model_slug()
        model_results_dir = Path(self.eval_config.results_dir) / model_slug
        common_summary_fields = {
            "Dataset": self.get_dataset_slug(),
            "Thinking": thinking_mode,
            **metric_row,
        }
        self._append_summary_row(
            model_results_dir / "summary_full.csv",
            pd.DataFrame(
                [
                    {
                        "Model": model_slug,
                        "Dataset Type": str(self.dataset_config.dataset_type),
                        "Text Format": "free_text",
                        **common_summary_fields,
                    }
                ]
            ),
        )
        self._append_summary_row(
            model_results_dir / "summary_brief.csv",
            pd.DataFrame([common_summary_fields]),
        )
        if self.mlflow_config:
            metrics = {
                "censored_samples": float(censored),
                "judged_samples": float(judged),
                "incomplete_responses": float(incomplete),
                "unknown_finish_reasons": float(unknown),
                "judge_unparseable": float(unparseable),
                "judge_incomplete": float(judge_incomplete),
                "judge_unknown_finish_reasons": float(judge_unknown),
                "metric_valid": float(metric_valid),
                "num_samples": float(CHINESE_CENSORSHIP_GROUP_COUNT),
            }
            if censorship_rate is not None:
                metrics["censorship_rate"] = censorship_rate
            self._log_mlflow_metrics(metrics)
            self._log_mlflow_artifacts()

    def _write_evaluation_contract(self, output_dir: Path) -> None:
        """Write immutable dataset, model, judge, decoding, and runtime provenance.

        Args:
            output_dir: Benchmark output directory receiving the contract JSON.
        """
        package_versions: dict[str, str | None] = {}
        for package in (
            "llm-behavior-eval",
            "datasets",
            "torch",
            "transformers",
            "vllm",
        ):
            try:
                package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                package_versions[package] = None
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        contract = {
            "contract_version": 1,
            "dataset": {
                "source": CHINESE_CENSORSHIP_DATASET_SOURCE,
                "config": CHINESE_CENSORSHIP_DATASET_CONFIG,
                "revision": CHINESE_CENSORSHIP_DATASET_REVISION,
                "split": CHINESE_CENSORSHIP_DATASET_SPLIT,
                "group_count": len(self.eval_group_ids),
            },
            "model": {
                "requested_model": self.eval_config.model_path_or_repo_id,
                "revision": _resolved_revision(
                    self.eval_config.model_path_or_repo_id,
                    self.eval_config.model_revision,
                ),
                "adapter": self.eval_config.lora_path_or_repo_id,
                "adapter_manifest_sha256": _adapter_manifest_sha256(
                    self.eval_config.lora_path_or_repo_id
                ),
            },
            "judge": {
                "requested_model": self.eval_config.judge_path_or_repo_id,
                "approved_model": CENSORSHIP_JUDGE_MODEL,
                "revision": _resolved_revision(
                    self.eval_config.judge_path_or_repo_id,
                    self.eval_config.judge_revision,
                ),
                "rubric_version": CENSORSHIP_JUDGE_RUBRIC_VERSION,
                "prompt": CENSORSHIP_JUDGE_PROMPT,
                "prompt_sha256": CENSORSHIP_JUDGE_PROMPT_SHA256,
            },
            "evaluation_config": self.eval_config.model_dump(
                exclude={"model_token", "judge_token"}, mode="json"
            ),
            "dataset_config": self.dataset_config.model_dump(mode="json"),
            "effective": {
                "model_batch_size": self.effective_batch_size,
                "judge_batch_size": self.effective_judge_batch_size,
                "tensor_parallel_size": (
                    self.eval_config.vllm_config.tensor_parallel_size
                    if self.eval_config.vllm_config is not None
                    and self.eval_config.vllm_config.tensor_parallel_size is not None
                    else torch.cuda.device_count()
                    if self.model_engine == "vllm" or self.judge_engine == "vllm"
                    else None
                ),
                "gpu_count": torch.cuda.device_count(),
                "gpu_names": gpu_names,
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": package_versions,
                "cuda": torch.version.cuda,
            },
        }
        (output_dir / "evaluation_contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8"
        )

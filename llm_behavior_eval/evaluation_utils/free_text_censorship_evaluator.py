"""Dedicated evaluator for CCPC-Bench."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict, cast

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset

from .base_evaluator import FreeTextSharedEvaluator, _GenerationRecord
from .censorship_utils import (
    CCPC_CANONICAL_ROWS,
    CCPC_DATASET_REPOSITORY,
    CCPC_DATASET_REVISION,
    CCPC_JUDGE_PROMPT_TEMPLATE,
    CCPC_JUDGE_SYSTEM_PROMPT,
    CCPC_MAX_ANSWER_TOKENS,
    CCPC_MAX_JUDGE_TOKENS,
    CCPC_REPETITION_PENALTY,
    CCPC_RUBRIC_VERSION,
    CCPC_TASK_VERSION,
    CCPC_TEMPERATURE,
    calculate_censorship_metrics,
    load_censorship_benchmark,
    parse_censorship_judgment,
)
from .dataset_config import DatasetConfig
from .eval_config import EvaluationConfig
from .eval_engine import EvalEngine
from .sampling_config import SamplingConfig
from .util_functions import is_model_multimodal, safe_apply_chat_template


@dataclass
class _CensorshipGenerationRecord(_GenerationRecord):
    questions: list[str]
    row_ids: list[str]
    finish_reasons: list[str | None]


class CensorshipResponse(TypedDict):
    """One persisted CCPC row: its source identity, evidence, and outcome."""

    benchmark_id: str
    question: str
    llm_answer: str
    finish_reason: str | None
    judge_finish_reason: str | None
    judge_raw: str
    judge_verdict: bool | None


MetricValue = str | int | float | bool | None


class FreeTextCensorshipEvaluator(FreeTextSharedEvaluator):
    """Evaluate a deterministic prefix of the configured CCPC-Bench split."""

    def __init__(
        self, eval_config: EvaluationConfig, dataset_config: DatasetConfig
    ) -> None:
        """Initialize the CCPC scoring evaluator.

        Args:
            eval_config: Evaluation settings.
            dataset_config: Dataset settings identifying CCPC-Bench.
        """
        self._validate_frozen_settings(eval_config)
        super().__init__(eval_config, dataset_config)

    @staticmethod
    def _validate_frozen_settings(eval_config: EvaluationConfig) -> None:
        """Reject settings that would make a CCPC result non-comparable."""
        required = {
            "max_answer_tokens": CCPC_MAX_ANSWER_TOKENS,
            "max_judge_tokens": CCPC_MAX_JUDGE_TOKENS,
            "sample": False,
            "sample_judge": False,
        }
        mismatches = {
            name: getattr(eval_config, name)
            for name, expected in required.items()
            if getattr(eval_config, name) != expected
        }
        sampling = eval_config.sampling_config
        if sampling.do_sample not in (None, False):
            mismatches["sampling_config.do_sample"] = sampling.do_sample
        if sampling.temperature not in (None, CCPC_TEMPERATURE):
            mismatches["sampling_config.temperature"] = sampling.temperature
        if mismatches:
            raise ValueError(
                "CCPC-Bench v6 uses frozen deterministic generation settings; "
                f"incompatible values: {mismatches}."
            )

    def prepare_dataloader(self) -> None:
        """Load, limit, and tokenize benchmark rows in published order."""
        benchmark = load_censorship_benchmark(self.eval_config.model_token)
        num_samples = (
            min(len(benchmark), self.eval_config.max_samples)
            if self.eval_config.max_samples
            else len(benchmark)
        )
        benchmark = benchmark.select(range(num_samples))
        self.benchmark_questions = cast("list[str]", benchmark["question"])
        self.benchmark_group_ids = cast("list[str]", benchmark["benchmark_id"])
        prompts = [
            safe_apply_chat_template(
                self.tokenizer,
                [{"role": "user", "content": f"{question}\n"}],
                is_multimodal=is_model_multimodal(
                    self.tokenizer.name_or_path,
                    self.trust_remote_code,
                    self.eval_config.model_token,
                ),
                max_answer_tokens=self.eval_config.max_answer_tokens,
                enable_thinking=self.eval_config.enable_thinking,
                enable_thinking_arg_name=self.eval_config.enable_thinking_arg_name,
                thinking_start_token=self.eval_config.thinking_start_token,
                thinking_end_token=self.eval_config.thinking_end_token,
                pass_max_answer_tokens=self.eval_config.pass_max_answer_tokens,
            )
            for question in self.benchmark_questions
        ]
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=self.dataset_config.preprocess_config.max_length,
        )
        self.eval_dataset = HFDataset.from_dict(
            {
                "test_input_ids": tokenized["input_ids"],
                "test_attention_mask": tokenized["attention_mask"],
                "ccpc_row_index": list(range(num_samples)),
            }
        )
        self.num_samples = num_samples
        self.eval_engine.set_dataset(self.eval_dataset)
        self.eval_loader = DataLoader(
            # A Hugging Face Dataset implements the map-style Dataset protocol.
            cast("Dataset", self.eval_dataset),
            batch_size=self.eval_engine.get_batch_size(),
            shuffle=False,
            collate_fn=self.data_collator,
        )
        self.has_stereotype = False

    def update_dataset_config(self, dataset_config: DatasetConfig) -> None:
        """Refresh run bookkeeping without reloading an already-validated CCPC snapshot.

        The base implementation always calls ``prepare_dataloader()``, which
        for CCPC means a second remote Hugging Face fetch. Because CCPC is
        deliberately published without a pinned revision, that second fetch
        could resolve to a different snapshot than the one already used for
        generation and desynchronize grading/accounting from the generated
        cohort. When ``dataset_config`` still identifies the same CCPC
        dataset, keep the validated benchmark state and only refresh the
        bookkeeping (seed, run-config cache) that legitimately changes
        between generation and grading.

        Args:
            dataset_config: The dataset configuration to apply.
        """
        same_dataset = (
            getattr(self, "dataset_config", None) is not None
            and dataset_config.dataset_id == self.dataset_config.dataset_id
        )
        self.dataset_config = dataset_config
        self._set_seed()
        if not same_dataset:
            self.prepare_dataloader()
        self._ensure_run_configuration_allowed()

    def _generate_model_answers(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[list[str], list[str | None]]:
        """Generate model answers with the frozen CCPC v6 settings.

        Args:
            input_ids: Tokenized benchmark prompts.
            attention_mask: Attention mask corresponding to the prompt tokens.

        Returns:
            Generated answers and their finish reasons.
        """
        return self.eval_engine.generate_answers(
            input_ids,
            attention_mask,
            SamplingConfig(
                do_sample=False,
                temperature=CCPC_TEMPERATURE,
                seed=(
                    self.dataset_config.seed
                    if self.dataset_config.seed is not None
                    else self.eval_config.sampling_config.seed
                ),
            ),
            repetition_penalty=CCPC_REPETITION_PENALTY,
        )

    def generate(self) -> Sequence[_GenerationRecord]:
        """Generate answers for the selected benchmark rows.

        Args:
            None.

        Returns:
            Batched generation records covering the selected benchmark rows.
        """
        self.ensure_test_model_ready()
        generations: list[_CensorshipGenerationRecord] = []
        with torch.inference_mode():
            for batch in self.eval_loader:
                row_indices = [int(index) for index in batch["ccpc_row_index"].tolist()]
                answers, finish_reasons = self._generate_model_answers(
                    batch["test_input_ids"], batch["test_attention_mask"]
                )
                generations.append(
                    _CensorshipGenerationRecord(
                        questions=[self.benchmark_questions[i] for i in row_indices],
                        row_ids=[self.benchmark_group_ids[i] for i in row_indices],
                        answers=answers,
                        finish_reasons=finish_reasons,
                    )
                )
        return generations

    def evaluate(self) -> None:
        """Run complete generation and grading with evaluator cleanup."""

        def _run() -> None:
            """Run generation and grading inside the cleanup wrapper."""
            generations = self.generate()
            self.free_test_model()
            with (
                self.dataset_mlflow_run(),
                self.get_judge_engine_context() as judge_engine,
            ):
                self.grade(generations, judge_engine)

        self._run_with_cleanup(_run)

    def _grade_batch(
        self,
        judge_engine: EvalEngine,
        questions: list[str],
        answers: list[str],
    ) -> tuple[list[bool | None], list[str], list[str | None]]:
        """Grade one batch with the binary judge rubric.

        Args:
            judge_engine: Engine hosting the judge model.
            questions: Source questions aligned with ``answers``.
            answers: Complete target answers to score.

        Returns:
            Verdicts, raw judge texts, and finish reasons aligned with
            ``questions``/``answers``.
        """
        self.prepare_judge_tokenizer()
        tokenizer = self._get_judge_tokenizer()
        prompts = [
            safe_apply_chat_template(
                tokenizer,
                [
                    {
                        "role": "system",
                        "content": CCPC_JUDGE_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": CCPC_JUDGE_PROMPT_TEMPLATE.format(
                            serialized_judge_input=json.dumps(
                                {"question": question, "response": answer},
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        ),
                    },
                ],
            )
            for question, answer in zip(questions, answers, strict=True)
        ]
        outputs = self.run_judge_with_backoff(
            judge_engine,
            prompts,
            repetition_penalty=CCPC_REPETITION_PENALTY,
        )
        judge_texts = [item[0].get("generated_text") or "" for item in outputs]
        finish_reasons = [item[0].get("finish_reason") for item in outputs]
        return (
            [parse_censorship_judgment(text) for text in judge_texts],
            judge_texts,
            finish_reasons,
        )

    def _save_results(self, responses: list[CensorshipResponse]) -> None:
        """Persist row evidence and metrics for the selected rows.

        Args:
            responses: Results in the benchmark's published row order.

        Returns:
            None.
        """
        response_group_ids = [response["benchmark_id"] for response in responses]
        if response_group_ids != self.benchmark_group_ids:
            raise ValueError(
                "CCPC-Bench responses must cover every benchmark_id in source order."
            )
        expected_rows = len(self.benchmark_group_ids)
        censorship_rate, stderr, judged_fraction = calculate_censorship_metrics(
            [response["judge_verdict"] for response in responses]
        )
        metrics: dict[str, MetricValue] = {
            "censorship_rate": censorship_rate,
            "stderr": stderr,
            "judged_fraction": judged_fraction,
            "canonical": expected_rows == CCPC_CANONICAL_ROWS,
            "canonical_rows": CCPC_CANONICAL_ROWS,
            "evaluated_rows": expected_rows,
            "accounted_rows": len(responses),
            "task_version": CCPC_TASK_VERSION,
            "evaluator_package": "llm-behavior-eval",
            "evaluator_version": version("llm-behavior-eval"),
            "dataset_repository": CCPC_DATASET_REPOSITORY,
            "dataset_revision": CCPC_DATASET_REVISION,
            "rubric_version": CCPC_RUBRIC_VERSION,
            "judge_model": self.eval_config.judge_path_or_repo_id,
            "judge_model_revision": None,
            "max_answer_tokens": CCPC_MAX_ANSWER_TOKENS,
            "answer_temperature": CCPC_TEMPERATURE,
            "max_judge_tokens": CCPC_MAX_JUDGE_TOKENS,
            "judge_temperature": CCPC_TEMPERATURE,
            "repetition_penalty": CCPC_REPETITION_PENALTY,
        }
        output_dir = self.get_output_dir()
        (output_dir / "responses.json").write_text(
            json.dumps(responses, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
        summary = {"Dataset": self.get_dataset_slug(), **metrics}
        model_dir = Path(self.eval_config.results_dir) / self.get_model_slug()
        self._append_summary_row(
            model_dir / "summary_full.csv",
            pd.DataFrame([{"Model": self.get_model_slug(), **summary}]),
        )
        self._append_summary_row(
            model_dir / "summary_brief.csv", pd.DataFrame([summary])
        )

    def _grade_impl(
        self,
        generations: Sequence[_GenerationRecord],
        judge_engine: EvalEngine | None = None,
    ) -> None:
        """Grade generated rows and preserve all determinate and failure states.

        Args:
            generations: Batched evaluated-model generation records.
            judge_engine: Engine hosting the approved judge model.
        """
        if judge_engine is None:
            raise ValueError("CCPC-Bench grading requires a judge engine.")
        responses: list[CensorshipResponse] = []
        for generation in cast("Sequence[_CensorshipGenerationRecord]", generations):
            judge_answers = self._format_answers(generation.answers)
            verdicts, judge_raw, judge_finishes = self._grade_batch(
                judge_engine, generation.questions, judge_answers
            )
            for (
                group_id,
                question,
                answer,
                model_finish,
                verdict,
                raw,
                judge_finish,
            ) in zip(
                generation.row_ids,
                generation.questions,
                generation.answers,
                generation.finish_reasons,
                verdicts,
                judge_raw,
                judge_finishes,
                strict=True,
            ):
                responses.append(
                    {
                        "benchmark_id": group_id,
                        "question": question,
                        "llm_answer": answer,
                        "finish_reason": model_finish,
                        "judge_finish_reason": judge_finish,
                        "judge_raw": raw,
                        "judge_verdict": verdict,
                    }
                )
        self._save_results(responses)

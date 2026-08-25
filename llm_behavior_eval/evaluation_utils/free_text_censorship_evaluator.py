"""Dedicated evaluator for the canonical CCPC-Bench contract."""

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset

from .base_evaluator import FreeTextSharedEvaluator, _GenerationRecord
from .censorship_utils import (
    CCPC_BENCHMARK_CONFIG,
    CCPC_DATASET_ID,
    CCPC_EXPECTED_ROWS,
    CCPC_JUDGE_MODEL,
    CCPC_JUDGE_PROMPT,
    CCPC_JUDGE_REPETITION_PENALTY,
    CCPC_MODEL_REPETITION_PENALTY,
    CensorshipBenchmarkConfig,
    CensorshipStatus,
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
    source_group_ids: list[str]
    finish_reasons: list[str | None]


class _CensorshipRunConfig(FreeTextSharedEvaluator.RunConfig):
    ccpc_benchmark: CensorshipBenchmarkConfig


class CensorshipResponse(TypedDict):
    source_group_id: str
    question: str
    llm_answer: str
    finish_reason: str | None
    judge_finish_reason: str | None
    judge_raw: str
    judge_verdict: bool | None
    status: str


MetricValue = str | int | float | bool | None


class FreeTextCensorshipEvaluator(FreeTextSharedEvaluator):
    """Evaluate every row of the pinned CCPC-Bench split."""

    def __init__(
        self, eval_config: EvaluationConfig, dataset_config: DatasetConfig
    ) -> None:
        """Initialize the evaluator after validating its frozen model contract.

        Args:
            eval_config: Evaluation and model settings for the run.
            dataset_config: Dataset settings identifying CCPC-Bench.

        Returns:
            None.
        """
        if dataset_config.dataset_id != CCPC_DATASET_ID:
            raise ValueError(f"CCPC-Bench requires dataset_id={CCPC_DATASET_ID!r}.")
        if eval_config.judge_path_or_repo_id != CCPC_JUDGE_MODEL:
            raise ValueError(
                f"CCPC-Bench requires judge {CCPC_JUDGE_MODEL!r}; "
                f"got {eval_config.judge_path_or_repo_id!r}."
            )
        super().__init__(eval_config, dataset_config)

    def prepare_dataloader(self) -> None:
        """Load and tokenize every benchmark row in published order."""
        benchmark = load_censorship_benchmark(self.eval_config.model_token)
        self.benchmark_questions = cast("list[str]", benchmark["question"])
        self.benchmark_group_ids = cast("list[str]", benchmark["source_group_id"])
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
                "ccpc_row_index": list(range(CCPC_EXPECTED_ROWS)),
            }
        )
        self.num_samples = CCPC_EXPECTED_ROWS
        self.eval_engine.set_dataset(self.eval_dataset)
        self.eval_loader = DataLoader(
            # A Hugging Face Dataset implements the map-style Dataset protocol.
            cast("Dataset", self.eval_dataset),
            batch_size=self.eval_engine.get_batch_size(),
            shuffle=False,
            collate_fn=self.data_collator,
        )
        self.has_stereotype = False

    def _generate_model_answers(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[list[str], list[str | None]]:
        """Generate model answers with the frozen model repetition penalty.

        Args:
            input_ids: Tokenized benchmark prompts.
            attention_mask: Attention mask corresponding to the prompt tokens.

        Returns:
            Generated answers and their finish reasons.
        """
        sampling = self.eval_config.sampling_config
        return self.eval_engine.generate_answers(
            input_ids,
            attention_mask,
            SamplingConfig(
                do_sample=(
                    sampling.do_sample
                    if sampling.do_sample is not None
                    else self.eval_config.sample
                ),
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                seed=(
                    self.dataset_config.seed
                    if self.dataset_config.seed is not None
                    else sampling.seed
                ),
            ),
            repetition_penalty=CCPC_MODEL_REPETITION_PENALTY,
        )

    def generate(self) -> Sequence[_GenerationRecord]:
        """Generate answers for the complete benchmark cohort.

        Args:
            None.

        Returns:
            Batched generation records covering all benchmark rows.
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
                        source_group_ids=[
                            self.benchmark_group_ids[i] for i in row_indices
                        ],
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
        """Grade one batch with the frozen binary judge rubric.

        Args:
            judge_engine: Engine hosting the approved judge model.
            questions: Original benchmark questions for this batch.
            answers: Model answers corresponding to the questions.

        Returns:
            Parsed verdicts, raw judge texts, and judge finish reasons.
        """
        self.prepare_judge_tokenizer()
        tokenizer = self._get_judge_tokenizer()
        prompts = [
            safe_apply_chat_template(
                tokenizer,
                [
                    {
                        "role": "user",
                        "content": CCPC_JUDGE_PROMPT.format(
                            question=question, response=answer
                        ),
                    }
                ],
            )
            for question, answer in zip(questions, answers, strict=True)
        ]
        outputs = self.run_judge_with_backoff(
            judge_engine,
            prompts,
            repetition_penalty=CCPC_JUDGE_REPETITION_PENALTY,
        )
        texts = [item[0].get("generated_text") or "" for item in outputs]
        finish_reasons = [item[0].get("finish_reason") for item in outputs]
        return (
            [parse_censorship_judgment(text) for text in texts],
            texts,
            finish_reasons,
        )

    @staticmethod
    def _status(
        model_finish: str | None,
        judge_finish: str | None,
        verdict: bool | None,
    ) -> CensorshipStatus:
        """Resolve model and judge outcomes to one exhaustive row status.

        Args:
            model_finish: Finish reason from evaluated-model generation.
            judge_finish: Finish reason from judge generation.
            verdict: Parsed judge verdict, if determinate.

        Returns:
            The status that preserves the first relevant failure condition.
        """
        if model_finish == "length":
            return CensorshipStatus.MODEL_INCOMPLETE
        if model_finish != "stop":
            return CensorshipStatus.MODEL_UNKNOWN
        if judge_finish == "length":
            return CensorshipStatus.JUDGE_INCOMPLETE
        if judge_finish != "stop":
            return CensorshipStatus.JUDGE_UNKNOWN
        if verdict is None:
            return CensorshipStatus.JUDGE_UNPARSEABLE
        return (
            CensorshipStatus.JUDGED_TRUE if verdict else CensorshipStatus.JUDGED_FALSE
        )

    def _save_results(self, responses: list[CensorshipResponse]) -> None:
        """Persist row evidence and complete-cohort metrics.

        Args:
            responses: Results in the benchmark's published row order.

        Returns:
            None.
        """
        statuses = Counter(response["status"] for response in responses)
        response_group_ids = [response["source_group_id"] for response in responses]
        if response_group_ids != self.benchmark_group_ids:
            raise ValueError(
                "CCPC-Bench responses must cover every source_group_id in "
                "published order."
            )
        judged = (
            statuses[CensorshipStatus.JUDGED_TRUE.value]
            + statuses[CensorshipStatus.JUDGED_FALSE.value]
        )
        metric_valid = judged == CCPC_EXPECTED_ROWS
        censorship_rate = (
            statuses[CensorshipStatus.JUDGED_TRUE.value] / CCPC_EXPECTED_ROWS
            if metric_valid
            else None
        )
        metrics: dict[str, MetricValue] = {
            "Censorship rate (%) ⬇️": (
                censorship_rate * 100 if censorship_rate is not None else None
            ),
            "Metric Valid": metric_valid,
            "Expected Samples": CCPC_EXPECTED_ROWS,
            "Accounted Samples": len(responses),
            "Judged True": statuses[CensorshipStatus.JUDGED_TRUE.value],
            "Judged False": statuses[CensorshipStatus.JUDGED_FALSE.value],
            **{
                status.value.replace("_", " ").title(): statuses[status.value]
                for status in CensorshipStatus
                if not status.value.startswith("judged_")
            },
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

        Returns:
            None.
        """
        if judge_engine is None:
            raise ValueError("CCPC-Bench grading requires a judge engine.")
        responses: list[CensorshipResponse] = []
        for generation in cast("Sequence[_CensorshipGenerationRecord]", generations):
            judge_answers = self._format_answers(generation.answers)
            indices = [
                i
                for i, finish_reason in enumerate(generation.finish_reasons)
                if finish_reason == "stop"
            ]
            verdicts: list[bool | None] = [None] * len(generation.answers)
            judge_raw = [""] * len(generation.answers)
            judge_finishes: list[str | None] = [None] * len(generation.answers)
            if indices:
                batch_verdicts, batch_raw, batch_finishes = self._grade_batch(
                    judge_engine,
                    [generation.questions[i] for i in indices],
                    [judge_answers[i] for i in indices],
                )
                for i, verdict, raw, finish in zip(
                    indices, batch_verdicts, batch_raw, batch_finishes, strict=True
                ):
                    verdicts[i], judge_raw[i], judge_finishes[i] = verdict, raw, finish
            for (
                group_id,
                question,
                answer,
                model_finish,
                verdict,
                raw,
                judge_finish,
            ) in zip(
                generation.source_group_ids,
                generation.questions,
                generation.answers,
                generation.finish_reasons,
                verdicts,
                judge_raw,
                judge_finishes,
                strict=True,
            ):
                status = self._status(model_finish, judge_finish, verdict)
                responses.append(
                    {
                        "source_group_id": group_id,
                        "question": question,
                        "llm_answer": answer,
                        "finish_reason": model_finish,
                        "judge_finish_reason": judge_finish,
                        "judge_raw": raw,
                        "judge_verdict": verdict
                        if status
                        in {
                            CensorshipStatus.JUDGED_TRUE,
                            CensorshipStatus.JUDGED_FALSE,
                        }
                        else None,
                        "status": status.value,
                    }
                )
        self._save_results(responses)

    def _current_run_config(self) -> _CensorshipRunConfig:
        """Return the base run identity plus frozen CCPC-Bench provenance.

        Args:
            None.

        Returns:
            The serializable run configuration and benchmark contract pins.
        """
        run_config = super()._current_run_config()
        run_config["evaluation_config"].pop("max_samples", None)
        return {
            **run_config,
            "ccpc_benchmark": CCPC_BENCHMARK_CONFIG.copy(),
        }

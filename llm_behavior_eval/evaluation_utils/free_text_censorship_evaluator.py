"""Dedicated evaluator for the canonical CCPC-Bench contract."""

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset

from .base_evaluator import FreeTextSharedEvaluator, _GenerationRecord
from .censorship_utils import (
    CCPC_DATASET_ID,
    CCPC_JUDGE_PROMPT,
    CCPC_JUDGE_REPETITION_PENALTY,
    CCPC_MODEL_REPETITION_PENALTY,
    CensorshipBenchmarkConfig,
    CensorshipIdentityField,
    CensorshipStatus,
    judge_prompt_overflows,
    load_censorship_benchmark,
    parse_censorship_judgment,
    validate_ccpc_judge_contract,
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


class _CensorshipRunConfig(FreeTextSharedEvaluator.RunConfig):
    ccpc_benchmark: CensorshipBenchmarkConfig


CensorshipResponse = dict[str, str | bool | None]


MetricValue = str | int | float | bool | None


class FreeTextCensorshipEvaluator(FreeTextSharedEvaluator):
    """Evaluate every row of the pinned CCPC-Bench split."""

    def __init__(
        self, eval_config: EvaluationConfig, dataset_config: DatasetConfig
    ) -> None:
        """Initialize after validating the frozen CCPC contract.

        Args:
            eval_config: Evaluation settings, including the frozen judge contract.
            dataset_config: CCPC dataset identity and optional frozen revision.
        """
        if dataset_config.dataset_id != CCPC_DATASET_ID:
            raise ValueError(f"CCPC-Bench requires dataset_id={CCPC_DATASET_ID!r}.")
        validate_ccpc_judge_contract(eval_config)
        super().__init__(eval_config, dataset_config)

    def prepare_dataloader(self) -> None:
        """Load and tokenize every benchmark row in published order."""
        resolved = load_censorship_benchmark(
            self.dataset_config, self.eval_config.model_token
        )
        benchmark = resolved.dataset
        self.identity_field: CensorshipIdentityField = resolved.identity_field
        self.expected_rows = resolved.benchmark_config["expected_rows"]
        self._ccpc_benchmark_config = resolved.benchmark_config
        self.benchmark_questions = cast("list[str]", benchmark["question"])
        self.benchmark_row_ids = resolved.row_ids
        prompts = [
            safe_apply_chat_template(
                self.tokenizer,
                [{"role": "user", "content": f"{question}\n"}],
                is_multimodal=is_model_multimodal(
                    self.tokenizer.name_or_path,
                    self.trust_remote_code,
                    self.eval_config.model_token,
                    self.eval_config.model_revision,
                ),
                max_answer_tokens=self.eval_config.max_answer_tokens,
                enable_thinking=self.eval_config.enable_thinking,
                enable_thinking_arg_name=self.eval_config.enable_thinking_arg_name,
                thinking_start_token=self.eval_config.thinking_start_token,
                thinking_end_token=self.eval_config.thinking_end_token,
                pass_max_answer_tokens=self.eval_config.pass_max_answer_tokens,
                reasoning_effort=self.eval_config.model_reasoning_effort,
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
                "ccpc_row_index": list(range(self.expected_rows)),
            }
        )
        self.num_samples = self.expected_rows
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
                        row_ids=[self.benchmark_row_ids[i] for i in row_indices],
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
    ) -> tuple[list[bool | None], list[str], list[str | None], list[bool]]:
        """Grade one batch with the frozen binary judge rubric.

        Overflowing prompts are excluded from the judge call (never truncated);
        see ``CCPC_JUDGE_MAX_PROMPT_TOKENS``.

        Args:
            judge_engine: Engine hosting the frozen judge model.
            questions: Source questions aligned with ``answers``.
            answers: Complete target answers to score.

        Returns:
            Verdicts, raw judge texts, finish reasons, and overflow flags aligned
            with ``questions``/``answers``.
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
        overflow_flags = [
            judge_prompt_overflows(len(tokenizer(prompt)["input_ids"]))
            for prompt in prompts
        ]
        judged_indices = [
            index for index, overflow in enumerate(overflow_flags) if not overflow
        ]
        verdicts: list[bool | None] = [None] * len(prompts)
        judge_texts: list[str] = [""] * len(prompts)
        finish_reasons: list[str | None] = [None] * len(prompts)
        if judged_indices:
            outputs = self.run_judge_with_backoff(
                judge_engine,
                [prompts[index] for index in judged_indices],
                repetition_penalty=CCPC_JUDGE_REPETITION_PENALTY,
            )
            for index, output in zip(judged_indices, outputs, strict=True):
                text = output[0].get("generated_text") or ""
                judge_texts[index] = text
                finish_reasons[index] = output[0].get("finish_reason")
                verdicts[index] = parse_censorship_judgment(text)
        return verdicts, judge_texts, finish_reasons, overflow_flags

    @staticmethod
    def _status(
        model_finish: str | None,
        judge_finish: str | None,
        verdict: bool | None,
        judge_overflow: bool = False,
    ) -> CensorshipStatus:
        """Resolve model and judge outcomes to one exhaustive row status.

        Args:
            model_finish: Target generation finish reason.
            judge_finish: Judge generation finish reason.
            verdict: Parsed judge verdict, if any.
            judge_overflow: Whether the complete judge prompt exceeded its budget.

        Returns:
            The determinate or failure status for one benchmark row.
        """
        if model_finish == "length":
            return CensorshipStatus.MODEL_INCOMPLETE
        if model_finish != "stop":
            return CensorshipStatus.MODEL_UNKNOWN
        if judge_overflow:
            return CensorshipStatus.JUDGE_PROMPT_OVERFLOW
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
        response_group_ids = [response[self.identity_field] for response in responses]
        if response_group_ids != self.benchmark_row_ids:
            raise ValueError(
                f"CCPC-Bench responses must cover every {self.identity_field} in "
                "published order."
            )
        judged = (
            statuses[CensorshipStatus.JUDGED_TRUE.value]
            + statuses[CensorshipStatus.JUDGED_FALSE.value]
        )
        metric_valid = judged == self.expected_rows
        censorship_rate = (
            statuses[CensorshipStatus.JUDGED_TRUE.value] / self.expected_rows
            if metric_valid
            else None
        )
        metrics: dict[str, MetricValue] = {
            "Censorship rate (%) ⬇️": (
                censorship_rate * 100 if censorship_rate is not None else None
            ),
            "Metric Valid": metric_valid,
            "Expected Samples": self.expected_rows,
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
            judge_overflows = [False] * len(generation.answers)
            if indices:
                batch_verdicts, batch_raw, batch_finishes, batch_overflows = (
                    self._grade_batch(
                        judge_engine,
                        [generation.questions[i] for i in indices],
                        [judge_answers[i] for i in indices],
                    )
                )
                for i, verdict, raw, finish, overflow in zip(
                    indices,
                    batch_verdicts,
                    batch_raw,
                    batch_finishes,
                    batch_overflows,
                    strict=True,
                ):
                    verdicts[i], judge_raw[i], judge_finishes[i], judge_overflows[i] = (
                        verdict,
                        raw,
                        finish,
                        overflow,
                    )
            for (
                group_id,
                question,
                answer,
                model_finish,
                verdict,
                raw,
                judge_finish,
                judge_overflow,
            ) in zip(
                generation.row_ids,
                generation.questions,
                generation.answers,
                generation.finish_reasons,
                verdicts,
                judge_raw,
                judge_finishes,
                judge_overflows,
                strict=True,
            ):
                status = self._status(
                    model_finish, judge_finish, verdict, judge_overflow
                )
                responses.append(
                    {
                        self.identity_field: group_id,
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

        Returns:
            The serializable run configuration and benchmark contract pins.
        """
        run_config = super()._current_run_config()
        run_config["evaluation_config"].pop("max_samples", None)
        return {
            **run_config,
            "ccpc_benchmark": self._ccpc_benchmark_config.copy(),
        }

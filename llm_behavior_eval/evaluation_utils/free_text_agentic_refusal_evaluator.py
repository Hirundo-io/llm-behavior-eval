"""Single-continuation agentic refusal evaluation without executing tools."""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset

from .agentic_refusal_utils import (
    AGENTIC_REFUSAL_DATASET,
    AGENTIC_REFUSAL_JUDGE_PROMPT,
    AGENTIC_REFUSAL_REVISION,
    AGENTIC_REFUSAL_RUBRIC_VERSION,
    calculate_agentic_refusal_metrics,
    load_agentic_refusal_benchmark,
    parse_agentic_refusal_judgment,
    resolve_agentic_dataset_config,
    select_agentic_pairs,
    serialize_agentic_judge_input,
)
from .base_evaluator import FreeTextSharedEvaluator, _GenerationRecord
from .dataset_config import DatasetConfig
from .eval_engine import EvalEngine
from .util_functions import is_model_multimodal, safe_apply_chat_template


@dataclass
class _AgenticRefusalGenerationRecord(_GenerationRecord):
    rows: list[dict[str, Any]]
    finish_reasons: list[str | None]


class FreeTextAgenticRefusalEvaluator(FreeTextSharedEvaluator):
    """Keep pairing, task-relative judging and metrics isolated from safety refusal."""

    def update_dataset_config(self, dataset_config: DatasetConfig) -> None:
        """Retain agentic preprocessing defaults when reusing an evaluator."""
        super().update_dataset_config(resolve_agentic_dataset_config(dataset_config))

    def prepare_dataloader(self) -> None:
        """Select complete pairs and fail rather than silently truncate an attack."""
        dataset = load_agentic_refusal_benchmark(self.eval_config.model_token)
        self.benchmark_rows = select_agentic_pairs(
            dataset.to_list(), self.eval_config.max_samples, self.dataset_config.seed
        )
        multimodal = is_model_multimodal(
            self.tokenizer.name_or_path,
            self.trust_remote_code,
            self.eval_config.model_token,
        )
        prompts = [
            safe_apply_chat_template(
                self.tokenizer,
                [
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["question"]},
                ],
                is_multimodal=multimodal,
                max_answer_tokens=self.eval_config.max_answer_tokens,
                enable_thinking=self.eval_config.enable_thinking,
                enable_thinking_arg_name=self.eval_config.enable_thinking_arg_name,
                thinking_start_token=self.eval_config.thinking_start_token,
                thinking_end_token=self.eval_config.thinking_end_token,
                pass_max_answer_tokens=self.eval_config.pass_max_answer_tokens,
            )
            for row in self.benchmark_rows
        ]
        tokenized = self.tokenizer(prompts, truncation=False, padding=True)
        max_length = self.dataset_config.preprocess_config.max_length
        if any(sum(mask) > max_length for mask in tokenized["attention_mask"]):
            raise ValueError(
                f"Agentic prompt exceeds max_length={max_length}; increase "
                "BIAS_PREPROCESS_MAX_LENGTH. Prompts are never silently truncated."
            )
        self.num_samples = len(self.benchmark_rows)
        self.eval_dataset = HFDataset.from_dict(
            {
                "test_input_ids": tokenized["input_ids"],
                "test_attention_mask": tokenized["attention_mask"],
                "agentic_row_index": list(range(self.num_samples)),
            }
        )
        self.eval_engine.set_dataset(self.eval_dataset)
        self.eval_loader = DataLoader(
            cast("Dataset", self.eval_dataset),
            batch_size=self.eval_engine.get_batch_size(),
            shuffle=False,
            collate_fn=self.data_collator,
        )
        self.has_stereotype = False

    def generate(self) -> Sequence[_GenerationRecord]:
        """Persist and resume exact source rows independently of inference batch size."""
        self.ensure_test_model_ready()
        saved = self.load_completed_generation_dicts()
        for item in saved:
            if (
                not isinstance(item, dict)
                or set(item) != {"rows", "answers", "finish_reasons"}
                or not isinstance(item["rows"], list)
                or not all(isinstance(row, dict) for row in item["rows"])
                or not isinstance(item["answers"], list)
                or not all(isinstance(answer, str) for answer in item["answers"])
                or not isinstance(item["finish_reasons"], list)
                or not all(
                    reason is None or isinstance(reason, str)
                    for reason in item["finish_reasons"]
                )
            ):
                raise ValueError(
                    "Malformed saved agentic generations; use a fresh output directory"
                )
        generations = [_AgenticRefusalGenerationRecord(**item) for item in saved]
        completed = [row for record in generations for row in record.rows]
        if completed != self.benchmark_rows[: len(completed)]:
            raise ValueError(
                "Saved agentic generations do not match the selected benchmark rows"
            )
        for record in generations:
            if len(record.rows) != len(record.answers) or len(record.rows) != len(
                record.finish_reasons
            ):
                raise ValueError("Misaligned saved agentic generations")
        with torch.inference_mode():
            for batch in self.eval_loader:
                indices = [int(index) for index in batch["agentic_row_index"].tolist()]
                offset = sum(index < len(completed) for index in indices)
                if offset == len(indices):
                    continue
                answers, reasons = self.generate_answers(
                    batch["test_input_ids"][offset:],
                    batch["test_attention_mask"][offset:],
                )
                rows = [self.benchmark_rows[index] for index in indices[offset:]]
                if len(answers) != len(rows) or len(reasons) != len(rows):
                    raise ValueError("Misaligned agentic model outputs")
                record = _AgenticRefusalGenerationRecord(
                    answers=answers, rows=rows, finish_reasons=reasons
                )
                generations.append(record)
                self.save_generations([asdict(record)])
        return generations

    def evaluate(self) -> None:
        """Run generation and grading using the shared cleanup and MLflow lifecycle."""

        def run() -> None:
            generations = self.generate()
            self.free_test_model()
            with self.dataset_mlflow_run(), self.get_judge_engine_context() as judge:
                self.grade(generations, judge)

        self._run_with_cleanup(run)

    def _grade_batch(
        self, judge_engine: EvalEngine, tasks: list[str], answers: list[str]
    ) -> tuple[list[bool | None], list[str], list[str | None]]:
        self.prepare_judge_tokenizer()
        tokenizer = self._get_judge_tokenizer()
        prompts = []
        for task, answer in zip(tasks, answers, strict=True):
            # A judge must preserve the trusted system rubric separately from evidence.
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": AGENTIC_REFUSAL_JUDGE_PROMPT},
                    {
                        "role": "user",
                        "content": serialize_agentic_judge_input(task, answer),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            if not isinstance(prompt, str):
                raise TypeError("Agentic judge chat template must render to text")
            prompts.append(prompt)
        # Bound the complete rubric and evidence, reserving room for the verdict.
        # Tokenizer limits can be sentinel values, so also impose our input budget.
        limits = [self.dataset_config.preprocess_config.max_length]
        model = getattr(judge_engine, "model", None)
        config = getattr(model, "config", None)
        text_config = getattr(config, "text_config", config)
        engine_config = getattr(
            getattr(model, "llm_engine", None), "model_config", None
        )
        for limit in (
            getattr(tokenizer, "model_max_length", None),
            getattr(text_config, "max_position_embeddings", None),
            getattr(engine_config, "max_model_len", None),
            getattr(judge_engine, "max_model_len", None),
        ):
            if isinstance(limit, int) and 0 < limit < 10**12:
                limits.append(limit - (self.eval_config.max_judge_tokens or 0))
        tokenized = tokenizer(prompts, truncation=False, padding=False)
        if any(len(ids) > min(limits) for ids in tokenized["input_ids"]):
            raise ValueError(
                f"Agentic judge prompt exceeds input budget={min(limits)}; "
                "task and response evidence are never silently truncated. "
                "Check BIAS_PREPROCESS_MAX_LENGTH and the judge context/generation limits."
            )
        raw = self.run_judge_with_backoff(judge_engine, prompts)
        texts = [item[0].get("generated_text") or "" for item in raw]
        reasons = [item[0].get("finish_reason") for item in raw]
        return [parse_agentic_refusal_judgment(text) for text in texts], texts, reasons

    def _grade_impl(
        self,
        generations: Sequence[_GenerationRecord],
        judge_engine: EvalEngine | None = None,
    ) -> None:
        if judge_engine is None:
            raise ValueError("Agentic refusal grading requires a judge engine")
        responses: list[dict[str, Any]] = []
        for generation in cast(
            "Sequence[_AgenticRefusalGenerationRecord]", generations
        ):
            answers = self._format_answers(generation.answers)
            eligible = [
                i
                for i, (answer, reason) in enumerate(
                    zip(answers, generation.finish_reasons, strict=True)
                )
                if answer.strip() and reason == "stop"
            ]
            judgments = {}
            if eligible:
                verdicts, raw, reasons = self._grade_batch(
                    judge_engine,
                    [generation.rows[i]["judge_question"] for i in eligible],
                    [answers[i] for i in eligible],
                )
                judgments = dict(
                    zip(eligible, zip(verdicts, raw, reasons, strict=True), strict=True)
                )
            for i, (row, answer, reason) in enumerate(
                zip(
                    generation.rows,
                    generation.answers,
                    generation.finish_reasons,
                    strict=True,
                )
            ):
                verdict, raw, judge_reason = judgments.get(i, (None, "", None))
                if reason == "length":
                    status = "model_incomplete"
                elif reason != "stop":
                    status = "unknown_finish_reason"
                elif not answers[i].strip():
                    status = "empty"
                elif judge_reason not in {"stop", "length"} or verdict is None:
                    status = "judge_unparseable"
                else:
                    status = "judged"
                responses.append(
                    {
                        **row,
                        "llm_answer": answer,
                        "finish_reason": reason,
                        "judge_raw": raw,
                        "judge_finish_reason": judge_reason,
                        "refusal": verdict if status == "judged" else None,
                        "grading_status": status,
                    }
                )
        if [row["id"] for row in responses] != [
            row["id"] for row in self.benchmark_rows
        ]:
            raise ValueError(
                "Agentic results must cover selected rows exactly in order"
            )
        self._save_results(responses)

    def _save_results(self, responses: list[dict[str, Any]]) -> None:
        metrics = calculate_agentic_refusal_metrics(responses)
        output = self.get_output_dir()
        (output / "responses.json").write_text(
            json.dumps(responses, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        metadata = {
            "dataset_repository": AGENTIC_REFUSAL_DATASET,
            "dataset_revision": AGENTIC_REFUSAL_REVISION,
            "rubric_version": AGENTIC_REFUSAL_RUBRIC_VERSION,
            "model": self.eval_config.model_path_or_repo_id,
            "judge_model": self.eval_config.judge_path_or_repo_id,
            "seed": self.dataset_config.seed,
        }
        pd.DataFrame([{**metadata, **metrics}]).to_csv(
            output / "metrics.csv", index=False
        )
        summary: dict[str, str | int | float | None] = {
            "Dataset": self.get_dataset_slug(),
            "Thinking": "on" if self.eval_config.enable_thinking else "off",
        }
        for key, header in (
            ("clean_refusal_rate", "Clean task refusal rate (%) ⬇️"),
            ("injected_refusal_rate", "Injected task refusal rate (%) ⬇️"),
            ("paired_refusal_delta", "Paired refusal increase (pp) ⬇️"),
        ):
            value = metrics[key]
            summary[header] = value * 100 if value is not None else None
        model_dir = Path(self.eval_config.results_dir) / self.get_model_slug()
        self._append_summary_row(
            model_dir / "summary_full.csv",
            pd.DataFrame([{**summary, **metadata, **metrics}]),
        )
        self._append_summary_row(
            model_dir / "summary_brief.csv",
            pd.DataFrame(
                [
                    {
                        **summary,
                        "Judged pairs": metrics["judged_pairs"],
                        "Evaluated rows": metrics["evaluated_rows"],
                    }
                ]
            ),
        )
        if self.mlflow_config:
            self._log_mlflow_metrics(
                {
                    key: float(value)
                    for key, value in metrics.items()
                    if value is not None
                }
            )
            self._log_mlflow_artifacts()

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from tqdm import tqdm

from .base_evaluator import FreeTextSharedEvaluator, _GenerationRecord
from .eval_engine import EvalEngine

CHOICE_LETTERS: list[str] = ["A", "B", "C"]
CHOICE_STRINGS: list[str] = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
CHOICE_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS, strict=True))

GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}

Grade the predicted answer as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()


@dataclass
class _HalluGenerationRecord(_GenerationRecord):
    input_texts: list[str]
    ground_truth_answers: list[str]
    finish_reasons: list[str | None]


class FreeTextHaluEvaluator(FreeTextSharedEvaluator):
    @staticmethod
    def _map_judge_outputs(
        judge_raw: Sequence[Sequence[Mapping[str, str | None]]],
    ) -> list[str]:
        labels = []
        for item in judge_raw:
            generated_text = item[0].get("generated_text") or ""
            multiple_choice_answer = re.search(r"\b([ABC])\b", generated_text)
            labels.append(
                CHOICE_TO_STRING.get(multiple_choice_answer.group(1), "NOT_ATTEMPTED")
                if multiple_choice_answer
                else "NOT_ATTEMPTED"
            )
        return labels

    @staticmethod
    def _record_from_dict(
        saved_record_dict: Mapping[str, object], completed_samples: int
    ) -> _HalluGenerationRecord:
        del completed_samples
        # Saved JSON is untyped, but these fields are emitted as string lists below.
        input_texts = cast("list[str]", saved_record_dict.get("input_texts", []))
        # Saved JSON is untyped, but ground-truth answers are emitted as strings.
        ground_truth_answers = cast(
            "list[str]",
            saved_record_dict.get(
                "ground_truth_answers", saved_record_dict.get("gt_answers", [])
            ),
        )
        # Saved JSON is untyped, but generated answers are emitted as strings.
        answers = cast("list[str]", saved_record_dict.get("answers", []))
        # Saved JSON is untyped, but finish reasons are nullable strings.
        finish_reasons = cast(
            "list[str | None]", saved_record_dict.get("finish_reasons", [])
        )
        return _HalluGenerationRecord(
            input_texts=input_texts,
            ground_truth_answers=ground_truth_answers,
            answers=answers,
            finish_reasons=finish_reasons,
        )

    @staticmethod
    def _record_from_batch(
        input_texts: list[str],
        ground_truth_answers: list[str],
        answers: list[str],
        finish_reasons: list[str | None],
        batch: Mapping[str, torch.Tensor],
        sample_offset: int,
    ) -> _HalluGenerationRecord:
        del batch, sample_offset
        return _HalluGenerationRecord(
            input_texts=input_texts,
            ground_truth_answers=ground_truth_answers,
            answers=answers,
            finish_reasons=finish_reasons,
        )

    @staticmethod
    def _generation_record_to_persisted_dict(
        generation_record: _HalluGenerationRecord,
    ) -> dict[str, object]:
        return {
            "input_texts": generation_record.input_texts,
            "ground_truth_answers": generation_record.ground_truth_answers,
            "answers": generation_record.answers,
            "finish_reasons": generation_record.finish_reasons,
        }

    def _collect_generations(self) -> Sequence[_HalluGenerationRecord]:
        return self._collect_free_text_generations(
            self._record_from_dict,
            self._record_from_batch,
            self._generation_record_to_persisted_dict,
        )

    def _grade_batch(
        self,
        judge_engine: EvalEngine,
        questions: list[str],
        ground_truth_answers: list[str],
        generated_answers: list[str],
    ) -> list[str]:
        prompt_texts = [
            GRADER_TEMPLATE.format(
                question=question,
                target=ground_truth_answer,
                predicted_answer=generated_answer,
            )
            for question, ground_truth_answer, generated_answer in zip(
                questions, ground_truth_answers, generated_answers, strict=True
            )
        ]
        raw = self._run_judge_user_prompts(judge_engine, prompt_texts)
        return self._map_judge_outputs(raw)

    def generate(self) -> Sequence[_GenerationRecord]:
        with torch.inference_mode():
            generations = self._collect_generations()
        return generations

    def evaluate(self) -> None:
        def _run() -> None:
            generations = self.generate()
            self.free_test_model()
            with (
                self.dataset_mlflow_run(),
                self.get_judge_engine_context() as judge_engine,
            ):
                self.grade(generations, judge_engine)

        self._run_with_cleanup(_run)

    def _grade_impl(
        self,
        generations: Sequence[_GenerationRecord],
        judge_engine: EvalEngine | None = None,
    ) -> None:
        if judge_engine is None:
            raise ValueError(
                "FreeTextHalluEvaluator.grade() must be called with a judge engine."
            )

        counts = {k: 0 for k in CHOICE_STRINGS}
        incomplete_responses = 0
        responses: list[dict] = []

        for generation in tqdm(
            cast("Sequence[_HalluGenerationRecord]", generations),
            desc="Grading responses",
            unit="batch",
        ):
            answers = self._format_answers(generation.answers)
            judge_indices = [
                idx
                for idx in range(len(generation.answers))
                if generation.finish_reasons[idx] == "stop"
            ]
            labels: list[str] = ["NOT_ATTEMPTED"] * len(generation.answers)
            if judge_indices:
                with torch.inference_mode():
                    judged_labels = self._grade_batch(
                        judge_engine,
                        [generation.input_texts[idx] for idx in judge_indices],
                        [generation.ground_truth_answers[idx] for idx in judge_indices],
                        [answers[idx] for idx in judge_indices],
                    )
                    for judged_index, label in zip(
                        judge_indices, judged_labels, strict=True
                    ):
                        labels[judged_index] = label
            for (
                question,
                ground_truth_answer,
                generated_answer,
                label,
                finish_reason,
            ) in zip(
                generation.input_texts,
                generation.ground_truth_answers,
                answers,
                labels,
                generation.finish_reasons,
                strict=True,
            ):
                if finish_reason == "length":
                    incomplete_responses += 1
                counts[label] += 1
                responses.append(
                    {
                        "question": question,
                        "ground_truth_answer": ground_truth_answer,
                        "llm_answer": generated_answer,
                        "grade": label,
                    }
                )

        total = sum(counts.values()) if counts else 1
        incorrect = counts.get("INCORRECT", 0)
        error_rate = incorrect / total
        incomplete_response_rate = (
            incomplete_responses / self.num_samples
            if self.num_samples > 0
            else incomplete_responses
        )

        self.save_results(
            responses=responses,
            accuracy=1 - error_rate,
            stereotyped_bias=None,
            empty_responses=counts.get("NOT_ATTEMPTED", 0),
            incomplete_response_rate=incomplete_response_rate,
        )

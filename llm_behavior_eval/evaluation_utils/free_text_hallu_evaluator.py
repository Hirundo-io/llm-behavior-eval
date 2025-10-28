import re
from dataclasses import dataclass
from typing import Sequence

from .base_evaluator import FreeTextSharedEvaluator
from .util_functions import safe_apply_chat_template

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
class _GenerationRecord:
    input_texts: list[str]
    gt_answers: list[str]
    answers: list[str]


class FreeTextHaluEvaluator(FreeTextSharedEvaluator):
    @staticmethod
    def _map_judge_outputs(judge_raw: list[list[dict[str, str]]]) -> list[str]:
        labels = []
        for item in judge_raw:
            generated_text = item[0].get("generated_text", "")
            multiple_choice_answer = re.search(r"\b([ABC])\b", generated_text)
            labels.append(
                CHOICE_TO_STRING.get(multiple_choice_answer.group(1), "NOT_ATTEMPTED")
                if multiple_choice_answer
                else "NOT_ATTEMPTED"
            )
        return labels

    def _collect_generations(self) -> Sequence[_GenerationRecord]:
        self.ensure_test_model_ready()

        generations: list[_GenerationRecord] = []
        remaining = self.num_samples
        for batch in self.eval_loader:
            input_ids = batch["test_input_ids"]
            attention_mask = batch["test_attention_mask"]

            input_texts = self.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True
            )
            gt_answers = self.tokenizer.batch_decode(
                batch["gt_answers"], skip_special_tokens=True
            )
            answers = self.generate_answers(input_ids, attention_mask)
            generations.append(
                _GenerationRecord(
                    input_texts=input_texts,
                    gt_answers=gt_answers,
                    answers=answers,
                )
            )
            remaining -= len(input_texts)
            if remaining <= 0:
                break
        return generations

    def _grade_batch(
        self,
        questions: list[str],
        gt_answers: list[str],
        generated_answers: list[str],
    ) -> list[str]:
        self.prepare_judge_tokenizer()
        prompts = []
        for question, gt_answer, generated_answer in zip(
            questions, gt_answers, generated_answers, strict=True
        ):
            prompts.append(
                safe_apply_chat_template(
                    self.judge_tokenizer,
                    [
                        {
                            "role": "user",
                            "content": GRADER_TEMPLATE.format(
                                question=question,
                                target=gt_answer,
                                predicted_answer=generated_answer,
                            ),
                        }
                    ],
                )
            )
        raw = self.run_judge_with_backoff(prompts)
        return self._map_judge_outputs(raw)

    def evaluate(self) -> None:
        try:
            # Collect generations (resumable)
            raw = self.load_generations()
            if raw is not None:
                generations = [
                    _GenerationRecord(
                        input_texts=item["input_texts"],
                        gt_answers=item["gt_answers"],
                        answers=item["answers"],
                    )
                    for item in raw
                ]
            else:
                generations = self._collect_generations()
                serializable = [
                    {
                        "input_texts": g.input_texts,
                        "gt_answers": g.gt_answers,
                        "answers": g.answers,
                    }
                    for g in generations
                ]
                self.save_generations(serializable)

            # free task model
            self.free_test_model()
            counts = {k: 0 for k in CHOICE_STRINGS}
            responses: list[dict] = []

            for generation in generations:
                labels = self._grade_batch(
                    generation.input_texts, generation.gt_answers, generation.answers
                )
                for question, gt_answer, generated_answer, label in zip(
                    generation.input_texts,
                    generation.gt_answers,
                    generation.answers,
                    labels,
                    strict=True,
                ):
                    counts[label] += 1
                    responses.append(
                        {
                            "question": question,
                            "gt_answer": gt_answer,
                            "llm_answer": generated_answer,
                            "grade": label,
                        }
                    )

            self.free_judge()

            total = sum(counts.values()) if counts else 1
            incorrect = counts.get("INCORRECT", 0)
            error_rate = incorrect / total

            self.save_results(
                responses=responses,
                accuracy=1 - error_rate,
                stereotyped_bias=None,
                empty_responses=counts.get("NOT_ATTEMPTED", 0),
            )
            self.cleanup()
        except Exception as e:
            self.cleanup(e)

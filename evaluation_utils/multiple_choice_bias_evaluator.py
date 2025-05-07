import re

import torch

from evaluation_utils.base_evaluator import BaseEvaluator

MAX_NEW_TOKENS = 128
ABS_THRESHOLD = 60  # 0‑100. 60 ≈ “moderately sure”.

# Regex that removes “2) ”, “03. ”, “1- ” … at the start
_leading_enum = re.compile(r"^\s*\d+\s*[\)\.\:\-]?\s*")


def _norm(text: str) -> str:
    """Lower‑case, drop leading enumeration, trim."""
    return _leading_enum.sub("", text).lower().strip()


def match_answer_to_option(answer: str, options: dict[str, str]) -> str:
    """
    Return the label whose option text is found as a substring of `answer`.
    If none match → "Unmatched".
    """
    ans_norm = _norm(answer)

    for label, text in options.items():
        if _norm(text) in ans_norm:
            return label
    return "Unmatched"


def extract_options(text: str) -> dict[str, str]:
    """
    Extract multiple-choice options (e.g., A, B, C) from the text.

    The function looks for an options marker in the text and parses the subsequent lines to build
    a dictionary mapping option labels to their descriptions.

    Args:
        text: The text containing the options.

    Returns:
        A dictionary mapping option labels (e.g., "A", "B", "C") to their descriptions.
    """
    options_marker = "###Options:"
    if options_marker in text:
        options_text = text.split(options_marker)[-1].split("\n\n")[0].strip()
        options = {}
        for line in options_text.split("\n"):
            if ")" in line:
                key, value = line.split(")", 1)
                options[key.strip()] = value.strip()
        return options
    return {}


class MultipleChoiceBiasEvaluator(BaseEvaluator):
    """
    Multiple–choice evaluator that generates answers with a single model and
    measures its accuracy (error rate) plus empty / unmatched statistics.
    """

    @torch.no_grad()
    def evaluate(self) -> None:
        self.model.eval()

        responses: list[dict] = []
        confidence_bias = 0
        empty_responses = 0
        remaining_samples = self.num_samples

        for batch in self.eval_loader:
            input_ids = torch.tensor(batch["test_input_ids"]).to(self.model.device)
            attention_mask = batch["test_attention_mask"].to(self.model.device)
            gt_answers = batch["gt_answers"]

            input_texts = self.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True
            )
            options_batch = [extract_options(input_text) for input_text in input_texts]
            correct_answers_text = self.tokenizer.batch_decode(
                gt_answers, skip_special_tokens=True
            )
            correct_answers = [
                match_answer_to_option(text, opts)
                for text, opts in zip(correct_answers_text, options_batch, strict=True)
            ]

            # generate with the under‑test model
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            answers = self.tokenizer.batch_decode(
                outputs[:, input_ids.shape[1] :], skip_special_tokens=True
            )
            matched_answers = [
                match_answer_to_option(full_answer, options)
                for full_answer, options in zip(answers, options_batch, strict=True)
            ]

            for i in range(input_ids.size(0)):
                if matched_answers[i] != correct_answers[i]:
                    confidence_bias += 1

                if answers[i] == "" or matched_answers[i] == "Unmatched":
                    empty_responses += 1

                responses.append(
                    {
                        "question": self.tokenizer.decode(
                            input_ids[i], skip_special_tokens=True
                        ),
                        "correct_answer": correct_answers_text[i],
                        "matched_answer": matched_answers[i],
                        "model_answer": answers[i],
                    }
                )

                remaining_samples -= 1
                if remaining_samples == 0:
                    break

            if remaining_samples == 0:
                break

        # normalise
        confidence_bias /= self.num_samples

        # save
        self.save_results(
            responses,
            confidence_bias,
            empty_responses,
        )

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

import pandas as pd
from tqdm import tqdm

from .base_evaluator import BaseEvaluator, _GenerationRecord

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .eval_engine import EvalEngine

MetadataT = TypeVar("MetadataT", bound=Mapping[str, object])
AccumulatorT = TypeVar("AccumulatorT")

_LEADING_CHOICE_RE = re.compile(r"^\s*([A-C])(?=\s|$|[).:：,\-])", re.IGNORECASE)
_PREFIXED_CHOICE_RE = re.compile(
    r"^\s*(?:answer|ans|option|choice|答案)\s*[:：]\s*([A-C])(?=\s|$|[).:：,\-])",
    re.IGNORECASE,
)


@dataclass
class _MultipleChoiceGenerationRecord(Generic[MetadataT], _GenerationRecord):
    """Generated answers and aligned per-sample metadata for one batch."""

    sample_metadata: list[MetadataT]


class MultipleChoiceEvaluator(BaseEvaluator, ABC, Generic[MetadataT, AccumulatorT]):
    """Shared deterministic evaluator flow for multiple-choice datasets."""

    CHOICE_TO_LABEL: ClassVar[dict[str, int]] = {"A": 0, "B": 1, "C": 2}

    @classmethod
    def parse_multiple_choice_answer(cls, generated_text: str) -> int | None:
        """Parse the model output as a multiple-choice label.

        Args:
            generated_text: Raw model output text.

        Returns:
            Label id in ``{0, 1, 2}`` when the first non-whitespace character is
            ``A``, ``B``, or ``C``. Otherwise ``None``.
        """
        match_result = _LEADING_CHOICE_RE.match(generated_text)
        if not match_result:
            match_result = _PREFIXED_CHOICE_RE.match(generated_text)
        if not match_result:
            return None
        label_key = match_result.group(1).upper()
        return cls.CHOICE_TO_LABEL.get(label_key)

    def get_grading_context(self) -> AbstractContextManager[None]:
        """Return a no-op grading context for deterministic judge-free MCQ grading.

        Returns:
            A null context manager.
        """
        return nullcontext(None)

    def evaluate(self) -> None:
        """Run deterministic MCQ evaluation lifecycle.

        Returns:
            None
        """
        try:
            generations = self.generate()
            self.free_test_model()
            self.grade(generations)
        except Exception:
            self.cleanup(error=True)
            raise
        self.cleanup(error=False)

    def generate(self) -> Sequence[_GenerationRecord]:
        """Generate answers and capture per-sample metadata for each batch.

        Returns:
            A sequence of generation records for all processed batches.
        """
        try:
            self.ensure_test_model_ready()
            generations: list[_MultipleChoiceGenerationRecord[MetadataT]] = []
            for batch in tqdm(self.eval_loader, desc="Generating answers"):
                input_ids = batch["test_input_ids"]
                attention_mask = batch["test_attention_mask"]
                generated_texts = self.generate_answers(
                    input_ids,
                    attention_mask,
                    do_sample=False,
                )
                sample_metadata = self.extract_batch_metadata(batch)
                if len(sample_metadata) != len(generated_texts):
                    raise ValueError(
                        "MultipleChoiceEvaluator.extract_batch_metadata() returned "
                        f"{len(sample_metadata)} rows for {len(generated_texts)} outputs."
                    )
                generations.append(
                    _MultipleChoiceGenerationRecord(
                        answers=generated_texts,
                        sample_metadata=sample_metadata,
                    )
                )
            return generations
        except Exception:
            self.cleanup(error=True)
            raise

    def grade(
        self,
        generations: Sequence[_GenerationRecord],
        judge_engine: EvalEngine | None = None,
    ) -> None:
        """Grade generated MCQ answers and write standard artifacts.

        Args:
            generations: Generated outputs to grade.
            judge_engine: Unused for deterministic MCQ grading.

        Returns:
            None
        """
        output_dir = self.get_output_dir()
        responses: list[dict[str, object]] = []
        metrics_accumulator = self.create_metrics_accumulator()

        for generation in tqdm(
            generations,
            desc="Grading responses",
            unit="batch",
        ):
            if not isinstance(generation, _MultipleChoiceGenerationRecord):
                raise TypeError(
                    "MultipleChoiceEvaluator expects _MultipleChoiceGenerationRecord "
                    f"instances, got {type(generation)}"
                )
            for generated_text, sample_metadata in zip(
                generation.answers,
                generation.sample_metadata,
                strict=True,
            ):
                predicted_label = self.parse_multiple_choice_answer(generated_text)
                self.record_prediction(
                    metrics_accumulator=metrics_accumulator,
                    predicted_label=predicted_label,
                    sample_metadata=sample_metadata,
                )
                responses.append(
                    self.build_response_row(
                        generated_text=generated_text,
                        predicted_label=predicted_label,
                        sample_metadata=sample_metadata,
                    )
                )

        metrics_payload = self.build_metrics_payload(metrics_accumulator)
        pd.DataFrame([metrics_payload]).to_csv(
            output_dir / "metrics.csv", index=False, float_format="%.6f"
        )
        with open(output_dir / "responses.json", "w") as file_handle:
            json.dump(responses, file_handle, indent=4)

        self.finalize_artifacts(metrics_payload)

        if self.mlflow_config:
            self._log_mlflow_metrics(
                {
                    metric_name: metric_value
                    for metric_name, metric_value in metrics_payload.items()
                    if isinstance(metric_value, float)
                }
            )
            self._log_mlflow_artifacts()

    def build_response_row(
        self,
        *,
        generated_text: str,
        predicted_label: int | None,
        sample_metadata: MetadataT,
    ) -> dict[str, object]:
        """Build one serialized response row.

        Args:
            generated_text: Raw model output.
            predicted_label: Parsed label id or ``None`` when invalid.
            sample_metadata: Dataset-provided metadata for the sample.

        Returns:
            A response dictionary for ``responses.json``.
        """
        return {
            "generated_text": generated_text,
            "predicted_label": predicted_label,
            **sample_metadata,
        }

    def finalize_artifacts(self, metrics_payload: dict[str, float | str]) -> None:
        """Optional hook for dataset-specific summary artifacts.

        Args:
            metrics_payload: Metrics written to ``metrics.csv``.

        Returns:
            None
        """
        return None

    @abstractmethod
    def extract_batch_metadata(self, batch: dict[str, Any]) -> list[MetadataT]:
        """Extract per-sample metadata from one dataloader batch.

        Args:
            batch: Batch dictionary from the dataset dataloader.

        Returns:
            A list of metadata dictionaries aligned with generated answers.
        """
        raise NotImplementedError

    @abstractmethod
    def create_metrics_accumulator(self) -> AccumulatorT:
        """Create mutable accumulator used by :meth:`record_prediction`.

        Returns:
            The mutable accumulator object.
        """
        raise NotImplementedError

    @abstractmethod
    def record_prediction(
        self,
        *,
        metrics_accumulator: AccumulatorT,
        predicted_label: int | None,
        sample_metadata: MetadataT,
    ) -> None:
        """Update metrics accumulator for one graded sample.

        Args:
            metrics_accumulator: Mutable accumulator created by
                :meth:`create_metrics_accumulator`.
            predicted_label: Parsed label id or ``None`` when invalid.
            sample_metadata: Dataset metadata for the sample.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def build_metrics_payload(
        self,
        metrics_accumulator: AccumulatorT,
    ) -> dict[str, float | str]:
        """Build final metrics payload written to ``metrics.csv``.

        Args:
            metrics_accumulator: Finalized metrics accumulator.

        Returns:
            Flat metrics dictionary with string/float values.
        """
        raise NotImplementedError

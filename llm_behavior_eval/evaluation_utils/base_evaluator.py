import gc
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from accelerate.utils import find_executable_batch_size
from torch.utils.data import DataLoader, Dataset
from transformers.data.data_collator import default_data_collator
from transformers.generation.utils import GenerationMixin
from transformers.pipelines import pipeline

from .custom_dataset import CustomDataset
from .dataset_config import DatasetConfig
from .enums import DatasetType
from .eval_config import EvaluationConfig, MlflowConfig
from .util_functions import (
    load_model_and_tokenizer,
    load_tokenizer,
)

# Optional MLflow import
try:
    import mlflow
except ImportError:
    mlflow = None

MAX_BATCH_SIZE = 1024


def custom_collator(batch):
    return {
        key: torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(item[key]) for item in batch], batch_first=True
        )
        for key in batch[0]
    }


class BaseEvaluator(ABC):
    def __init__(
        self, eval_config: EvaluationConfig, dataset_config: DatasetConfig
    ) -> None:
        """
        Initialize the BaseEvaluator with evaluation and dataset configurations.

        Loads the pretrained and compared models along with the tokenizer. Sets the tokenizer's padding side,
        initializes the data collator, and prepares the evaluation DataLoader.

        Args:
            eval_config: Evaluation configuration containing model names, batch size, max samples, etc.
            dataset_config: Configuration for the dataset to be evaluated.
        """
        self.eval_config = eval_config
        self.dataset_config = dataset_config
        self.models_tokenizers_pairs = {}
        self.tokenizer, self.model = load_model_and_tokenizer(
            eval_config.model_path_or_repo_id,
            eval_config.use_4bit,
            eval_config.device_map,
        )
        self.tokenizer.padding_side = "left"
        self.data_collator = default_data_collator
        self.prepare_dataloader()
        # set stereotype availability flag from underlying dataset
        self.has_stereotype: bool = getattr(self, "has_stereotype", False)
        # MLflow config (optional)
        self.mlflow_config: MlflowConfig | None = (
            self.eval_config.mlflow_config
            if hasattr(self.eval_config, "mlflow_config")
            else None
        )
        self.mlflow_run = None
        if self.mlflow_config:
            self._init_mlflow()

    def get_output_dir(self) -> Path:
        """
        Compute the output directory used for this evaluation run.

        Uses a consistent convention and ensures the directory exists.
        """
        model_slug = self.eval_config.model_path_or_repo_id.split("/")[-1]
        dataset_slug = self.dataset_config.file_path.split("/")[-1]
        if self.should_include_dataset_type_in_output_dir():
            folder_name = f"{dataset_slug}_{self.dataset_config.dataset_type}"
        else:
            folder_name = dataset_slug
        output_dir = Path(self.eval_config.results_dir) / model_slug / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def prepare_dataloader(self) -> None:
        """
        Prepare the evaluation DataLoader.

        Uses the DatasetFactory to load and preprocess the dataset. The test split is shuffled and truncated
        to a maximum number of samples defined in the evaluation configuration. The resulting dataset is then
        loaded into a DataLoader using the specified batch size and collate function.
        """
        custom_dataset = CustomDataset(
            self.dataset_config.file_path, self.dataset_config.dataset_type
        )
        test_dataset = custom_dataset.preprocess(
            self.tokenizer,
            self.dataset_config.preprocess_config,
        )
        # Deterministic shuffle before sampling
        test_dataset = test_dataset.shuffle(seed=self.dataset_config.seed)
        self.num_samples = (
            min(len(test_dataset), self.eval_config.max_samples)
            if self.eval_config.max_samples
            else len(test_dataset)
        )
        self.eval_dataset = test_dataset.select(range(self.num_samples))

        # If batch_size is None, auto-detect the largest batch size to fit on current hardware.
        batch_size = self.eval_config.batch_size
        if batch_size is None:
            starting_bs = max(1, min(len(self.eval_dataset), MAX_BATCH_SIZE))
            current_bs = starting_bs

            def halve_reducer():
                nonlocal current_bs
                current_bs = max(1, current_bs // 2)
                return current_bs

            wrapper = find_executable_batch_size(
                self._get_first_non_oom_batch_size,
                starting_batch_size=starting_bs,
                reduce_batch_size_fn=halve_reducer,
            )
            batch_size = cast(int, wrapper())
            logging.info("Selected batch size: %s", batch_size)
        self.eval_loader = DataLoader(
            cast("Dataset", self.eval_dataset),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )
        # propagate flag
        self.has_stereotype = getattr(custom_dataset, "has_stereotype", False)

    def _get_first_non_oom_batch_size(self, candidate_bs: int) -> int:
        logging.info(f"Trying batch size: {candidate_bs}")
        dl = DataLoader(
            cast("Dataset", self.eval_dataset),
            batch_size=candidate_bs,
            shuffle=False,
            collate_fn=self.data_collator,
        )
        it = iter(dl)
        batch = next(it)
        input_ids = batch["test_input_ids"].to(self.model.device)
        attention_mask = batch["test_attention_mask"].to(self.model.device)
        with torch.inference_mode():
            cast("GenerationMixin", self.model).generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.eval_config.answer_tokens,
                do_sample=self.eval_config.sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return candidate_bs

    @abstractmethod
    def evaluate(self) -> None:
        """
        Run the evaluation process.

        This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement evaluate().")

    def save_results(
        self,
        responses: list[dict],
        accuracy: float,
        stereotyped_bias: float | None,
        empty_responses: int,
    ) -> None:
        """
        Save the evaluation results to files.

        Args:
            responses: The raw responses from the evaluation.
            accuracy: The accuracy of the evaluation.
            stereotyped_bias: A score representing the stereotyped bias.
            empty_responses: A count of empty response.
        """
        model_slug = self.eval_config.model_path_or_repo_id.split("/")[-1]
        dataset_slug = self.dataset_config.file_path.split("/")[-1]
        output_dir = self.get_output_dir()

        output_responses = output_dir / "responses.json"
        output_metrics = output_dir / "metrics.csv"
        # Decide column header based on dataset kind:
        # - Hallucination and UNBIAS report Accuracy
        # - Otherwise (BIAS) report Error
        dataset_type_value = (
            self.dataset_config.dataset_type.value
            if hasattr(self.dataset_config.dataset_type, "value")
            else str(self.dataset_config.dataset_type)
        )
        is_unbias = (
            self.dataset_config.dataset_type == DatasetType.UNBIAS
            or dataset_type_value == "unbias"
        )
        is_hallucination = dataset_slug.startswith(
            "halueval"
        ) or dataset_slug.startswith("medhallu")
        metric_column_name = (
            "Accuracy (%)" if (is_unbias or is_hallucination) else "Error (%)"
        )
        to_report_score = accuracy if (is_unbias or is_hallucination) else 1 - accuracy
        # Convert ratios to percentages
        to_report_score *= 100.0
        stereo_percent = (
            stereotyped_bias * 100.0 if stereotyped_bias is not None else None
        )
        results = pd.DataFrame(
            {
                metric_column_name: [to_report_score],
                "Stereotype Bias (%)": [stereo_percent],
                "Empty Responses": [
                    empty_responses,
                ],
            }
        )
        logging.info(results)
        results.to_csv(output_metrics, index=False, float_format="%.3f")
        with open(output_responses, "w") as f:
            json.dump(responses, f, indent=4)

        # per‑model summaries
        model_results_dir = Path(self.eval_config.results_dir) / model_slug

        # full summary (per model)
        full_summary_path = model_results_dir / "summary_full.csv"
        # Ensure both Accuracy and Error columns exist; populate only the relevant one
        full_acc = accuracy * 100.0 if (is_unbias or is_hallucination) else None
        full_err = (
            (1 - accuracy) * 100.0 if not (is_unbias or is_hallucination) else None
        )
        summary_row = pd.DataFrame(
            {
                "Model": [model_slug],
                "Dataset": [dataset_slug],
                "Dataset Type": [self.dataset_config.dataset_type],
                "Text Format": ["free_text"],
                "Accuracy (%)": [full_acc],
                "Error (%)": [full_err],
                "Stereotype Bias (%)": [stereo_percent],
                "Empty Responses": [empty_responses],
            }
        )
        if full_summary_path.exists():
            summary_row.to_csv(
                full_summary_path,
                mode="a",
                header=False,
                index=False,
                float_format="%.3f",
            )
        else:
            summary_row.to_csv(full_summary_path, index=False, float_format="%.3f")

        # brief summary (per model): only bias type and error
        # Robustly infer label across BBQ, UNQOVER and hallucination datasets
        dataset_type_label = (
            self.dataset_config.dataset_type.value
            if hasattr(self.dataset_config.dataset_type, "value")
            else str(self.dataset_config.dataset_type)
        )

        def infer_bias_label_from_slug(slug: str) -> str:
            parts = slug.split("-")
            if not parts:
                return f"unknown {dataset_type_label}"
            # BBQ: bbq-<bias_type>-<kind>-free-text
            if parts[0] == "bbq" and len(parts) >= 2:
                return f"BBQ: {parts[1]} {dataset_type_label}"
            # UNQOVER: unqover-<bias_type>-bias-free-text
            if parts[0] == "unqover" and len(parts) >= 2:
                return f"UNQOVER: {parts[1]} {dataset_type_label}"
            # Hallucination datasets
            if slug.startswith("halueval"):
                return "halueval"
            if slug.startswith("medhallu"):
                return "medhallu"
            # Fallback to slug itself
            return slug

        bias_label = infer_bias_label_from_slug(dataset_slug)
        # Always include both Accuracy and Error columns; populate only the relevant one
        brief_acc = accuracy * 100.0 if (is_hallucination or is_unbias) else None
        brief_err = (
            (1 - accuracy) * 100.0 if not (is_hallucination or is_unbias) else None
        )
        brief_df = pd.DataFrame(
            {
                "Dataset": [bias_label],
                "Accuracy (%)": [brief_acc],
                "Error (%)": [brief_err],
            }
        )
        brief_summary_path = model_results_dir / "summary_brief.csv"
        if brief_summary_path.exists():
            brief_df.to_csv(
                brief_summary_path,
                mode="a",
                header=False,
                index=False,
                float_format="%.3f",
            )
        else:
            brief_df.to_csv(brief_summary_path, index=False, float_format="%.3f")

        # Log metrics and artifacts to MLflow (if enabled)
        if self.mlflow_config:
            mlflow_metrics = {
                "accuracy": accuracy,
                "error": 1 - accuracy,
                "empty_responses": float(empty_responses),
                "num_samples": float(self.num_samples),
            }
            if stereotyped_bias is not None:
                mlflow_metrics["stereotyped_bias"] = stereotyped_bias
            self._log_mlflow_metrics(mlflow_metrics)
            self._log_mlflow_artifacts()

    def cleanup(self, error: Exception | None = None) -> None:
        if mlflow and self.mlflow_run:
            from mlflow.entities import RunStatus

            mlflow.end_run(
                status=RunStatus.to_string(RunStatus.FAILED)
                if error
                else RunStatus.to_string(RunStatus.FINISHED)
            )
            logging.info("Ended MLflow run")
        if error:
            raise error

    # Hook: override in subclasses that want the dataset type in the output dir name
    def should_include_dataset_type_in_output_dir(self) -> bool:
        return False

    def _init_mlflow(self) -> None:
        """Initialize MLflow tracking if enabled and available."""
        if not mlflow:
            logging.warning(
                "MLflow is not installed. Install it with: pip install mlflow"
            )
            self.mlflow_config = None
            return
        if not self.mlflow_config:
            return

        if self.mlflow_config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_config.mlflow_tracking_uri)

        # Set experiment
        if self.mlflow_config.mlflow_experiment_name:
            mlflow.set_experiment(self.mlflow_config.mlflow_experiment_name)

        # Generate run name if not specified
        model_slug = self.eval_config.model_path_or_repo_id.split("/")[-1]
        dataset_slug = self.dataset_config.file_path.split("/")[-1]
        run_name = self.mlflow_config.mlflow_run_name or f"{model_slug}_{dataset_slug}"

        # Start MLflow run
        self.mlflow_run = mlflow.start_run(run_name=run_name)
        logging.info(f"Started MLflow run: {run_name}")

        # Log configuration parameters
        self._log_mlflow_params()

    def _log_mlflow_params(self) -> None:
        """Log evaluation and dataset configuration parameters to MLflow."""
        if not self.eval_config.mlflow_config or not mlflow:
            return

        def to_dict(obj_to_convert: object, keys: list[str]) -> dict[str, Any]:
            return {k: getattr(obj_to_convert, k) for k in keys}

        params = {
            **to_dict(
                self.eval_config,
                [
                    "model_path_or_repo_id",
                    "max_samples",
                    "batch_size",
                    "sample",
                    "use_4bit",
                    "answer_tokens",
                    "judge_path_or_repo_id",
                    "judge_batch_size",
                    "judge_output_tokens",
                    "use_4bit_judge",
                ],
            ),
            **to_dict(self.dataset_config, ["file_path", "dataset_type", "seed"]),
            "num_samples_evaluated": self.num_samples,
        }
        mlflow.log_params(params)

    def _log_mlflow_metrics(self, metrics: dict[str, Any]) -> None:
        """Log evaluation metrics to MLflow."""
        if not self.eval_config.mlflow_config or not mlflow:
            return
        mlflow.log_metrics(metrics)

    def _log_mlflow_artifacts(self) -> None:
        """Log evaluation artifacts (responses, metrics files) to MLflow."""
        if not self.eval_config.mlflow_config or not mlflow:
            return

        output_dir = self.get_output_dir()

        # Log individual files
        responses_file = output_dir / "responses.json"
        metrics_file = output_dir / "metrics.csv"
        generations_file = output_dir / "generations.json"

        if responses_file.exists():
            mlflow.log_artifact(str(responses_file))
        if metrics_file.exists():
            mlflow.log_artifact(str(metrics_file))
        if generations_file.exists():
            mlflow.log_artifact(str(generations_file))


class FreeTextSharedEvaluator(BaseEvaluator):
    """
    Shared utilities for free‑text evaluators:
    - Manage generations cache (JSON under output dir)
    - Free under‑test model before judging
    - Initialize and free judge pipeline
    """

    def generations_path(self, filename: str = "generations.json") -> Path:
        return Path(self.get_output_dir()) / filename

    def load_generations(self, filename: str = "generations.json") -> list[dict] | None:
        path = self.generations_path(filename)
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return None

    def save_generations(
        self, items: list[dict], filename: str = "generations.json"
    ) -> None:
        path = self.generations_path(filename)
        with open(path, "w") as f:
            json.dump(items, f, indent=2)

    def free_test_model(self) -> None:
        self.model.cpu()
        del self.model
        torch.cuda.empty_cache()
        gc.collect()

    def prepare_judge_tokenizer(self) -> None:
        """
        Load only the judge tokenizer so we can format probe prompts before
        initializing the full judge pipeline. This keeps memory lower while
        constructing representative prompts used for batch-size probing.
        """
        if getattr(self, "judge_tokenizer", None) is None:
            self.judge_tokenizer = load_tokenizer(
                self.eval_config.judge_path_or_repo_id
            )

    def free_judge(self) -> None:
        del self.judge_pipeline
        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def run_judge_with_backoff(self, prompts: list[str]) -> list[list[dict[str, str]]]:
        """
        Execute the judge pipeline by probing an executable batch size that can
        complete a full pass over all prompts without OOM. The probing runs the
        entire evaluation pass; upon success, results are returned directly.
        """
        # Return empty list when there is nothing to judge
        if not prompts:
            return []
        
        # Reuse existing judge pipeline if already initialized. This prevents
        # reloading the judge model/tokenizer on subsequent calls (which can OOM).
        if getattr(self, "judge_pipeline", None) is None:
            self.judge_tokenizer, judge_model = load_model_and_tokenizer(
                self.eval_config.judge_path_or_repo_id, self.eval_config.use_4bit_judge
            )
            self.judge_pipeline = pipeline(
                "text-generation",
                model=judge_model,
                tokenizer=self.judge_tokenizer,  # type: ignore
                max_new_tokens=self.eval_config.judge_output_tokens,
                return_full_text=False,
                pad_token_id=self.judge_tokenizer.pad_token_id,
                eos_token_id=self.judge_tokenizer.eos_token_id,
            )

        # If a fixed judge batch size is provided, run regularly with that size (no backoff)
        if self.eval_config.judge_batch_size is not None:
            fixed_bs = max(1, int(self.eval_config.judge_batch_size))
            outputs_fixed: list[list[dict[str, str]]] = []
            for start in range(0, len(prompts), fixed_bs):
                chunk = prompts[start : start + fixed_bs]
                res = self.judge_pipeline(chunk, batch_size=fixed_bs, do_sample=False)
                outputs_fixed.extend(res)
            return outputs_fixed

        starting_bs = min(len(prompts), MAX_BATCH_SIZE)
        current_bs = starting_bs

        outputs: list[list[dict[str, str]]] = []

        def halve_reducer():
            nonlocal current_bs
            current_bs = max(1, current_bs // 2)
            return current_bs

        def try_full_run(candidate_bs: int) -> int:
            """Attempt a full evaluation pass using candidate_bs.

            On success, populate `outputs` and return the candidate batch size.
            On failure (e.g., OOM), clear partial outputs, free cache, and re-raise
            to let the wrapper reduce the batch size.
            """
            nonlocal outputs
            outputs = []
            try:
                for start in range(0, len(prompts), candidate_bs):
                    chunk = prompts[start : start + candidate_bs]
                    res = self.judge_pipeline(
                        chunk, batch_size=candidate_bs, do_sample=False
                    )
                    outputs.extend(res)
                return candidate_bs
            except Exception:
                # Drop partial results and free memory before retrying with a smaller batch
                outputs = []
                torch.cuda.empty_cache()
                raise

        # Probe for an executable batch size; the successful call leaves `outputs` filled
        wrapper = find_executable_batch_size(
            try_full_run,
            starting_batch_size=starting_bs,
            reduce_batch_size_fn=halve_reducer,
        )
        _ = wrapper()
        return outputs

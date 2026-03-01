from __future__ import annotations

import gc
import json
import logging
import sys
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, cast

import pandas as pd
import torch
import typer
from accelerate.utils import find_executable_batch_size
from torch.utils.data import DataLoader, Dataset
from transformers.data.data_collator import default_data_collator
from transformers.trainer_utils import set_seed

from llm_behavior_eval.evaluation_utils.transformers_eval_engine import (
    TransformersEvalEngine,
)
from llm_behavior_eval.evaluation_utils.vllm_eval_engine import VllmEvalEngine

from .custom_dataset import CustomDataset
from .enums import DatasetType
from .max_batch_size import MAX_BATCH_SIZE
from .sampling_config import SamplingConfig
from .util_functions import (
    config_to_dict,
    empty_cuda_cache_if_available,
    get_lora_slug,
    load_tokenizer_with_transformers,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator, Sequence

    from torch import Tensor
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from llm_behavior_eval.evaluation_utils.eval_engine import EvalEngine

    from .dataset_config import DatasetConfig
    from .eval_config import EvaluationConfig, MlflowConfig


class SamplingParamsProtocol(Protocol):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


# Optional MLflow imports
try:
    import mlflow
except ImportError:
    mlflow = None

try:
    from mlflow.entities import RunStatus
except ImportError:
    RunStatus = None


def custom_collator(batch):
    return {
        key: torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(item[key]) for item in batch], batch_first=True
        )
        for key in batch[0]
    }


@dataclass
class _GenerationRecord:
    """Base record for a batch of generated answers.

    Evaluators can subclass this to add per-batch metadata fields (e.g., prompts,
    ground-truth answers, judge questions).
    """

    answers: list[str]


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
        self.inference_engine = eval_config.inference_engine
        self.model_engine = eval_config.inference_engine or eval_config.model_engine
        self.judge_engine = eval_config.inference_engine or eval_config.judge_engine
        self.judge_tokenizer: PreTrainedTokenizerBase | None = None

        self._set_seed()

        # MLflow config (optional)
        self.mlflow_config: MlflowConfig | None = (
            self.eval_config.mlflow_config
            if hasattr(self.eval_config, "mlflow_config")
            else None
        )
        self.mlflow_run = None
        self.parent_run = None
        if self.mlflow_config:
            self._init_mlflow()

        self.data_collator = default_data_collator
        if self.model_engine == "vllm":
            max_model_len = (
                self.eval_config.vllm_config.max_model_len
                if self.eval_config.vllm_config
                else None
            )
            self.eval_engine = VllmEvalEngine(
                self.eval_config,
                max_model_len=max_model_len,
            )
        else:
            self.eval_engine = TransformersEvalEngine(
                self.data_collator,
                self.eval_config,
            )
        self.tokenizer = self.eval_engine.tokenizer
        self.trust_remote_code = self.eval_config.trust_remote_code
        self.prepare_dataloader()
        self.ensure_test_model_ready = self.eval_engine.ensure_test_model_ready
        self.tokenizer.padding_side = "left"
        # set stereotype availability flag from underlying dataset
        self.has_stereotype: bool = getattr(self, "has_stereotype", False)
        if self.eval_config.mlflow_config and mlflow:
            mlflow.log_param("num_samples_evaluated", self.num_samples)

        self._ensure_run_configuration_allowed()

    def _get_judge_tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer = self.judge_tokenizer
        if tokenizer is None:
            raise RuntimeError("Judge tokenizer is not initialized.")
        return tokenizer

    def get_model_slug(self) -> str:
        """Get the slug for the model, used for building output directories

        Returns:
            The slug for the model.
        """
        if self.eval_config.model_output_dir:
            return self.eval_config.model_output_dir

        model_slug = self.eval_config.model_path_or_repo_id.split("/")[-1]
        if self.eval_config.lora_path_or_repo_id:
            mlflow_tracking_uri = (
                self.eval_config.mlflow_config.mlflow_tracking_uri
                if self.eval_config.mlflow_config
                else None
            )
            model_slug += f"-lora-{get_lora_slug(self.eval_config.lora_path_or_repo_id, mlflow_tracking_uri=mlflow_tracking_uri)}"
        return model_slug

    def get_dataset_slug(self) -> str:
        """Get the slug for the dataset, used for building output structures (directories and mlflow runs)

        Returns:
            The slug for the dataset.
        """
        return self.dataset_config.file_path.split("/")[-1]

    def get_output_dir(self) -> Path:
        """
        Compute the output directory used for this evaluation run.

        Uses a consistent convention and ensures the directory exists.

        Returns:
            The output directory.
        """
        model_slug = self.get_model_slug()
        dataset_slug = self.get_dataset_slug()
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
            trust_remote_code=self.trust_remote_code,
            max_answer_tokens=self.eval_config.max_answer_tokens,
            reasoning=self.eval_config.reasoning,
            pass_max_answer_tokens=self.eval_config.pass_max_answer_tokens,
            token=self.eval_config.model_token,
        )
        # Deterministic shuffle before sampling
        test_dataset = test_dataset.shuffle(seed=self.dataset_config.seed)
        self.num_samples = (
            min(len(test_dataset), self.eval_config.max_samples)
            if self.eval_config.max_samples
            else len(test_dataset)
        )
        self.eval_dataset = test_dataset.select(range(self.num_samples))
        self.eval_engine.set_dataset(self.eval_dataset)

        self.eval_loader = DataLoader(
            cast("Dataset", self.eval_dataset),
            batch_size=self.eval_engine.get_batch_size(),
            shuffle=False,
            collate_fn=self.data_collator,
        )
        # propagate flag
        self.has_stereotype = getattr(custom_dataset, "has_stereotype", False)

    def generate_answers(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        do_sample: bool | None = None,
    ) -> list[str]:
        return self.eval_engine.generate_answers(
            input_ids,
            attention_mask,
            sampling_config=SamplingConfig(
                do_sample=(
                    do_sample
                    if do_sample is not None
                    else self.eval_config.sampling_config.do_sample
                    if self.eval_config.sampling_config.do_sample is not None
                    else self.eval_config.sample
                ),
                temperature=self.eval_config.sampling_config.temperature,
                top_p=self.eval_config.sampling_config.top_p,
                top_k=self.eval_config.sampling_config.top_k,
                seed=self.dataset_config.seed or self.eval_config.sampling_config.seed,
            ),
        )

    @abstractmethod
    def evaluate(self) -> None:
        """
        Run the evaluation process.

        This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement evaluate().")

    @abstractmethod
    def generate(self) -> Sequence[_GenerationRecord]:
        """
        Generate the answers for the evaluation.

        This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement generate().")

    @abstractmethod
    def grade(
        self,
        generations: Sequence[_GenerationRecord],
        judge_engine: EvalEngine | None = None,
    ) -> None:
        """
        Grade the answers for the evaluation.

        This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement grade().")

    def update_dataset_config(self, dataset_config: DatasetConfig) -> None:
        """
        Update the dataset configuration for the evaluation.

        This is an abstract method that must be implemented by subclasses.
        """
        self.dataset_config = dataset_config
        self._set_seed()
        self.prepare_dataloader()
        self._ensure_run_configuration_allowed()

    @abstractmethod
    def get_grading_context(self) -> AbstractContextManager:
        """
        Context manager for the grading process.

        This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_grading_context().")

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
        model_slug = self.get_model_slug()
        dataset_slug = self.get_dataset_slug()
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
        is_purple_llama_injection = dataset_slug.startswith(
            "prompt-injection-purple-llama"
        )
        # Prompt-injection grading stores model safety as accuracy, so attack success is 1 - accuracy.
        attack_success_rate = 1 - accuracy
        if is_purple_llama_injection:
            metric_column_name = "Attack success rate (%) ⬇️"
            metric_ratio_value = attack_success_rate
        elif is_unbias or is_hallucination:
            metric_column_name = "Accuracy (%) ⬆️"
            metric_ratio_value = accuracy
        else:
            metric_column_name = "Error (%) ⬇️"
            metric_ratio_value = attack_success_rate
        # Convert ratios to percentages
        metric_percentage_value = metric_ratio_value * 100.0
        stereo_percent = (
            stereotyped_bias * 100.0 if stereotyped_bias is not None else None
        )
        results = pd.DataFrame(
            {
                metric_column_name: [metric_percentage_value],
                "Stereotype Bias (%)": [stereo_percent],
                "Empty Responses": [
                    empty_responses,
                ],
            }
        )
        results = self._drop_empty_columns(results)
        logging.info(
            "Results for dataset=%s dataset_type=%s", dataset_slug, dataset_type_value
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
            attack_success_rate * 100.0
            if not (is_unbias or is_hallucination or is_purple_llama_injection)
            else None
        )
        full_attack_success_rate = (
            attack_success_rate * 100.0 if is_purple_llama_injection else None
        )
        summary_row = pd.DataFrame(
            {
                "Model": [model_slug],
                "Dataset": [dataset_slug],
                "Dataset Type": [str(self.dataset_config.dataset_type)],
                "Text Format": ["free_text"],
                "Accuracy (%) ⬆️": [full_acc],
                "Error (%) ⬇️": [full_err],
                "Attack success rate (%) ⬇️": [full_attack_success_rate],
                "Stereotype Bias (%)": [stereo_percent],
                "Empty Responses": [empty_responses],
            }
        )
        self._append_summary_row(full_summary_path, summary_row)

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
            attack_success_rate * 100.0
            if not (is_hallucination or is_unbias or is_purple_llama_injection)
            else None
        )
        brief_attack_success_rate = (
            attack_success_rate * 100.0 if is_purple_llama_injection else None
        )
        brief_df = pd.DataFrame(
            {
                "Dataset": [bias_label],
                "Accuracy (%) ⬆️": [brief_acc],
                "Error (%) ⬇️": [brief_err],
                "Attack success rate (%) ⬇️": [brief_attack_success_rate],
            }
        )
        brief_summary_path = model_results_dir / "summary_brief.csv"
        self._append_summary_row(brief_summary_path, brief_df)

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

    @staticmethod
    def _drop_empty_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
        normalized_dataframe = dataframe.replace("", pd.NA)
        return normalized_dataframe.dropna(axis=1, how="all")

    def _append_summary_row(
        self,
        summary_file_path: Path,
        summary_row: pd.DataFrame,
    ) -> None:
        if summary_file_path.exists():
            existing_summary = pd.read_csv(summary_file_path)
            combined_summary = pd.concat(
                [existing_summary, summary_row], ignore_index=True, sort=False
            )
        else:
            combined_summary = summary_row

        combined_summary = self._drop_empty_columns(combined_summary)
        combined_summary.to_csv(summary_file_path, index=False, float_format="%.3f")

    def cleanup(self, error: bool = False) -> None:
        """
        Cleanup the active MLFlow run if it exists, and set its status accordingly.

        Args:
            error: Flag indicating if the evaluation failed. If True, the run is marked as FAILED, otherwise FINISHED.
        """
        if mlflow and mlflow.active_run():
            active_run = mlflow.active_run()
            run_id = active_run.info.run_id if active_run else None
            if RunStatus is not None:
                status = (
                    RunStatus.to_string(RunStatus.FAILED)
                    if error
                    else RunStatus.to_string(RunStatus.FINISHED)
                )
            else:
                status = "FAILED" if error else "FINISHED"
            mlflow.end_run(status=status)
            logging.info(f"Ended MLflow run {run_id}")

    # Hook: override in subclasses that want the dataset type in the output dir name
    def should_include_dataset_type_in_output_dir(self) -> bool:
        return False

    def _init_mlflow(self) -> None:
        """Initialize MLflow tracking if enabled and available."""
        if not mlflow:
            logging.warning(
                "MLflow is not installed. Install it with: [uv] pip install mlflow"
            )
            self.mlflow_config = None
            return
        if not self.mlflow_config:
            return

        if self.parent_run:
            raise RuntimeError(
                "MLFlow main run already initialized, cannot initialize another one. Did you mean to launch a dataset run?"
            )

        if self.mlflow_config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_config.mlflow_tracking_uri)

        # Set experiment
        if self.mlflow_config.mlflow_experiment_name:
            mlflow.set_experiment(self.mlflow_config.mlflow_experiment_name)

        # Generate run name if not specified
        model_slug = self.get_model_slug()

        active_run = mlflow.active_run()
        if active_run is not None:
            logging.warning(
                f"Using existing active run {active_run.info.run_id} as main run"
            )
            self.parent_run = self.mlflow_run = active_run
        else:
            # Set up parent run with model slug name only
            run_name = self.mlflow_config.mlflow_run_name or f"{model_slug}"
            # Start MLflow parent run
            self.mlflow_run = mlflow.start_run(run_name=run_name)
            self.parent_run = self.mlflow_run
            logging.info(f"Started MLflow main run: {run_name}")

        mlflow.log_params(
            config_to_dict(
                self.eval_config,
                [
                    "model_path_or_repo_id",
                    "max_samples",
                    "batch_size",
                    "sample",
                    "use_4bit",
                    "max_answer_tokens",
                    "judge_path_or_repo_id",
                    "judge_batch_size",
                    "max_judge_tokens",
                    "use_4bit_judge",
                    "reasoning",
                ],
            )
        )

    @contextmanager
    def dataset_mlflow_run(self, run_name: str | None = None) -> Iterator[None]:
        """
        Initialize a new MLFlow run for a specific dataset evaluation, and make sure it is closed after the context is exited.

        Args:
            run_name: The name of the run. If not provided, the dataset file path will be used.
        """
        error = True
        try:
            if mlflow and self.mlflow_config:
                dataset_slug = self.get_dataset_slug()
                if run_name is None:
                    logging.info(f"Using run name {dataset_slug} for dataset run")
                    run_name = dataset_slug

                if not self.parent_run:
                    raise RuntimeError(
                        "Main MLFlow run not found, cannot launch dataset run before initializing MLFlow"
                    )

                self.mlflow_run = mlflow.start_run(run_name=run_name, nested=True)
                logging.info(f"Started MLflow dataset run: {run_name}")
                mlflow.log_params(
                    config_to_dict(
                        self.dataset_config, ["file_path", "dataset_type", "seed"]
                    )
                )

            yield
            error = False
        finally:
            self.cleanup(error)

    def _set_seed(self) -> None:
        if self.dataset_config.seed is not None:
            set_seed(self.dataset_config.seed)
        elif self.eval_config.sampling_config.seed is not None:
            set_seed(self.eval_config.sampling_config.seed)

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
        generations_file = output_dir / "generations.jsonl"

        if responses_file.exists():
            mlflow.log_artifact(str(responses_file))
        if metrics_file.exists():
            mlflow.log_artifact(str(metrics_file))
        if generations_file.exists():
            mlflow.log_artifact(str(generations_file))

    def run_config_path(self) -> Path:
        return self.get_output_dir() / "run_config.json"

    class RunConfig(TypedDict):
        evaluation_config: dict[str, Any]
        dataset_config: dict[str, Any]

    def free_test_model(self) -> None:
        self.eval_engine.free_model()
        del self.eval_engine
        empty_cuda_cache_if_available()
        gc.collect()
        print(
            f"VRAM reserved after freeing test model is {torch.cuda.memory_reserved() / 1e9}GB",
        )

    def _current_run_config(self) -> RunConfig:
        evaluation_config_snapshot = self.eval_config.model_dump(
            exclude={"model_token", "judge_token"},
            exclude_none=True,
        )
        dataset_config_snapshot = self.dataset_config.model_dump(exclude_none=True)

        return {
            "evaluation_config": json.loads(
                json.dumps(evaluation_config_snapshot, default=str)
            ),
            "dataset_config": json.loads(
                json.dumps(dataset_config_snapshot, default=str)
            ),
        }

    def _write_run_config(self, run_config: RunConfig) -> None:
        with open(self.run_config_path(), "w") as file_handle:
            json.dump(run_config, file_handle, indent=2)

    def _clear_output_files(self) -> None:
        for filename in ["responses.json", "metrics.csv", "generations.jsonl"]:
            output_file = self.get_output_dir() / filename
            if output_file.exists():
                output_file.unlink()

    def _ensure_run_configuration_allowed(self) -> None:
        run_config = self._current_run_config()
        config_path = self.run_config_path()

        if not config_path.exists():
            self._write_run_config(run_config)
            return

        with open(config_path) as file_handle:
            existing_run_config = json.load(file_handle)

        if existing_run_config == run_config:
            logging.info(
                "Existing outputs at %s match current configuration; continuing with cached generations if present.",
                config_path,
            )
            return

        if self.eval_config.replace_existing_output:
            logging.info(
                "Replacing outputs at %s because --replace-existing-output was provided.",
                config_path,
            )
            self._clear_output_files()
            self._write_run_config(run_config)
            return

        if sys.stdin.isatty():
            should_replace = typer.confirm(
                (
                    "Existing evaluation outputs were produced with a different configuration. "
                    "Replace them and rerun?"
                ),
                default=False,
            )
            if should_replace:
                logging.info("User approved replacing outputs at %s.", config_path)
                self._clear_output_files()
                self._write_run_config(run_config)
                return

        raise RuntimeError(
            "Existing evaluation outputs were produced with a different configuration. "
            "Re-run with --replace-existing-output to overwrite them."
        )


class FreeTextSharedEvaluator(BaseEvaluator):
    """
    Shared utilities for free‑text evaluators:
    - Manage generations cache (JSON under output dir)
    - Free under‑test model before judging
    - Initialize and free judge pipeline
    """

    def generations_path(self, filename: str = "generations.jsonl") -> Path:
        return Path(self.get_output_dir()) / filename

    def load_generations(
        self, filename: str = "generations.jsonl"
    ) -> list[dict] | None:
        path = self.generations_path(filename)
        if path.exists():
            with open(path) as file_handle:
                generations = [json.loads(line) for line in file_handle if line.strip()]
                return generations or None
        return None

    def reset_generations_file(self, filename: str = "generations.jsonl") -> None:
        path = self.generations_path(filename)
        if path.exists():
            path.unlink()

    def save_generations(
        self, items: list[dict], filename: str = "generations.jsonl"
    ) -> None:
        path = self.generations_path(filename)
        with open(path, "a") as file_handle:
            for item in items:
                file_handle.write(json.dumps(item))
                file_handle.write("\n")

    def load_completed_generation_dicts(
        self, filename: str = "generations.jsonl"
    ) -> list[dict]:
        """
        Load completed generations, logging reuse before appending new batches.
        """

        existing_generations = self.load_generations(filename)
        if not existing_generations:
            self.reset_generations_file(filename)
            return []

        logging.info(
            "Found %s completed generation batches in %s; new batches will be appended.",
            len(existing_generations),
            self.generations_path(filename),
        )
        return existing_generations

    def prepare_judge_tokenizer(self) -> None:
        """
        Load only the judge tokenizer so we can format probe prompts before
        initializing the full judge pipeline. This keeps memory lower while
        constructing representative prompts used for batch-size probing.
        """
        if getattr(self, "judge_tokenizer", None) is None:
            self.judge_tokenizer = load_tokenizer_with_transformers(
                self.eval_config.judge_path_or_repo_id,
                token=self.eval_config.judge_token,
                trust_remote_code=self.eval_config.trust_remote_code,
            )
            # left padding is useful when batch-generating variable-length prompts
            self.judge_tokenizer.padding_side = "left"
            # ensure we have a pad token for the judge model as not all models have it
            if not self.judge_tokenizer.pad_token:
                self.judge_tokenizer.pad_token = self.judge_tokenizer.eos_token

    def free_judge(self, judge_engine: EvalEngine | None = None) -> None:
        if judge_engine is not None:
            judge_engine.free_model()
        if hasattr(self, "judge_tokenizer") and self.judge_tokenizer is not None:
            del self.judge_tokenizer
        empty_cuda_cache_if_available()
        gc.collect()

    @torch.no_grad()
    def run_judge_with_backoff(
        self,
        judge_engine: EvalEngine,
        prompts: list[str],
    ) -> list[list[dict[str, str]]]:
        """
        Execute the judge by probing an executable batch size that can
        complete a full pass over all prompts without OOM. The probing runs the
        entire evaluation pass; upon success, results are returned directly.

        Args:
            judge_engine: The judge eval engine to use.
            prompts: List of prompts to judge.

        Returns:
            List of judge outputs in format [{"generated_text": ...}, ...] per prompt.
        """
        # If a fixed judge batch size is provided, run regularly with that size (no backoff)
        if self.eval_config.judge_batch_size is not None:
            fixed_batch_size = max(1, int(self.eval_config.judge_batch_size))
            outputs_fixed: list[list[dict[str, str]]] = []
            for start in range(0, len(prompts), fixed_batch_size):
                chunk = prompts[start : start + fixed_batch_size]
                result = self._process_judge_prompts_batch(judge_engine, chunk)
                outputs_fixed.extend(result)
            return outputs_fixed

        starting_batch_size = min(len(prompts), MAX_BATCH_SIZE)
        current_bs = starting_batch_size

        outputs: list[list[dict[str, str]]] = []

        def halve_reducer():
            nonlocal current_bs
            current_bs = max(1, current_bs // 2)
            return current_bs

        def try_full_run(candidate_batch_size: int) -> int:
            """Attempt a full evaluation pass using candidate_bs.

            On success, populate `outputs` and return the candidate batch size.
            On failure (e.g., OOM), clear partial outputs, free cache, and re-raise
            to let the wrapper reduce the batch size.
            """
            nonlocal outputs
            outputs = []
            try:
                for start in range(0, len(prompts), candidate_batch_size):
                    chunk = prompts[start : start + candidate_batch_size]
                    res = self._process_judge_prompts_batch(
                        judge_engine, chunk, candidate_batch_size
                    )
                    outputs.extend(res)
                return candidate_batch_size
            except Exception:
                # Drop partial results and free memory before retrying with a smaller batch
                outputs = []
                empty_cuda_cache_if_available()
                raise

        # Probe for an executable batch size; the successful call leaves `outputs` filled
        wrapper = find_executable_batch_size(
            try_full_run,
            starting_batch_size=starting_batch_size,
            reduce_batch_size_fn=halve_reducer,
        )
        wrapper()
        return outputs

    def _process_judge_prompts_batch(
        self,
        judge_engine: EvalEngine,
        prompts: list[str],
        batch_size: int | None = None,
        do_sample: bool | None = None,
    ) -> list[list[dict[str, str]]]:
        """
        Process a batch of prompts through the judge engine.

        Args:
            judge_engine: The judge eval engine to use.
            prompts: List of prompt strings.
            batch_size: Optional batch size override for the engine (used during backoff).
            do_sample: Whether to sample from the model.

        Returns:
            List where each element is [{"generated_text": answer}, ...] for that prompt.
        """
        judge_tokenizer = self._get_judge_tokenizer()

        # Tokenize prompts
        tokenized = judge_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        # transformers' tokenizer classes don't have the correct type hints for the return_tensors argument
        input_ids = cast("Tensor", tokenized["input_ids"])
        attention_mask = cast("Tensor", tokenized["attention_mask"])

        # Generate answers using judge engine for deterministic judging
        resolved_do_sample = (
            self.eval_config.sample_judge if do_sample is None else do_sample
        )
        answers = judge_engine.generate_answers(
            input_ids,
            attention_mask,
            sampling_config=SamplingConfig(
                do_sample=resolved_do_sample,
                temperature=self.eval_config.sampling_config.temperature,
                top_p=self.eval_config.sampling_config.top_p,
                top_k=self.eval_config.sampling_config.top_k,
                seed=self.dataset_config.seed or self.eval_config.sampling_config.seed,
            ),
        )

        # Format output to match pipeline format: [{"generated_text": answer}, ...] per prompt
        result: list[list[dict[str, str]]] = []
        for answer in answers:
            result.append([{"generated_text": answer}])

        return result

    def get_grading_context(self) -> AbstractContextManager[EvalEngine]:
        """
        Context manager for the grading process.
        """
        return self.get_judge_engine_context()

    @contextmanager
    def get_judge_engine_context(self) -> Generator[EvalEngine, None, None]:
        """
        Context manager for the judge engine. Creates an eval engine instance
        (either VllmEvalEngine or TransformersEvalEngine) based on configuration.

        Yields:
            The judge eval engine instance.
        """
        try:
            if not (hasattr(self, "eval_engine") and self.eval_engine.is_judge):
                judge_engine = None

                self.prepare_judge_tokenizer()

                # Create appropriate judge engine based on config
                if self.judge_engine == "vllm":
                    max_model_len = (
                        self.eval_config.vllm_config.judge_max_model_len
                        if self.eval_config.vllm_config
                        else None
                    )
                    judge_engine = VllmEvalEngine(
                        self.eval_config,
                        is_judge=True,
                        max_model_len=max_model_len,
                    )
                else:
                    judge_engine = TransformersEvalEngine(
                        self.data_collator,
                        self.eval_config,
                        is_judge=True,
                    )
                judge_engine.set_dataset(self.eval_dataset)

                self.eval_engine = judge_engine
            yield self.eval_engine
        finally:
            if hasattr(self, "eval_engine"):
                self.free_judge(self.eval_engine)

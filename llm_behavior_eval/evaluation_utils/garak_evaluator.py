from __future__ import annotations

import json
import logging
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from . import garak_util
from .base_evaluator import BaseEvaluator
from .eval_engine import EvalEngine
from .garak_util import GarakConfig
from .util_functions import load_tokenizer_with_transformers

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .dataset_config import DatasetConfig
    from .eval_config import EvaluationConfig

logger = logging.getLogger(__name__)


class _HttpOnlyEngine(EvalEngine):
    """Lightweight stand-in engine used when reusing a remote OpenAI endpoint.

    Garak talks to the remote endpoint directly via the HTTP generator, so no
    local model is loaded. This object only satisfies the attributes the base
    evaluator lifecycle expects (tokenizer, is_judge, ready/free hooks).
    """

    is_judge = False

    def __init__(self, tokenizer: Any) -> None:
        """Store the tokenizer that ``BaseEvaluator`` expects every engine to expose."""
        self.tokenizer = tokenizer

    def ensure_test_model_ready(self) -> None:
        """No-op lifecycle hook; the remote endpoint owns model readiness."""
        return None

    def free_model(self) -> None:
        """No-op lifecycle hook; this engine does not own a local model."""
        return None

    def set_dataset(self, eval_dataset: Any) -> None:
        """No-op lifecycle hook; garak builds prompts instead of using a dataset."""
        return None

    def generate_answers(
        self,
        input_ids: Any,
        attention_mask: Any,
        sampling_config: Any,
    ) -> tuple[list[str], list[str | None]]:
        """Reject the normal evaluator generation path for HTTP-only garak runs."""
        raise NotImplementedError("HTTP-only garak runs generate through garak.")

    def get_batch_size(self) -> int:
        """Return a dummy batch size required by the ``EvalEngine`` interface."""
        return 1


class GarakEvaluator(BaseEvaluator):
    """System-leak behavior backed by garak probes.

    Design A: by default the model is loaded in-process through the normal
    ``BaseEvaluator`` path (``VllmEvalEngine``), and a thin garak generator
    adapter routes probe prompts to that engine. When ``garak_config.base_url``
    is set, an external OpenAI-compatible endpoint is reused instead and no local
    model is loaded.
    """

    def __init__(
        self, eval_config: EvaluationConfig, dataset_config: DatasetConfig
    ) -> None:
        """Resolve garak run metadata before ``BaseEvaluator`` snapshots config."""
        garak_config = eval_config.garak_config or GarakConfig()

        self._system_prompt = garak_util.system_prompt_for(
            eval_config.enable_thinking
        )
        self._num_generations = garak_config.num_generations
        resolved_probes = garak_util.resolve_probes(
            garak_config.probes, garak_config.probe_tags
        )
        garak_config.resolved_probes = resolved_probes
        garak_config.system_prompt_hash = garak_util.system_prompt_hash(
            self._system_prompt
        )
        eval_config.garak_config = garak_config
        self._resolved_probes = resolved_probes

        super().__init__(eval_config, dataset_config)

    def _get_garak_config(self) -> GarakConfig:
        """Return the normalized garak config attached before base initialization."""
        if self.eval_config.garak_config is None:
            raise RuntimeError("GarakEvaluator requires eval_config.garak_config.")
        return self.eval_config.garak_config

    def _chat_template_kwargs(self) -> dict[str, bool]:
        """Build thinking kwargs for vLLM/OpenAI chat-template rendering."""
        arg_name = self.eval_config.enable_thinking_arg_name or "enable_thinking"
        return {arg_name: self.eval_config.enable_thinking}

    def _create_eval_engine(self) -> EvalEngine:
        """Override BaseEvaluator engine creation for garak's two execution modes -
        in-process vLLM, or a no-load HTTP stand-in for base_url runs."""
        garak_config = self._get_garak_config()
        if garak_config.base_url:
            tokenizer = load_tokenizer_with_transformers(
                self.eval_config.model_path_or_repo_id,
                token=self.eval_config.model_token,
                trust_remote_code=self.eval_config.trust_remote_code,
            )
            return _HttpOnlyEngine(tokenizer)

        from .vllm_eval_engine import VllmEvalEngine

        max_model_len = (
            self.eval_config.vllm_config.max_model_len
            if self.eval_config.vllm_config
            else None
        )
        return VllmEvalEngine(self.eval_config, max_model_len=max_model_len)

    def prepare_dataloader(self) -> None:
        """Satisfy ``BaseEvaluator`` setup without loading an HF dataset.

        Garak owns prompt construction through its probes, so this evaluator only
        records the sample count that the base lifecycle and MLflow logging need.
        ``num_samples`` is the total number of attempts (probe prompts) across the
        resolved probes. Falls back to the probe count if probe prompts cannot be enumerated.
        """
        self.has_stereotype = False
        try:
            self.num_samples = garak_util.count_probe_attempts(
                self._resolved_probes,
                system_prompt=self._system_prompt,
                num_generations=self._num_generations,
            )
        except Exception as exc:
            logger.warning(
                "Could not count probe attempts; falling back to probe count (%s).",
                exc,
            )
            self.num_samples = len(self._resolved_probes)

    def should_include_dataset_type_in_output_dir(self) -> bool:
        """Override BaseEvaluator output naming to omit dataset-type suffixes."""
        return False

    def get_dataset_slug(self) -> str:
        """Override BaseEvaluator's dataset slug with garak's reasoning-mode name."""
        return f"garak-{'WithReasoning' if self.eval_config.enable_thinking else 'NoReasoning'}"

    def get_grading_context(self) -> AbstractContextManager:
        """Satisfy BaseEvaluator's grading context hook without a judge model, since garak evaluation is deterministic."""
        return nullcontext()

    def _get_vllm_model(self) -> Any:
        from .vllm_eval_engine import VllmEvalEngine
        if not isinstance(self.eval_engine, VllmEvalEngine):
            raise RuntimeError("In-process garak generation requires VllmEvalEngine.")
        return self.eval_engine.model

    # --- generation ---------------------------------------------------------
    def _build_generator(self) -> Any:
        """Create the garak generator for either HTTP endpoint or in-process vLLM."""
        garak_config = self._get_garak_config()
        if garak_config.base_url:
            api_key = garak_config.api_key or "dummy"
            return garak_util.OpenAICompatGenerator(
                model_name=self.eval_config.model_path_or_repo_id,
                base_url=garak_config.base_url,
                api_key=api_key,
                temperature=garak_config.temperature,
                top_p=garak_config.top_p,
                max_tokens=garak_config.max_tokens,
                stop=garak_config.stop,
                chat_template_kwargs=self._chat_template_kwargs(),
            )

        seed = (
            self.dataset_config.seed
            if self.dataset_config.seed is not None
            else self.eval_config.sampling_config.seed
        )
        return garak_util.InProcessVllmGenerator(
            llm=self._get_vllm_model(),
            model_name=self.eval_config.model_path_or_repo_id,
            temperature=garak_config.temperature,
            top_p=garak_config.top_p,
            max_tokens=garak_config.max_tokens,
            stop=garak_config.stop,
            seed=seed,
            chat_template_kwargs=self._chat_template_kwargs(),
        )

    def generate(self) -> Sequence[dict[str, Any]]:  # pyrefly: ignore[bad-override]
        """Run garak probes and persist raw attempt records to ``generations.jsonl``.

        This is the garak ``system_leak_eval.py`` equivalent. Summary artifacts
        are written in ``grade()`` / ``_grade_impl()``, matching the framework's
        normal generate-then-grade split.
        """
        self.ensure_test_model_ready()
        generator = self._build_generator()
        records = garak_util.run_probes(
            generator,
            system_prompt=self._system_prompt,
            num_generations=self._num_generations,
            probe_names=self._resolved_probes,
            result_path=self.generations_path("generations.jsonl"),
        )
        return records

    def evaluate(self) -> None:
        """Direct API entry point that runs garak generation and summary writing.

        The CLI path calls ``generate()`` and ``grade()`` separately. This method
        exists for callers that invoke ``evaluator.evaluate()`` directly.
        """
        error = True
        try:
            generations = self.generate()
            self.free_test_model()
            with self.dataset_mlflow_run():
                self.grade(cast("Any", generations), None)
            error = False
        finally:
            if self.started_mlflow_run:
                self.cleanup(error)

    # --- grading / output ---------------------------------------------------
    def _grade_impl(
        self,
        generations: Sequence[dict[str, Any]],
        judge_engine: EvalEngine | None = None,
    ) -> None:
        """Summarize garak attempt records and write evaluator output artifacts.

        This is the garak ``system_leak_summary.py`` equivalent and is called by
        the normal CLI grading loop after all probe attempts have been generated.
        """
        del judge_engine  # garak needs no judge model
        attempts, probe_timing = garak_util.load_attempts_from_file(
            self.generations_path("generations.jsonl"),
            system_prompt=self._system_prompt,
        )
        if not attempts:
            attempts, probe_timing = garak_util.normalize_attempts(
                generations, system_prompt=self._system_prompt
            )

        summary = garak_util.summarize(attempts, probe_timing)
        metrics = garak_util.overall_metrics(attempts)
        family_avg = garak_util.family_macro_average(summary["per_family"])
        responses = self._expand_responses(attempts)
        self._save_garak_results(responses, summary, metrics, family_avg)

    @staticmethod
    def _expand_responses(
        attempts: dict[tuple[str, int], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Flatten per-attempt garak outputs into response rows for ``responses.json``."""
        responses: list[dict[str, Any]] = []
        for (probe, seq), record in sorted(attempts.items()):
            for output_index, (output, analysis) in enumerate(
                zip(record["outputs"], record["leak_analysis"], strict=True)
            ):
                responses.append(
                    {
                        "probe": probe,
                        "seq": seq,
                        "output_index": output_index,
                        "output": output,
                        "leak_analysis": analysis,
                        "any_leak": analysis["any_leak"],
                    }
                )
        return responses

    def _save_garak_results(
        self,
        responses: list[dict[str, Any]],
        summary: dict[str, Any],
        metrics: dict[str, float | int],
        family_avg: dict[str, float],
    ) -> None:
        """Write garak metrics, detailed responses, summaries, and MLflow artifacts.

        Output layout (single source of truth for full detail is garak_summary.json):
        - metrics.csv: one-row, output-weighted overall rates + counts for this run.
        - garak_summary.json: per-probe and per-family detail (grows with probes).
        - shared summary_full.csv / summary_brief.csv: a single, fixed-width
          headline (family macro-average system-leak rate) plus counts, so adding
          probe families later never widens the cross-behavior comparison files.
        """
        output_dir = self.get_output_dir()
        thinking_mode = "on" if self.eval_config.enable_thinking else "off"
        model_slug = self.get_model_slug()
        dataset_slug = self.get_dataset_slug()

        rate_columns = {
            "System leak rate (%) ⬇️": metrics["any_leak_rate"] * 100.0,
            "Attempt leak rate (%) ⬇️": metrics["attempt_leak_rate"] * 100.0,
            "Exact secret value leak rate (%) ⬇️": metrics["exact_value_rate"] * 100.0,
            "Instruction leak rate (%) ⬇️": metrics["instruction_rate"] * 100.0,
            "Full prompt leak rate (%) ⬇️": metrics["full_prompt_rate"] * 100.0,
        }
        count_columns = {
            "Probes": metrics["probes"],
            "Attempts": metrics["attempts"],
            "Outputs": metrics["outputs"],
        }
        headline_column = "System leak rate (family avg %) ⬇️"
        family_leak_headline = family_avg["any_leak_rate"] * 100.0

        # metrics.csv
        metrics_df = pd.DataFrame(
            {
                **{key: [value] for key, value in rate_columns.items()},
                "Thinking": [thinking_mode],
                **{key: [value] for key, value in count_columns.items()},
            }
        )
        logger.info("Garak results for dataset=%s", dataset_slug)
        logger.info(metrics_df)
        metrics_df.to_csv(output_dir / "metrics.csv", index=False, float_format="%.3f")

        # responses.json
        with open(output_dir / "responses.json", "w", encoding="utf-8") as handle:
            json.dump(responses, handle, indent=4, ensure_ascii=False)

        # garak_summary.json (per-probe / per-family detail)
        with open(output_dir / "garak_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

        # per-model summaries (fixed-width headline + counts; no per-dimension columns)
        model_results_dir = Path(self.eval_config.results_dir) / model_slug
        summary_full = pd.DataFrame(
            {
                "Model": [model_slug],
                "Dataset": [dataset_slug],
                "Dataset Type": [str(self.dataset_config.dataset_type)],
                "Text Format": ["garak"],
                "Thinking": [thinking_mode],
                headline_column: [family_leak_headline],
                **{key: [value] for key, value in count_columns.items()},
            }
        )
        summary_brief = pd.DataFrame(
            {
                "Dataset": [dataset_slug],
                "Thinking": [thinking_mode],
                headline_column: [family_leak_headline],
            }
        )
        self._append_summary_row(model_results_dir / "summary_full.csv", summary_full)
        self._append_summary_row(model_results_dir / "summary_brief.csv", summary_brief)

        if self.mlflow_config:
            self._log_mlflow_metrics(
                {
                    "system_leak_rate": float(metrics["any_leak_rate"]),
                    "attempt_leak_rate": float(metrics["attempt_leak_rate"]),
                    "exact_value_leak_rate": float(metrics["exact_value_rate"]),
                    "instruction_leak_rate": float(metrics["instruction_rate"]),
                    "full_prompt_leak_rate": float(metrics["full_prompt_rate"]),
                    "system_leak_rate_family_avg": float(family_avg["any_leak_rate"]),
                    "attempt_leak_rate_family_avg": float(
                        family_avg["attempt_leak_rate"]
                    ),
                    "probes": float(metrics["probes"]),
                    "attempts": float(metrics["attempts"]),
                    "outputs": float(metrics["outputs"]),
                }
            )
            self._log_mlflow_artifacts()

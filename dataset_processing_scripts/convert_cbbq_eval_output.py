"""Convert llm_behavior_eval CBBQ artifacts into Hirundo CBBQ evaluator format.

This script reads ``responses.json`` and ``metrics.csv`` files produced by
``llm_behavior_eval/evaluation_utils/cbbq_evaluator.py`` and rewrites them into
folder and CSV layouts used by:
- ``CBBQ/cbbq/eval_bias.py``
- ``CBBQ/cbbq/eval_disamb.py``
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

DATASET_FOLDER_PATTERN = re.compile(
    r"^cbbq-(?P<bias_type>.+)-(?P<split>bias|unbias)-multi-choice(?:_.*)?$"
)

LABEL_TO_LETTER: dict[int, str] = {0: "A", 1: "B", 2: "C"}
POLARITY_TO_TOKEN: dict[int, str] = {0: "neg", 1: "non_neg"}


@dataclass(frozen=True)
class CbbqRunFolder:
    """Metadata for one CBBQ run folder produced by the evaluator.

    Args:
        path: Source folder containing ``responses.json`` and ``metrics.csv``.
        bias_type: CBBQ dimension token, for example ``"age"``.
        split: ``"bias"`` for ambiguous or ``"unbias"`` for disambiguous.

    Returns:
        None
    """

    path: Path
    bias_type: str
    split: str


def _normalize_label(raw_value: object) -> int | None:
    """Normalize parsed CBBQ label values to ``{0,1,2}``.

    Args:
        raw_value: Raw value from ``responses.json``.

    Returns:
        Normalized integer label or ``None`` when invalid.
    """

    if raw_value is None:
        return None
    if isinstance(raw_value, int) and raw_value in LABEL_TO_LETTER:
        return raw_value
    if isinstance(raw_value, float) and raw_value.is_integer():
        int_value = int(raw_value)
        if int_value in LABEL_TO_LETTER:
            return int_value
    if isinstance(raw_value, str) and raw_value.strip() in {"0", "1", "2"}:
        return int(raw_value.strip())
    return None


def _normalize_polarity(raw_value: object) -> str | None:
    """Normalize raw polarity values to CBBQ tokens.

    Args:
        raw_value: Raw ``question_polarity`` value from ``responses.json``.

    Returns:
        ``"neg"`` or ``"non_neg"`` when recognized; otherwise ``None``.
    """

    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"neg", "non_neg"}:
            return normalized
        if normalized in {"0", "1"}:
            return POLARITY_TO_TOKEN[int(normalized)]

    if isinstance(raw_value, int):
        return POLARITY_TO_TOKEN.get(raw_value)

    if isinstance(raw_value, float) and raw_value.is_integer():
        return POLARITY_TO_TOKEN.get(int(raw_value))

    return None


def _parse_run_folder(path: Path) -> CbbqRunFolder | None:
    """Parse a model subfolder into CBBQ run metadata.

    Args:
        path: Candidate run folder under a model results directory.

    Returns:
        Parsed ``CbbqRunFolder`` when the name matches CBBQ format, else ``None``.
    """

    match_result = DATASET_FOLDER_PATTERN.match(path.name)
    if match_result is None:
        return None

    split = match_result.group("split")
    bias_type = match_result.group("bias_type")
    return CbbqRunFolder(path=path, bias_type=bias_type, split=split)


def _discover_cbbq_runs(model_results_dir: Path) -> list[CbbqRunFolder]:
    """Discover all CBBQ evaluator output folders in one model directory.

    Args:
        model_results_dir: Path like ``results/<model_slug>``.

    Returns:
        Sorted list of run folders with required source files present.
    """

    runs: list[CbbqRunFolder] = []
    for child in sorted(model_results_dir.iterdir()):
        if not child.is_dir():
            continue

        parsed = _parse_run_folder(child)
        if parsed is None:
            continue

        responses_path = child / "responses.json"
        metrics_path = child / "metrics.csv"
        if responses_path.exists() and metrics_path.exists():
            runs.append(parsed)

    return runs


def _read_metrics_row(metrics_path: Path) -> dict[str, object]:
    """Read the first metrics row from one evaluator ``metrics.csv``.

    Args:
        metrics_path: Path to source metrics CSV.

    Returns:
        First row as a dictionary.
    """

    metrics_df = pd.read_csv(metrics_path)
    if metrics_df.empty:
        raise ValueError(f"metrics.csv is empty: {metrics_path}")
    return metrics_df.iloc[0].to_dict()


def _metric_as_float(metrics_row: dict[str, object], key: str, default: float) -> float:
    """Extract one float metric key from a row dictionary.

    Args:
        metrics_row: Parsed metrics row.
        key: Column name.
        default: Fallback value.

    Returns:
        Float metric value.
    """

    if key not in metrics_row:
        return default

    raw_value = metrics_row[key]
    if raw_value is None:
        return default

    if isinstance(raw_value, bool):
        return float(raw_value)
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    if isinstance(raw_value, str):
        try:
            return float(raw_value)
        except ValueError:
            return default
    return default


def _load_responses(responses_path: Path) -> list[dict[str, Any]]:
    """Load evaluator ``responses.json`` rows.

    Args:
        responses_path: Path to source responses JSON file.

    Returns:
        List of response dictionaries.
    """

    with responses_path.open(encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if not isinstance(payload, list):
        raise ValueError(f"responses.json must contain a list: {responses_path}")

    rows: list[dict[str, Any]] = []
    for row in payload:
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _convert_bias_outputs(responses: list[dict[str, Any]]) -> pd.DataFrame:
    """Build CBBQ ``eval_bias.py`` compatible outputs table.

    Args:
        responses: Source response rows.

    Returns:
        Dataframe with ``generated``, ``label``, and ``raw_output`` columns.
    """

    rows: list[dict[str, object]] = []
    for response_row in responses:
        predicted_label = _normalize_label(response_row.get("predicted_label"))
        if predicted_label is None:
            continue

        gold_label = _normalize_label(response_row.get("gold_label"))
        generated_text = str(response_row.get("generated_text", ""))

        rows.append(
            {
                "generated": LABEL_TO_LETTER[predicted_label],
                "label": "" if gold_label is None else str(gold_label),
                "raw_output": generated_text,
            }
        )

    outputs_df = pd.DataFrame(rows)
    return outputs_df.reindex(columns=["generated", "label", "raw_output"])


def _convert_disamb_outputs(responses: list[dict[str, Any]]) -> pd.DataFrame:
    """Build CBBQ ``eval_disamb.py`` compatible outputs table.

    Args:
        responses: Source response rows.

    Returns:
        Dataframe with reference disambiguated output columns.
    """

    rows: list[dict[str, object]] = []
    for response_row in responses:
        predicted_label = _normalize_label(response_row.get("predicted_label"))
        gold_label = _normalize_label(response_row.get("gold_label"))
        if predicted_label is None or gold_label is None:
            continue

        polarity = _normalize_polarity(response_row.get("question_polarity"))
        generated_text = str(response_row.get("generated_text", ""))

        is_bias_aligned = (
            (polarity == "neg" and predicted_label == 0)
            or (polarity == "non_neg" and predicted_label == 1)
        )

        rows.append(
            {
                "prediction": LABEL_TO_LETTER[predicted_label],
                "label": LABEL_TO_LETTER[gold_label],
                "polarity": polarity,
                "is_bias_aligned": is_bias_aligned,
                "raw_output": generated_text,
            }
        )

    outputs_df = pd.DataFrame(rows)
    return outputs_df.reindex(
        columns=[
            "prediction",
            "label",
            "polarity",
            "is_bias_aligned",
            "raw_output",
        ]
    )


def _convert_one_run(
    run_folder: CbbqRunFolder,
    output_root: Path,
    model_tag: str,
) -> Path:
    """Convert one run folder to CBBQ reference artifact layout.

    Args:
        run_folder: Parsed source run metadata.
        output_root: Root output directory for converted artifacts.
        model_tag: Model name to write in converted metrics.

    Returns:
        Path of the converted run directory.
    """

    responses = _load_responses(run_folder.path / "responses.json")
    metrics_row = _read_metrics_row(run_folder.path / "metrics.csv")

    examples = int(_metric_as_float(metrics_row, "num_samples", float(len(responses))))
    no_answer_rate = _metric_as_float(metrics_row, "invalid_answer_rate", 0.0)

    if run_folder.split == "bias":
        target_dir = output_root / "ambiguous" / run_folder.bias_type
        outputs_df = _convert_bias_outputs(responses)
        converted_metrics = pd.DataFrame(
            [
                {
                    "bias": run_folder.bias_type,
                    "model": model_tag,
                    "examples": examples,
                    "bias_score": _metric_as_float(metrics_row, "bias_score", 0.0),
                    "reverse_bias_score": _metric_as_float(
                        metrics_row,
                        "reverse_bias_score",
                        0.0,
                    ),
                    "neutrality_rate": _metric_as_float(
                        metrics_row,
                        "neutrality_rate",
                        0.0,
                    ),
                    "neutrality_error_rate": _metric_as_float(
                        metrics_row,
                        "neutrality_error_rate",
                        0.0,
                    ),
                    "no_answer_rate": no_answer_rate,
                }
            ]
        )
    else:
        target_dir = output_root / "disambiguous" / run_folder.bias_type
        outputs_df = _convert_disamb_outputs(responses)
        converted_metrics = pd.DataFrame(
            [
                {
                    "bias": run_folder.bias_type,
                    "model": model_tag,
                    "examples": examples,
                    "accuracy": _metric_as_float(
                        metrics_row,
                        "disambiguated_accuracy",
                        0.0,
                    ),
                    "disamb_bias_score": _metric_as_float(
                        metrics_row,
                        "disambiguated_bias_score",
                        0.0,
                    ),
                    "no_answer_rate": no_answer_rate,
                }
            ]
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    outputs_df.to_csv(target_dir / "outputs.csv", index=False)
    converted_metrics.to_csv(target_dir / "metrics.csv", index=False)
    return target_dir


def convert_model_results(
    model_results_dir: Path,
    output_root: Path,
    model_tag: str | None,
) -> list[Path]:
    """Convert all CBBQ evaluator runs for one model.

    Args:
        model_results_dir: Source folder containing CBBQ dataset run folders.
        output_root: Root destination for CBBQ-compatible output.
        model_tag: Optional model tag for output metrics (defaults to folder name).

    Returns:
        List of converted run directories.
    """

    if not model_results_dir.is_dir():
        raise FileNotFoundError(
            f"Model results directory does not exist: {model_results_dir}"
        )

    resolved_model_tag = model_tag or model_results_dir.name
    run_folders = _discover_cbbq_runs(model_results_dir)
    if not run_folders:
        raise ValueError(
            "No CBBQ run folders with responses.json + metrics.csv were found under "
            f"{model_results_dir}"
        )

    converted_paths: list[Path] = []
    for run_folder in run_folders:
        converted_paths.append(
            _convert_one_run(
                run_folder=run_folder,
                output_root=output_root,
                model_tag=resolved_model_tag,
            )
        )

    return converted_paths


def main(
    model_results_dir: Annotated[
        Path,
        typer.Option(
            "--model-results-dir",
            help=(
                "Path to one model results directory (contains cbbq-* run folders)."
            ),
        ),
    ],
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help=(
                "Directory where CBBQ-compatible ambiguous/disambiguous folders "
                "will be written."
            ),
        ),
    ] = Path("converted_cbbq"),
    model_tag: Annotated[
        str | None,
        typer.Option(
            "--model-tag",
            help="Model label to write into converted metrics.csv rows.",
        ),
    ] = None,
) -> None:
    """Convert llm_behavior_eval CBBQ outputs to CBBQ reference evaluator format.

    Args:
        model_results_dir: One model folder under ``results/``.
        output_root: Target root for converted files.
        model_tag: Optional model name override in metrics rows.

    Returns:
        None
    """

    converted_paths = convert_model_results(model_results_dir, output_root, model_tag)
    print(f"Converted {len(converted_paths)} run folder(s) into: {output_root}")
    for converted_path in converted_paths:
        print(f"- {converted_path}")


if __name__ == "__main__":
    typer.run(main)

"""Upload CBBQ CSV subsets to HuggingFace Dataset repositories.

Converts raw CBBQ CSV files into ``DatasetDict({'train': ...})`` objects and
pushes them to repos following the convention:
`hirundo-io/cbbq-<bias_type>-<kind>-multi-choice`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypeAlias, cast

import pandas as pd
import typer
from datasets import Dataset, DatasetDict
from datasets.utils.logging import disable_progress_bar
from huggingface_hub import HfApi

from llm_behavior_eval.evaluation_utils.enums import CBBQ_BIAS_TYPES

CBBQ_REPO_ORG: str = "hirundo-io"
MULTIPLE_CHOICE_SUFFIX: str = "multi-choice"
CBBQ_SOURCE_ROOT: str = "data"
CBBQ_SOURCE_SPLITS: dict[str, str] = {
    "ambiguous": "bias",
    "disambiguous": "unbias",
}
CBBQ_REQUIRED_COLUMNS: tuple[str, ...] = (
    "context",
    "question",
    "ans0",
    "ans1",
    "ans2",
    "label",
    "question_polarity",
)
CBBQ_REQUIRED_COLUMNS_LIST: list[str] = list(CBBQ_REQUIRED_COLUMNS)


def _build_empty_cbbq_frame() -> pd.DataFrame:
    """Build an empty CBBQ dataframe with all required columns."""
    return pd.DataFrame({column: pd.Series(dtype="object") for column in CBBQ_REQUIRED_COLUMNS_LIST})
CBBQ_RAW_BASE = "https://raw.githubusercontent.com/YFHuangxxxx/CBBQ/master/data"

ValidationErrors: TypeAlias = list[str]


@dataclass(frozen=True)
class CbbqConversionPlan:
    """A single source CSV to target repository conversion plan."""

    bias_type: str
    source_split: str
    target_split: str
    source_csv: Path | str
    target_repo_id: str


@dataclass(frozen=True)
class CbbqValidatedSplit:
    """Validated, normalized rows and metadata for one CBBQ split."""

    plan: CbbqConversionPlan
    df: pd.DataFrame
    row_count: int


def build_repo_id(bias_type: str, target_split: str) -> str:
    """Build one CBBQ HuggingFace repository id.

    Args:
        bias_type: Canonical CBBQ bias dimension.
        target_split: Either ``"bias"`` or ``"unbias"``.

    Returns:
        HuggingFace repository id.
    """

    return f"{CBBQ_REPO_ORG}/cbbq-{bias_type}-{target_split}-{MULTIPLE_CHOICE_SUFFIX}"


def normalize_bias_types(requested_types: list[str] | None) -> tuple[str, ...]:
    """Normalize and validate requested CBBQ bias types.

    Args:
        requested_types: Optional list passed from CLI.

    Returns:
        Canonicalized tuple of bias type identifiers.
    """

    canonical_map = {bias_type.lower(): bias_type for bias_type in sorted(CBBQ_BIAS_TYPES)}
    if not requested_types:
        return tuple(sorted(CBBQ_BIAS_TYPES, key=str.lower))

    resolved: list[str] = []
    unknown: list[str] = []
    for requested in requested_types:
        canonical = canonical_map.get(requested.lower())
        if canonical is None:
            unknown.append(requested)
        elif canonical not in resolved:
            resolved.append(canonical)

    if unknown:
        raise ValueError(
            f"Unknown CBBQ types: {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(sorted(CBBQ_BIAS_TYPES))}."
        )
    return tuple(resolved)


def build_conversion_plan(
    cbbq_root: Path | None,
    types: tuple[str, ...],
) -> list[CbbqConversionPlan]:
    """Build all conversion plans under either a local root or GitHub raw URLs.

    Args:
        cbbq_root: Local CBBQ root path, or ``None`` for remote GitHub mode.
        types: Bias types to include.

    Returns:
        Conversion plans for each requested type and split.
    """

    plans: list[CbbqConversionPlan] = []
    for bias_type in types:
        for source_split, target_split in CBBQ_SOURCE_SPLITS.items():
            if cbbq_root is None:
                source_csv: Path | str = (
                    f"{CBBQ_RAW_BASE}/{bias_type}/{source_split}/{source_split}.csv"
                )
            else:
                source_csv = (
                    cbbq_root
                    / CBBQ_SOURCE_ROOT
                    / bias_type
                    / source_split
                    / f"{source_split}.csv"
                )

            plans.append(
                CbbqConversionPlan(
                    bias_type=bias_type,
                    source_split=source_split,
                    target_split=target_split,
                    source_csv=source_csv,
                    target_repo_id=build_repo_id(
                        bias_type=bias_type,
                        target_split=target_split,
                    ),
                )
            )
    return plans


def validate_and_project_cbbq_dataframe(
    df: pd.DataFrame,
    *,
    bias_type: str,
    source_split: str,
) -> tuple[pd.DataFrame, ValidationErrors]:
    """Validate one CBBQ split and normalize expected evaluator fields.

    Args:
        df: Loaded source CSV.
        bias_type: Dimension for diagnostic output.
        source_split: ``ambiguous`` or ``disambiguous``.

    Returns:
        Tuple containing normalized dataframe and a list of validation errors.
    """

    errors: ValidationErrors = []
    if df.empty:
        return _build_empty_cbbq_frame(), [
            f"[{bias_type}/{source_split}] CSV has no rows"
        ]

    missing_columns = set(CBBQ_REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        return _build_empty_cbbq_frame(), [
            f"[{bias_type}/{source_split}] missing required columns: "
            f"{sorted(missing_columns)}"
        ]

    projected = df.loc[:, CBBQ_REQUIRED_COLUMNS_LIST].copy()
    normalized_labels: pd.Series = cast(
        "pd.Series",
        pd.to_numeric(projected["label"], errors="coerce"),
    )
    normalized_polarity: pd.Series = cast(
        "pd.Series",
        projected["question_polarity"].astype("string").str.strip().str.lower()
    )

    valid_labels = normalized_labels.isin(pd.Index([0, 1, 2]))
    invalid_label_rows = cast("pd.Series", normalized_labels[~valid_labels])
    if not invalid_label_rows.empty:
        errors.append(
            f"[{bias_type}/{source_split}] invalid label values in "
            f"{int(invalid_label_rows.size)} row(s); value={invalid_label_rows.iat[0]}"
        )

    valid_polarity = normalized_polarity.isin(pd.Index(["neg", "non_neg"]))
    invalid_polarity_rows = cast(
        "pd.Series",
        normalized_polarity[~valid_polarity],
    )
    if invalid_polarity_rows.size:
        errors.append(
            f"[{bias_type}/{source_split}] invalid question_polarity values in "
            f"{int(invalid_polarity_rows.size)} row(s); "
            f"value={invalid_polarity_rows.iat[0]}"
        )

    for text_column in ("context", "question", "ans0", "ans1", "ans2"):
        text_series: pd.Series = cast(
        "pd.Series",
            projected[text_column].astype("string"),
        )
        invalid_text: pd.Series = cast(
        "pd.Series",
            text_series.isna() | text_series.str.strip().eq(""),
        )
        if invalid_text.any():
            errors.append(
                f"[{bias_type}/{source_split}] empty required values in "
                f"'{text_column}' for {int(invalid_text.sum())} row(s)"
            )

    if errors:
        return _build_empty_cbbq_frame(), errors

    projected["label"] = normalized_labels.astype("Int64").astype("int64")
    projected["question_polarity"] = normalized_polarity
    return projected.reset_index(drop=True), errors


def validate_all(cbbq_root: Path | None, types: tuple[str, ...]) -> list[CbbqValidatedSplit]:
    """Validate all selected CBBQ splits and return ready-to-upload tables."""

    plans = build_conversion_plan(cbbq_root, types)
    validated: list[CbbqValidatedSplit] = []
    validation_errors: ValidationErrors = []

    for plan in plans:
        source_csv = plan.source_csv

        if isinstance(source_csv, Path) and not source_csv.exists():
            validation_errors.append(
                f"[{plan.bias_type}/{plan.source_split}] Missing CSV: {source_csv}"
            )
            continue

        try:
            source_df = pd.read_csv(source_csv)
        except (FileNotFoundError, pd.errors.ParserError, OSError) as exc:
            validation_errors.append(
                f"[{plan.bias_type}/{plan.source_split}] Unable to read CSV: {source_csv}. "
                f"Error: {exc}"
            )
            continue

        normalized_df, errors = validate_and_project_cbbq_dataframe(
            source_df,
            bias_type=plan.bias_type,
            source_split=plan.source_split,
        )
        if errors:
            validation_errors.extend(errors)
            continue

        validated.append(
            CbbqValidatedSplit(
                plan=plan,
                df=normalized_df,
                row_count=len(normalized_df),
            )
        )

    if validation_errors:
        raise ValueError("Validation failed:\n" + "\n".join(validation_errors))

    return validated


def repo_exists(repo_id: str, token: str | None) -> bool:
    """Check whether a dataset repository already exists on the Hub."""

    return HfApi(token=token).repo_exists(repo_id=repo_id, repo_type="dataset")


def _normalize_types(requested_types: list[str] | None) -> list[str] | None:
    """Normalize CLI type values that can be passed as comma-separated chunks."""

    if not requested_types:
        return None
    output: list[str] = []
    for item in requested_types:
        output.extend([part.strip() for part in item.split(",") if part.strip()])
    return output or None


def run_upload(
    cbbq_dir: Path | None,
    token: str | None,
    dry_run: bool,
    overwrite: bool,
    skip_existing: bool,
    requested_types: list[str] | None,
) -> None:
    """Execute a single upload run."""

    if overwrite and skip_existing:
        raise ValueError("Cannot use both --overwrite and --skip-existing together.")

    selected_types = normalize_bias_types(_normalize_types(requested_types))
    token = token or os.getenv("HF_TOKEN")
    if not dry_run and not skip_existing and token is None:
        raise ValueError(
            "HF token is required for pushing datasets. "
            "Pass --token or set HF_TOKEN, or run with --dry-run first."
        )

    cbbq_root = cbbq_dir
    if cbbq_root is None:
        _run_with_root(None, selected_types, dry_run, overwrite, skip_existing, token)
        return

    if not cbbq_root.is_dir():
        raise FileNotFoundError(f"CBBQ directory does not exist: {cbbq_root}")

    _run_with_root(cbbq_root, selected_types, dry_run, overwrite, skip_existing, token)


def _run_with_root(
    cbbq_root: Path | None,
    selected_types: tuple[str, ...],
    dry_run: bool,
    overwrite: bool,
    skip_existing: bool,
    token: str | None,
) -> None:
    """Run validation and optional push using an existing CBBQ root or remote mode."""

    validated = validate_all(cbbq_root, selected_types)
    print("Planned upload targets:")
    for item in validated:
        print(f"- {item.plan.target_repo_id} ({item.row_count} rows)")

    if dry_run:
        print("Dry-run complete; no datasets were pushed.")
        return

    if not overwrite and any(
        repo_exists(item.plan.target_repo_id, token) for item in validated
    ):
        if not skip_existing:
            existing = [
                item.plan.target_repo_id
                for item in validated
                if repo_exists(item.plan.target_repo_id, token)
            ]
            raise RuntimeError(
                "Refusing to overwrite existing repos. Use --overwrite or --skip-existing. "
                f"Existing: {', '.join(existing)}"
            )

    for item in validated:
        if repo_exists(item.plan.target_repo_id, token) and skip_existing:
            print(f"Skipping existing repo: {item.plan.target_repo_id}")
            continue

        dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(
                    item.df,
                    preserve_index=False,
                )
            }
        )
        dataset.push_to_hub(item.plan.target_repo_id, token=token)
        print(f"Uploaded: {item.plan.target_repo_id}")


def main(
    cbbq_dir: Annotated[
        Path | None,
        typer.Option("--cbbq-dir", help="Path to local CBBQ checkout (skips remote fetch)."),
    ] = None,
    token: Annotated[
        str | None,
        typer.Option(
            "--token",
            help="HF token for uploading. If omitted, uses HF_TOKEN env var.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate data and print planned repos only."),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Allow overwriting repos that already exist."),
    ] = False,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing",
            help="Skip repos that already exist on the Hub instead of overwriting.",
        ),
    ] = False,
    types: Annotated[
        list[str] | None,
        typer.Option(
            "--types",
            help="Optional list of CBBQ bias types to process (default: all 14). "
            "Repeatable and can be comma-separated.",
        ),
    ] = None,
) -> None:
    """Run the CBBQ upload script.

    Args:
        cbbq_dir: Optional path to local CBBQ checkout (skips remote fetch).
        token: HF token for uploading.
        dry_run: Validate data and print planned repos only.
        overwrite: Allow overwriting repos that already exist.
        skip_existing: Skip existing repos on the Hub instead of overwriting.
        types: Optional bias types to process; defaults to all types.
    """

    disable_progress_bar()
    run_upload(cbbq_dir, token, dry_run, overwrite, skip_existing, types)


if __name__ == "__main__":
    typer.run(main)

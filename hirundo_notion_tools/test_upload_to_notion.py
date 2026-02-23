import json
from pathlib import Path

import pytest

from hirundo_notion_tools import upload_to_notion
from hirundo_notion_tools.upload_to_notion import (
    NotionError,
    _build_dataset_number_properties,
    _compute_properties_to_add,
    build_text_like_property_value,
    ensure_database_properties,
    resolve_property_type,
)


def test_resolve_property_type_uses_existing_schema() -> None:
    existing = {
        "Model name": {
            "type": "multi_select",
            "multi_select": {"options": [{"name": "foo"}]},
        },
        "Judge name": {"type": "rich_text", "rich_text": {}},
    }

    assert resolve_property_type(existing, "Model name", "auto") == "multi_select"
    assert resolve_property_type(existing, "Judge name", "auto") == "rich_text"


def test_resolve_property_type_conflict_raises() -> None:
    existing = {"Model name": {"type": "rich_text", "rich_text": {}}}
    with pytest.raises(NotionError) as excinfo:
        resolve_property_type(existing, "Model name", "multi_select")
    assert "exists as type" in str(excinfo.value)


def test_resolve_property_type_missing_defaults_to_multi_select_in_auto_mode() -> None:
    existing: dict[str, object] = {}
    assert resolve_property_type(existing, "Model name", "auto") == "multi_select"
    assert resolve_property_type(existing, "Model name", "rich_text") == "rich_text"


def test_build_text_like_property_value_multi_select() -> None:
    assert build_text_like_property_value("multi_select", "gpt-4") == {
        "multi_select": [{"name": "gpt-4"}]
    }


def test_compute_properties_to_add_creates_expected_shapes() -> None:
    existing = {"Already": {"type": "number", "number": {"format": "number"}}}
    to_add = _compute_properties_to_add(
        existing_properties=existing,
        desired_number_props=["Already", "NewMetric"],
        desired_text_like_props={"Model name": "multi_select", "Judge name": "select"},
    )
    assert to_add["NewMetric"] == {"number": {"format": "number"}}
    assert to_add["Model name"] == {"multi_select": {"options": []}}
    assert to_add["Judge name"] == {"select": {"options": []}}


def test_parse_summary_brief_prefers_error_else_accuracy(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary_brief.csv"
    csv_path.write_text(
        "Dataset,Accuracy (%),Error (%)\n"
        "BBQ: age bias,,58.400\n"
        "BBQ: age unbias,86.400,\n"
        "Empty row,,\n",
        encoding="utf-8",
    )

    got = upload_to_notion.parse_summary_brief(csv_path)
    assert got["BBQ: age bias"] == pytest.approx(58.4)
    assert got["BBQ: age unbias"] == pytest.approx(86.4)
    assert "Empty row" not in got


def test_parse_summary_brief_supports_renamed_metric_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary_brief.csv"
    csv_path.write_text(
        "Dataset,Accuracy (%) ⬆️,Error (%) ⬇️\n"
        "BBQ: race bias,,41.600\n"
        "BBQ: race unbias,72.200,\n",
        encoding="utf-8",
    )

    got = upload_to_notion.parse_summary_brief(csv_path)
    assert got["BBQ: race bias"] == pytest.approx(41.6)
    assert got["BBQ: race unbias"] == pytest.approx(72.2)


def test_parse_summary_brief_allows_single_metric_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary_brief.csv"
    csv_path.write_text(
        "Dataset,Accuracy (%) ⬆️\nCrows Pairs,67.000\n",
        encoding="utf-8",
    )

    got = upload_to_notion.parse_summary_brief(csv_path)
    assert got["Crows Pairs"] == pytest.approx(67.0)


def test_parse_summary_brief_supports_attack_success_rate_columns(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "summary_brief.csv"
    csv_path.write_text(
        "Dataset,Attack success rate (%) ⬇️\nJailbreakBench,25.000\n",
        encoding="utf-8",
    )

    got = upload_to_notion.parse_summary_brief(csv_path)
    assert got["JailbreakBench"] == pytest.approx(25.0)


def test_parse_summary_brief_prefers_error_then_attack_success_rate_then_accuracy(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "summary_brief.csv"
    csv_path.write_text(
        "Dataset,Accuracy (%) ⬆️,Attack success rate (%),Error (%) ⬇️\n"
        "Dataset one,70.000,31.000,12.000\n"
        "Dataset two,66.000,22.000,\n"
        "Dataset three,55.000,,\n",
        encoding="utf-8",
    )

    got = upload_to_notion.parse_summary_brief(csv_path)
    assert got["Dataset one"] == pytest.approx(12.0)
    assert got["Dataset two"] == pytest.approx(22.0)
    assert got["Dataset three"] == pytest.approx(55.0)


def test_extract_model_and_judge_names_reads_first_run_config(tmp_path: Path) -> None:
    page_dir = tmp_path / "page"
    run_dir = page_dir / "some_dataset"
    run_dir.mkdir(parents=True)

    run_cfg = {
        "evaluation_config": {
            "model_path_or_repo_id": "meta-llama/Llama-3.2-3B-Instruct",
            "judge_path_or_repo_id": "google/gemma-3-27b-it",
        }
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_cfg), encoding="utf-8")

    model, judge = upload_to_notion.extract_model_and_judge_names(page_dir)
    assert model == "meta-llama/Llama-3.2-3B-Instruct"
    assert judge == "google/gemma-3-27b-it"


def test_ensure_database_properties_raises_on_existing_non_number_dataset_props() -> (
    None
):
    class DummyClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, object | None]] = []

        def request(
            self, method: str, path: str, payload: dict[str, object] | None = None
        ) -> dict[str, object]:
            self.calls.append((method, path, payload))
            if method == "GET" and path == "/databases/db123":
                return {
                    "properties": {
                        "OK": {"type": "number", "number": {"format": "number"}},
                        "Bad": {"type": "rich_text", "rich_text": {}},
                    }
                }
            raise AssertionError(f"Unexpected call: {method} {path}")

    client = DummyClient()
    with pytest.raises(NotionError) as excinfo:
        ensure_database_properties(
            client=client,  # type: ignore[arg-type] (duck-typed)
            database_id="db123",
            desired_number_props=["OK", "Bad", "Missing"],
            desired_text_like_props={},
            dry_run=False,
        )
    msg = str(excinfo.value)
    assert "Bad" in msg
    assert "rich_text" in msg
    assert "--no-ensure-properties" in msg
    assert all(call[0] != "PATCH" for call in client.calls)


def test_build_dataset_number_properties_raises_on_non_number_schema() -> None:
    existing = {"Bad": {"type": "select", "select": {"options": [{"name": "x"}]}}}
    with pytest.raises(NotionError) as excinfo:
        _build_dataset_number_properties(
            dataset_values={"Bad": 1.0, "OK": 2.0},
            existing_db_props=existing,
            database_id="db123",
        )
    msg = str(excinfo.value)
    assert "Bad" in msg
    assert "select" in msg
    assert "db123" in msg

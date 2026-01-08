"""
Upload evaluation results under `llm_behavior_eval/results/` into a Notion database. (Written by GPT 5.2 with some modifications)

Expected results layout:
- llm_behavior_eval/results/<page_dir>/summary_brief.csv
- llm_behavior_eval/results/<page_dir>/**/run_config.json

For each `<page_dir>`:
- Create/update a Notion page (title = `<page_dir>` by default)
- For each row in `summary_brief.csv`:
  - Notion property name = `Dataset` column
  - Notion property value = `Error (%)` if present else `Accuracy (%)` (as a Number)
- Add two more properties:
  - "Model name": taken from `evaluation_config.model_path_or_repo_id` in any `run_config.json`
  - "Judge name": taken from `evaluation_config.judge_path_or_repo_id` in any `run_config.json`

Env vars:
- NOTION_TOKEN: a Notion integration secret
- NOTION_DATABASE_ID: target database ID

Notes:
- This script can optionally create missing database properties via the Notion API.
- Notion rate limits apply; the client includes basic retry/backoff for 429s.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


class NotionError(RuntimeError):
    pass


def _compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class NotionClient:
    token: str
    notion_version: str = NOTION_VERSION
    api_base: str = NOTION_API_BASE
    timeout_s: int = 60
    min_delay_s: float = 0.35  # ~3 req/sec max; keep some slack.
    max_retries: int = 6

    def request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = self.api_base.rstrip("/") + "/" + path.lstrip("/")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.netloc != "api.notion.com":
            raise NotionError(f"Refusing to call unexpected Notion URL: {url}")
        body: bytes | None = None
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Notion-Version": self.notion_version,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if payload is not None:
            body = _compact_json(payload).encode("utf-8")

        # Basic retry loop for transient errors & rate limits.
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                # Exponential-ish backoff; still keep under a few seconds.
                time.sleep(min(self.min_delay_s * (2**attempt), 8.0))
            else:
                time.sleep(self.min_delay_s)

            try:
                conn = http.client.HTTPSConnection(
                    parsed.netloc, timeout=self.timeout_s
                )
                try:
                    conn.request(
                        method.upper(), parsed.path, body=body, headers=headers
                    )
                    resp = conn.getresponse()
                    status = resp.status
                    raw = resp.read().decode("utf-8")
                    resp_headers = {k: v for (k, v) in resp.getheaders()}
                finally:
                    conn.close()

                if 200 <= status <= 299:
                    if not raw:
                        return {}
                    return json.loads(raw)

                last_err = NotionError(
                    f"Notion HTTP {status} for {method} {path}: {raw}"
                )
                retry_after = resp_headers.get("Retry-After")
                if status == 429:
                    if retry_after is not None:
                        try:
                            time.sleep(float(retry_after))
                        except ValueError:
                            pass
                    continue
                # Some 5xx can be transient.
                if 500 <= status <= 599:
                    continue
                raise last_err
            except (OSError, http.client.HTTPException) as e:
                last_err = e
                continue

        raise NotionError(
            f"Notion request failed after retries: {method} {path}"
        ) from last_err


def find_results_pages(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    return sorted([p for p in results_dir.iterdir() if p.is_dir()])


def parse_summary_brief(summary_brief_csv: Path) -> dict[str, float]:
    """
    Returns {notion_property_name: value_as_number}.

    Chooses `Error (%)` if present, else `Accuracy (%)`.
    """
    if not summary_brief_csv.exists():
        raise FileNotFoundError(f"Missing summary_brief.csv: {summary_brief_csv}")

    out: dict[str, float] = {}
    with summary_brief_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"Dataset", "Accuracy (%)", "Error (%)"}
        if reader.fieldnames is None or not required_cols.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError(
                f"{summary_brief_csv} missing required columns. "
                f"Got: {reader.fieldnames}; required: {sorted(required_cols)}"
            )

        for row in reader:
            dataset = (row.get("Dataset") or "").strip()
            if not dataset:
                continue
            err_raw = (row.get("Error (%)") or "").strip()
            acc_raw = (row.get("Accuracy (%)") or "").strip()
            chosen = err_raw if err_raw != "" else acc_raw
            if chosen == "":
                continue
            try:
                out[dataset] = float(chosen)
            except ValueError as e:
                raise ValueError(
                    f"Non-numeric value for dataset={dataset!r} in {summary_brief_csv}: {chosen!r}"
                ) from e
    return out


def _first_run_config(page_dir: Path) -> Path | None:
    for p in page_dir.rglob("run_config.json"):
        if p.is_file():
            return p
    return None


def extract_model_and_judge_names(page_dir: Path) -> tuple[str | None, str | None]:
    run_config_path = _first_run_config(page_dir)
    if run_config_path is None:
        return None, None

    data = json.loads(run_config_path.read_text(encoding="utf-8"))
    eval_cfg = data.get("evaluation_config") or {}
    model = eval_cfg.get("model_path_or_repo_id")
    judge = eval_cfg.get("judge_path_or_repo_id")
    model_s = str(model) if model is not None else None
    judge_s = str(judge) if judge is not None else None
    return model_s, judge_s


def get_title_property_name(database: dict[str, Any]) -> str:
    props = database.get("properties") or {}
    for name, spec in props.items():
        if (spec or {}).get("type") == "title":
            return name
    raise NotionError("Could not find a title property in the Notion database schema.")


def build_rich_text(value: str) -> dict[str, Any]:
    return {"rich_text": [{"type": "text", "text": {"content": value}}]}


def build_number(value: float) -> dict[str, Any]:
    return {"number": value}


def _build_dataset_number_properties(
    *,
    dataset_values: dict[str, float],
    existing_db_props: dict[str, Any],
    database_id: str | None = None,
) -> dict[str, Any]:
    """
    Build Notion page properties for dataset metric values.

    Dataset metric properties are expected to be Notion "number" properties. If the
    DB schema already defines a dataset property with a non-number type, we refuse
    to write (rather than silently overwriting or coercing).
    """
    offending: list[tuple[str, str]] = []
    out: dict[str, Any] = {}
    for prop_name, value in dataset_values.items():
        existing = existing_db_props.get(prop_name)
        if existing is not None:
            actual = str((existing or {}).get("type") or "")
            if actual != "number":
                offending.append((prop_name, actual))
                continue
        out[prop_name] = build_number(value)

    if offending:
        db_hint = f" (database_id={database_id})" if database_id else ""
        details = ", ".join([f"{n!r}: {t!r}" for (n, t) in sorted(offending)])
        raise NotionError(
            "Notion DB schema mismatch: refusing to write numeric dataset metrics to "
            f"non-number properties{db_hint}. Offending properties: {details}. "
            "Fix the database schema (change these properties to type 'number'), or "
            "rerun with --no-ensure-properties if you want to disable schema management."
        )

    return out


def build_title(value: str) -> dict[str, Any]:
    return {"title": [{"type": "text", "text": {"content": value}}]}


def build_select(value: str) -> dict[str, Any]:
    return {"select": {"name": value}}


def build_multi_select(values: list[str]) -> dict[str, Any]:
    return {"multi_select": [{"name": v} for v in values]}


def _normalize_prop_type(value: str) -> str:
    v = value.strip().lower().replace("-", "_")
    if v == "auto":
        return "auto"
    if v in {"rich_text", "select", "multi_select"}:
        return v
    raise ValueError(
        f"Unsupported property type: {value!r}. Expected one of auto/rich_text/select/multi_select."
    )


def resolve_property_type(
    existing_properties: dict[str, Any],
    prop_name: str,
    requested_type: str,
) -> str:
    """
    Decide which Notion property type to use when writing `prop_name`.

    - If the property exists in the DB schema, we always use its actual type.
      If `requested_type` is explicit (not "auto") and disagrees, we raise.
    - If it does not exist, we use:
      - `requested_type` if explicit
      - "multi_select" if requested_type == "auto"
    """
    requested = _normalize_prop_type(requested_type)
    existing = existing_properties.get(prop_name)
    if existing is not None:
        actual = str((existing or {}).get("type") or "")
        if actual not in {"rich_text", "select", "multi_select"}:
            raise NotionError(
                f"Property {prop_name!r} exists but has unsupported type {actual!r}; "
                "supported: rich_text/select/multi_select."
            )
        if requested != "auto" and requested != actual:
            raise NotionError(
                f"Property {prop_name!r} exists as type={actual!r} but was requested as {requested!r}."
            )
        return actual
    return "multi_select" if requested == "auto" else requested


def build_text_like_property_value(prop_type: str, value: str) -> dict[str, Any]:
    if prop_type == "rich_text":
        return build_rich_text(value)
    if prop_type == "select":
        return build_select(value)
    if prop_type == "multi_select":
        return build_multi_select([value])
    raise ValueError(f"Unsupported property type for text-like value: {prop_type!r}")


def _compute_properties_to_add(
    existing_properties: dict[str, Any],
    desired_number_props: Iterable[str],
    desired_text_like_props: dict[str, str],
) -> dict[str, Any]:
    to_add: dict[str, Any] = {}
    for name in desired_number_props:
        if name in existing_properties:
            continue
        to_add[name] = {"number": {"format": "number"}}
    for name, prop_type in desired_text_like_props.items():
        if name in existing_properties:
            continue
        if prop_type == "rich_text":
            to_add[name] = {"rich_text": {}}
        elif prop_type == "select":
            to_add[name] = {"select": {"options": []}}
        elif prop_type == "multi_select":
            to_add[name] = {"multi_select": {"options": []}}
        else:
            raise ValueError(
                f"Unsupported property type for DB creation: {prop_type!r}"
            )
    return to_add


def ensure_database_properties(
    client: NotionClient,
    database_id: str,
    desired_number_props: Iterable[str],
    desired_text_like_props: dict[str, str],
    dry_run: bool,
) -> None:
    db = client.request("GET", f"/databases/{database_id}")
    existing = db.get("properties") or {}

    offending: list[tuple[str, str]] = []
    for name in desired_number_props:
        spec = existing.get(name)
        if spec is None:
            continue
        actual = str((spec or {}).get("type") or "")
        if actual != "number":
            offending.append((name, actual))
    if offending:
        details = ", ".join([f"{n!r}: {t!r}" for (n, t) in sorted(offending)])
        raise NotionError(
            "Notion DB schema mismatch: expected dataset properties to be type 'number', "
            f"but the following already exist with a different type (database_id={database_id}): "
            f"{details}. Refusing to overwrite. "
            "Fix the database schema (change these properties to type 'number'), or "
            "rerun with --no-ensure-properties if you want to skip property management."
        )

    to_add = _compute_properties_to_add(
        existing_properties=existing,
        desired_number_props=desired_number_props,
        desired_text_like_props=desired_text_like_props,
    )

    if not to_add:
        return
    if dry_run:
        print(
            f"[dry-run] Would add {len(to_add)} database properties: {sorted(to_add.keys())}"
        )
        return
    client.request("PATCH", f"/databases/{database_id}", payload={"properties": to_add})


def find_page_by_title(
    client: NotionClient, database_id: str, title_prop: str, title: str
) -> dict[str, Any] | None:
    payload = {
        "page_size": 1,
        "filter": {
            "property": title_prop,
            "title": {"equals": title},
        },
    }
    resp = client.request("POST", f"/databases/{database_id}/query", payload=payload)
    results = resp.get("results") or []
    if not results:
        return None
    return results[0]


def create_page(
    client: NotionClient,
    database_id: str,
    title_prop: str,
    title: str,
    properties: dict[str, Any],
    dry_run: bool,
) -> str:
    payload = {
        "parent": {"database_id": database_id},
        "properties": {
            title_prop: build_title(title),
            **properties,
        },
    }
    if dry_run:
        print(
            f"[dry-run] Would create page title={title!r} with {len(properties)} properties"
        )
        return "dry-run"
    resp = client.request("POST", "/pages", payload=payload)
    return str(resp["id"])


def update_page(
    client: NotionClient,
    page_id: str,
    properties: dict[str, Any],
    dry_run: bool,
) -> None:
    payload = {"properties": properties}
    if dry_run:
        print(
            f"[dry-run] Would update page_id={page_id} with {len(properties)} properties"
        )
        return
    client.request("PATCH", f"/pages/{page_id}", payload=payload)


def _infer_page_title(page_dir: Path) -> str:
    return page_dir.name


def _collect_all_dataset_property_names(
    results_dir: Path, limit: int | None = None
) -> set[str]:
    names: set[str] = set()
    for i, page_dir in enumerate(find_results_pages(results_dir)):
        if limit is not None and i >= limit:
            break
        summary = page_dir / "summary_brief.csv"
        if not summary.exists():
            continue
        names.update(parse_summary_brief(summary).keys())
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Upload llm-behavior-eval results into Notion."
    )
    parser.add_argument(
        "--results-dir",
        default=str(Path("llm_behavior_eval") / "results"),
        help="Path to the results directory (default: llm_behavior_eval/results).",
    )
    parser.add_argument(
        "--database-id",
        default=os.environ.get("NOTION_DATABASE_ID", ""),
        help="Notion database ID (or set NOTION_DATABASE_ID).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("NOTION_TOKEN", ""),
        help="Notion integration token (or set NOTION_TOKEN).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print actions without calling Notion."
    )
    parser.add_argument(
        "--no-ensure-properties",
        action="store_true",
        help="Do not auto-create missing database properties (will fail if missing).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N page directories (useful for testing).",
    )
    parser.add_argument(
        "--model-prop-type",
        default="auto",
        choices=["auto", "rich_text", "select", "multi_select"],
        help='Property type for "Model name". If "auto", uses DB schema if present; otherwise creates/uses multi_select.',
    )
    parser.add_argument(
        "--judge-prop-type",
        default="auto",
        choices=["auto", "rich_text", "select", "multi_select"],
        help='Property type for "Judge name". If "auto", uses DB schema if present; otherwise creates/uses multi_select.',
    )
    args = parser.parse_args(argv)

    if not args.database_id:
        raise SystemExit("Missing --database-id (or NOTION_DATABASE_ID).")
    if not args.token and not args.dry_run:
        raise SystemExit("Missing --token (or NOTION_TOKEN).")

    results_dir = Path(args.results_dir).resolve()

    # In dry-run mode, we don't need a real token (but keep client object simple).
    client = NotionClient(token=args.token or "dry-run-token")

    title_prop = "Name"
    existing_db_props: dict[str, Any] = {}
    if not args.dry_run:
        db = client.request("GET", f"/databases/{args.database_id}")
        existing_db_props = db.get("properties") or {}
        title_prop = get_title_property_name(db)

    model_prop = "Model name"
    judge_prop = "Judge name"

    model_prop_type = resolve_property_type(
        existing_db_props, model_prop, args.model_prop_type
    )
    judge_prop_type = resolve_property_type(
        existing_db_props, judge_prop, args.judge_prop_type
    )

    if not args.no_ensure_properties:
        dataset_props = _collect_all_dataset_property_names(
            results_dir, limit=args.limit
        )
        ensure_database_properties(
            client=client,
            database_id=args.database_id,
            desired_number_props=sorted(dataset_props),
            desired_text_like_props={
                model_prop: model_prop_type,
                judge_prop: judge_prop_type,
            },
            dry_run=args.dry_run,
        )

    page_dirs = find_results_pages(results_dir)
    if args.limit is not None:
        page_dirs = page_dirs[: args.limit]

    for page_dir in page_dirs:
        summary_path = page_dir / "summary_brief.csv"
        if not summary_path.exists():
            continue

        page_title = _infer_page_title(page_dir)
        dataset_values = parse_summary_brief(summary_path)
        model_name, judge_name = extract_model_and_judge_names(page_dir)

        properties: dict[str, Any] = _build_dataset_number_properties(
            dataset_values=dataset_values,
            existing_db_props=existing_db_props,
            database_id=args.database_id,
        )
        if model_name is not None:
            properties[model_prop] = build_text_like_property_value(
                model_prop_type, model_name
            )
        if judge_name is not None:
            properties[judge_prop] = build_text_like_property_value(
                judge_prop_type, judge_name
            )

        if args.dry_run:
            page_id = "dry-run"
            existing = None
        else:
            existing = find_page_by_title(
                client, args.database_id, title_prop, page_title
            )
            page_id = str(existing["id"]) if existing is not None else ""

        if existing is None:
            page_id = create_page(
                client=client,
                database_id=args.database_id,
                title_prop=title_prop,
                title=page_title,
                properties=properties,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                print(f"Created page {page_title!r} ({page_id})")
        else:
            update_page(
                client=client,
                page_id=page_id,
                properties=properties,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                print(f"Updated page {page_title!r} ({page_id})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Strict, offline LLM-169 evidence comparison; never infers legacy row IDs."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

FAMILIES = (
    "bbq",
    "unqover",
    "bloom",
    "halueval",
    "medhallu",
    "prompt_injection",
    "ccpc",
    "xstest",
    "orbench",
)
REQUIRED_RUN = (
    "model_revision",
    "judge_revision",
    "tokenizer_revision",
    "dataset_revision",
    "backend",
    "packages",
    "config",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_log(path: Path) -> dict:
    """Use Inspect's public reader, including compressed .eval logs."""
    from inspect_ai.log import read_eval_log

    with path.open("rb") as stream:
        return read_eval_log(stream).model_dump(mode="json")


def field(row: dict, path: str):
    value = row
    for key in path.split("."):
        value = value[key]
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def compare_case(root: Path, case: dict) -> dict:
    """Compare explicitly mapped fields and units, failing closed on absent evidence."""
    result = {
        "name": case["name"],
        "family": case["family"],
        "status": "incomplete",
        "errors": [],
        "mismatches": [],
        "metrics": [],
    }
    try:
        if case["family"] not in FAMILIES:
            raise ValueError("unknown family")
        for side in ("legacy", "native"):
            run = case["runs"][side]
            if any(not run.get(key) for key in REQUIRED_RUN):
                raise ValueError(
                    f"{side}: incomplete revision/runtime/config provenance"
                )
        responses = root / case["responses"]
        identity = json.loads((root / case["identity"]).read_text())
        if identity["responses_sha256"] != digest(responses):
            raise ValueError("row identity sidecar does not match responses SHA256")
        old = json.loads(responses.read_text())
        if case.get("generations"):
            generation_path = root / case["generations"]
            if identity.get("generations_sha256") != digest(generation_path):
                raise ValueError("generation identity hash mismatch")
            flattened = []
            for line in generation_path.read_text().splitlines():
                batch = json.loads(line)
                keys = case["generation_fields"]
                columns = [batch[key] for key in keys]
                flattened.extend(
                    dict(zip(keys, values, strict=True))
                    for values in zip(*columns, strict=True)
                )
            if len(flattened) != len(old):
                raise ValueError("generation/response row count mismatch")
            old = [
                {**row, "generation": generation}
                for row, generation in zip(old, flattened, strict=True)
            ]
        ids = identity["ids"]
        if not old or len(old) != len(ids) or len(ids) != len(set(ids)):
            raise ValueError("empty, duplicate or missing legacy identities")
        if any(not isinstance(i, str) or not i for i in ids):
            raise ValueError("IDs must be nonempty strings")
        log = read_log(root / case["eval"])
        if log["status"] != "success":
            raise ValueError("Inspect run did not complete successfully")
        samples = log.get("samples") or []
        if any(s.get("error") for s in samples):
            raise ValueError("Inspect contains sample errors")
        native_ids = [str(s["id"]) for s in samples]
        if len(native_ids) != len(set(native_ids)):
            raise ValueError("duplicate Inspect IDs/epochs")
        if set(ids) != set(native_ids) or len(ids) != case["expected_samples"]:
            result["missing_native"] = sorted(set(ids) - set(native_ids))
            result["missing_legacy"] = sorted(set(native_ids) - set(ids))
            raise ValueError("sample coverage mismatch")
        new = dict(zip(native_ids, samples, strict=True))
        # A caller must map all these dimensions, never just aggregate scores.
        dimensions = case["fields"]
        required = {
            "verdict",
            "answer",
            "judge_question",
            "finish_reason",
            "rendered_prompt",
        }
        if not required <= dimensions.keys():
            raise ValueError("missing generation/judge field mappings")
        confusion = Counter()
        agree = 0
        for ident, legacy in zip(ids, old, strict=True):
            for name, mapping in dimensions.items():
                try:
                    left = field(legacy, mapping["legacy"])
                    right = field(new[ident], mapping["native"])
                    if name == "finish_reason":
                        left = "max_tokens" if left == "length" else left
                    if name == "verdict":
                        confusion[(canonical(left), canonical(right))] += 1
                        agree += left == right
                    if left != right:
                        result["mismatches"].append(
                            {
                                "id": ident,
                                "field": name,
                                "legacy": left,
                                "native": right,
                            }
                        )
                except (KeyError, TypeError, ValueError) as exc:
                    result["errors"].append(
                        f"{ident}: {name}: missing/invalid field ({exc})"
                    )
        result["verdict_agreement_percent"] = 100 * agree / len(ids)
        result["confusion"] = [
            {"legacy": a, "native": b, "count": n}
            for (a, b), n in sorted(confusion.items())
        ]
        with (root / case["metrics_csv"]).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1 or not case["metrics"]:
            raise ValueError(
                "expected one legacy metric row and nonempty metric mappings"
            )
        required_metrics = {
            "halueval": {"accuracy"},
            "medhallu": {"accuracy"},
            "prompt_injection": {"attack_success_rate"},
            "ccpc": {"censorship_rate"},
            "xstest": {"safe_refusal_rate", "unsafe_refusal_rate"},
            "orbench": {"safe_refusal_rate", "unsafe_refusal_rate"},
        }.get(case["family"], {"accuracy", "error"})
        if not required_metrics <= {m["name"] for m in case["metrics"]}:
            raise ValueError("missing primary metric mapping")
        for mapping in case["metrics"]:
            name = mapping["name"]
            unit = mapping["unit"]
            expected_unit = (
                "percent"
                if (case["family"] in ("halueval", "medhallu") and name == "accuracy")
                or (
                    case["family"] == "prompt_injection"
                    and name == "attack_success_rate"
                )
                else "fraction"
            )
            if unit != expected_unit and (
                name in ("accuracy", "error", "attack_success_rate", "censorship_rate")
                or name.endswith("_rate")
                or unit not in ("count", "index")
            ):
                raise ValueError(f"{name}: incorrect native unit")
            left = float(rows[0][mapping["legacy_column"]])
            left = left * mapping["scale"] + mapping.get("offset", 0)
            candidates = [
                s["metrics"][mapping["native_metric"]]["value"]
                for s in (log.get("results") or {}).get("scores", [])
                if s["name"] == mapping["native_scorer"]
                and mapping["native_metric"] in s["metrics"]
            ]
            if len(candidates) != 1:
                raise ValueError(f"{name}: missing/ambiguous native metric")
            right = float(candidates[0])
            if not math.isfinite(left) or not math.isfinite(right):
                result["errors"].append(f"{name}: undefined metric")
                continue
            delta = right - left
            result["metrics"].append(
                {
                    "name": name,
                    "unit": unit,
                    "legacy": left,
                    "native": right,
                    "delta": delta,
                    "delta_pp": delta
                    if unit == "percent"
                    else delta * 100
                    if unit == "fraction"
                    else None,
                }
            )
        result["samples"] = len(ids)
        result["provenance_equal"] = case["runs"]["legacy"] == case["runs"]["native"]
        result["status"] = "incomplete" if result["errors"] else "compared"
        result["acceptance"] = "pending owner decision; no tolerance approved"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["errors"].append(str(exc))
    return result


def compare(manifest: dict, root: Path) -> dict:
    names = [c["name"] for c in manifest["cases"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate case names")
    rows = [compare_case(root, case) for case in manifest["cases"]]
    coverage = json.loads(Path(__file__).with_name("pending.json").read_text())[
        "expected_cases"
    ]
    for name in dict.fromkeys([*manifest["expected_cases"], *coverage]):
        if name not in names:
            rows.append(
                {
                    "name": name,
                    "status": "incomplete",
                    "errors": ["missing evidence pair"],
                }
            )
    return {"schema": "llm169-parity-v1", "acceptance": "not granted", "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(json.loads(args.manifest.read_text()), args.manifest.parent)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 2 if any(row["status"] == "incomplete" for row in report["rows"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())

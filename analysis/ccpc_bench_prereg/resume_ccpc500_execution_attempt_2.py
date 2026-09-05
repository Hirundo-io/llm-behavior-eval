#!/usr/bin/env python3
"""Qualify Azure transport and launch CCPC500 execution attempt 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import ssl
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import build_ccpc500_training_data as generation  # noqa: E402

PARENT = Path("/home/ubuntu/hirundo-research/probe/rsch76-ccpc500-training-data-v1")
EXECUTION = PARENT / "execution_attempt_2"
CONTRACT = Path(
    "/home/ubuntu/hirundo-research/probe/rsch76-ccpc500-training-contract-v1"
)
QA_V1 = Path(
    "/home/ubuntu/hirundo-research/probe/rsch76-ccpc500-supervision-qa-contract-v1"
)
QA_V11 = Path(
    "/home/ubuntu/hirundo-research/probe/rsch76-ccpc500-supervision-qa-contract-v1.1"
)
EXPECTED_MODEL = "gpt-5-2025-08-07"


def now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    """Return a file digest."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write stable JSON."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def endpoint_values() -> tuple[str, str, str, str]:
    """Require all Azure variables without disclosing the secret."""
    names = (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_KEY",
    )
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing Azure environment variables: {', '.join(missing)}")
    return (
        os.environ["AZURE_OPENAI_ENDPOINT"],
        os.environ["AZURE_OPENAI_API_VERSION"],
        os.environ["AZURE_OPENAI_DEPLOYMENT"],
        os.environ["AZURE_OPENAI_API_KEY"],
    )


def transport_diagnostics() -> dict[str, Any]:
    """Probe DNS, TCP, and TLS independently without sending Azure API content."""
    endpoint, api_version, deployment, _ = endpoint_values()
    parsed = urlparse(endpoint)
    host = parsed.hostname
    if not host:
        raise RuntimeError("Azure endpoint has no hostname")
    result: dict[str, Any] = {
        "timestamp": now(),
        "endpoint_hostname": host,
        "api_version": api_version,
        "deployment": deployment,
        "proxy_variables_present": {
            name: bool(os.environ.get(name))
            for name in (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "NO_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
                "no_proxy",
            )
        },
    }
    try:
        addresses = sorted(
            {
                item[4][0]
                for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            }
        )
        result["dns"] = {"result": "PASS", "addresses": addresses}
    except Exception as exc:
        result["dns"] = {
            "result": "FAIL",
            "exception_class": type(exc).__name__,
            "message": str(exc),
        }
        return result
    try:
        with socket.create_connection((host, 443), timeout=20):
            pass
        result["tcp"] = {"result": "PASS"}
    except Exception as exc:
        result["tcp"] = {
            "result": "FAIL",
            "exception_class": type(exc).__name__,
            "message": str(exc),
        }
        return result
    try:
        context = ssl.create_default_context()
        with (
            socket.create_connection((host, 443), timeout=20) as sock,
            context.wrap_socket(sock, server_hostname=host) as tls_sock,
        ):
            certificate = tls_sock.getpeercert()
        result["tls"] = {
            "result": "PASS",
            "protocol": tls_sock.version(),
            "certificate_subject": certificate.get("subject", []),
        }
    except Exception as exc:
        result["tls"] = {
            "result": "FAIL",
            "exception_class": type(exc).__name__,
            "message": str(exc),
        }
    return result


def neutral_request(label: str) -> dict[str, Any]:
    """Make one neutral structured request under the frozen Azure client settings."""
    from openai import AzureOpenAI
    from openai import __version__ as openai_version

    endpoint, api_version, deployment, key = endpoint_values()
    config = generation.read_json(PARENT / "supervision_generator_config.json")
    template = (PARENT / "supervision_generator_template.txt").read_text(
        encoding="utf-8"
    )
    if api_version != config["api_version"] or deployment != config["deployment"]:
        raise RuntimeError("Azure deployment/API version differs from frozen config")
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        api_key=key,
        max_retries=0,
        timeout=config["retry"]["timeout_seconds"],
    )
    started = now()
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": template},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": "Return the single word connected.",
                            "request_form": "other",
                        }
                    ),
                },
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=config["max_completion_tokens"],
            reasoning_effort=config["reasoning_effort"],
        )
        raw = response.choices[0].message.content or ""
        parsed = json.loads(raw)
        parsed_ok = (
            isinstance(parsed, dict)
            and set(parsed) == {"answer"}
            and isinstance(parsed["answer"], str)
            and bool(parsed["answer"].strip())
        )
        return {
            "label": label,
            "started_at": started,
            "completed_at": now(),
            "result": "PASS"
            if parsed_ok and response.model == EXPECTED_MODEL
            else "FAIL",
            "sdk_version": openai_version,
            "response_id": response.id,
            "returned_model": response.model,
            "created": response.created,
            "system_fingerprint": response.system_fingerprint,
            "finish_reason": response.choices[0].finish_reason,
            "parsed": parsed_ok,
            "failure_reason": None
            if parsed_ok and response.model == EXPECTED_MODEL
            else "parse failure or returned model mismatch",
        }
    except Exception as exc:
        cause = exc.__cause__ or exc.__context__
        return {
            "label": label,
            "started_at": started,
            "completed_at": now(),
            "result": "FAIL",
            "sdk_version": openai_version,
            "exception_class": type(exc).__name__,
            "message": str(exc),
            "underlying_exception_class": type(cause).__name__ if cause else None,
            "underlying_message": str(cause) if cause else None,
        }


def qualify(_: argparse.Namespace) -> None:
    """Perform diagnostics plus one required neutral Azure qualification."""
    EXECUTION.mkdir(exist_ok=True)
    diagnostics = transport_diagnostics()
    write_json(EXECUTION / "CONNECTIVITY_DIAGNOSTICS.json", diagnostics)
    result = (
        neutral_request("startup_qualification")
        if diagnostics.get("tls", {}).get("result") == "PASS"
        else {"result": "FAIL", "failure_reason": "DNS/TCP/TLS diagnostic failed"}
    )
    result["expected_model"] = EXPECTED_MODEL
    write_json(EXECUTION / "AZURE_CONNECTIVITY_QUALIFICATION.json", result)
    if result["result"] != "PASS":
        raise RuntimeError("BLOCK — AZURE TEACHER CONNECTIVITY NOT QUALIFIED")


def prepare(_: argparse.Namespace) -> None:
    """Create execution-2 lineage only after the qualification record passes."""
    qualification = generation.read_json(
        EXECUTION / "AZURE_CONNECTIVITY_QUALIFICATION.json"
    )
    if qualification["result"] != "PASS":
        raise RuntimeError("connectivity qualification did not pass")
    for name in (
        "biased_membership.jsonl",
        "supervision_generator_config.json",
        "supervision_generator_template.txt",
    ):
        target = EXECUTION / name
        if not target.exists():
            target.write_bytes((PARENT / name).read_bytes())
    for source_path in (Path(__file__), SCRIPT_DIR / "build_ccpc500_training_data.py"):
        target = EXECUTION / source_path.name
        if not target.exists():
            target.write_bytes(source_path.read_bytes())
    write_json(
        EXECUTION / "execution_manifest.json",
        {
            "execution": "attempt_2",
            "created_at": now(),
            "scientific_contract_unchanged": True,
            "execution_retry_reason": "provider connectivity; not row-level content regeneration",
            "fixed_biased_ids": 4856,
            "links": {
                "training_contract_manifest_sha256": sha256(
                    CONTRACT / "final_training_design_manifest.json"
                ),
                "generator_config_sha256": sha256(
                    PARENT / "supervision_generator_config.json"
                ),
                "qa_v1_sha256s_sha256": sha256(QA_V1 / "SHA256SUMS"),
                "qa_v1_1_sha256s_sha256": sha256(QA_V11 / "SHA256SUMS"),
                "failed_execution_snapshot_sha256s": sha256(
                    PARENT / "qa_execution_v1_1/PRE_QA_RAW_SHA256SUMS"
                ),
                "connectivity_qualification_sha256": sha256(
                    EXECUTION / "AZURE_CONNECTIVITY_QUALIFICATION.json"
                ),
            },
        },
    )


def canary(_: argparse.Namespace) -> None:
    """Run three additional neutral transport canaries before corpus concurrency."""
    results = [neutral_request(f"canary_{number}") for number in range(1, 4)]
    write_json(
        EXECUTION / "AZURE_TRANSPORT_CANARY.json",
        {
            "results": results,
            "result": "PASS"
            if all(row["result"] == "PASS" for row in results)
            else "FAIL",
        },
    )
    if any(row["result"] != "PASS" for row in results):
        raise RuntimeError("BLOCK — AZURE TEACHER CONNECTIVITY NOT QUALIFIED")


def generate(args: argparse.Namespace) -> None:
    """Launch the unchanged fixed-ID generator only after all canaries pass."""
    canary_result = generation.read_json(EXECUTION / "AZURE_TRANSPORT_CANARY.json")
    if canary_result["result"] != "PASS":
        raise RuntimeError("transport canary did not pass")
    generation.OUTPUT = EXECUTION
    generation.generate_biased(SimpleNamespace(workers=args.workers))


def main() -> None:
    """Run one explicit execution-2 lifecycle stage."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(required=True)
    commands.add_parser("qualify").set_defaults(func=qualify)
    commands.add_parser("prepare").set_defaults(func=prepare)
    commands.add_parser("canary").set_defaults(func=canary)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--workers", type=int, default=8)
    generate_parser.set_defaults(func=generate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

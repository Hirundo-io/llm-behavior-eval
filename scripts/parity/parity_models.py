"""Parity-run model settings, read only from a benchmark's ``parity.yaml``.

The benchmark code sets no inference settings, so every qualification runner
takes the model under test's GenerateConfig from ``generate_config`` and the
judge's from ``model_roles.grader.config``. Runners add only evidence-capture
request fields (``CAPTURE_BODY``, an optional per-case ``cache_salt``) and one
connection per role. Provider and server arguments (vLLM limits, dtype, seeds,
content format) are operator inputs: the runner's equivalent of ``-M``.

The qualified runs served the judge with vLLM, while the shipped files name an
``hf/`` grader. Runners therefore keep the grader *config* from ``parity.yaml``
but bind the judge snapshot they are given, after checking it is the
repository ``parity.yaml`` names (so an unqualified judge cannot slip in).
"""

from pathlib import Path
from typing import Any

# Prompt token IDs in each response: boundary evidence, not a legacy setting.
CAPTURE_BODY = {"return_token_ids": True}


def parity_yaml(family: str) -> dict[str, Any]:
    """The raw mapping of ``llm_behavior_eval/<family>/parity.yaml``."""
    import yaml

    from llm_behavior_eval.common.parity import parity_config_path

    return yaml.safe_load(parity_config_path(f"llm_behavior_eval.{family}").read_text())


def _with_capture(config: dict[str, Any], cache_salt: str | None) -> dict[str, Any]:
    body = {**(config.get("extra_body") or {}), **CAPTURE_BODY}
    if cache_salt is not None:
        body["cache_salt"] = cache_salt
    return {**config, "extra_body": body, "max_connections": 1}


def answer_settings(family: str, *, cache_salt: str | None = None) -> dict[str, Any]:
    """Eval-level GenerateConfig kwargs for the model under test."""
    return _with_capture(parity_yaml(family)["generate_config"], cache_salt)


def grader_settings(family: str, *, cache_salt: str | None = None) -> Any:
    """The grader's GenerateConfig."""
    from inspect_ai.model import GenerateConfig

    grader = parity_yaml(family)["model_roles"]["grader"]
    return GenerateConfig(**_with_capture(grader["config"], cache_salt))


def grader_reference(family: str) -> str:
    """The judge repository ``parity.yaml`` names, without its provider prefix."""
    model = parity_yaml(family)["model_roles"]["grader"]["model"]
    return model.split("/", 1)[1]


def check_judge(family: str, judge: str) -> None:
    """Reject a judge snapshot that is not the ``parity.yaml`` repository.

    Accepts the repository id itself, an HF cache snapshot of it
    (``models--org--name/snapshots/<sha>``) or a directory named after it.
    """
    reference = grader_reference(family)
    path = Path(judge)
    cache = "models--" + reference.replace("/", "--")
    if (
        judge != reference
        and cache not in path.parts
        and path.name != (reference.split("/")[-1])
    ):
        raise ValueError(
            f"{family} parity is qualified with {reference}; got judge {judge}"
        )


def expected_request_settings(family: str) -> dict[str, dict[str, Any]]:
    """Per-role request settings a parity run must record, capture fields excluded."""
    config = parity_yaml(family)
    return {
        "answer": dict(config["generate_config"]),
        "judge": dict(config["model_roles"]["grader"]["config"]),
    }

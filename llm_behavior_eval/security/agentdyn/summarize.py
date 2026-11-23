"""Compare the two arms of a paired AgentDyn run, episode by episode.

Reads the logs ``llm-behavior-eval-agentdyn`` wrote, checks that both arms ran
the same task arguments and exactly the same episodes, and writes, for every
suite (and ``all``), mode and outcome, both arms' rates and the hardened-minus-
base difference with a 95% interval from a paired bootstrap that resamples
user tasks: the attacks on one user task move together, so they are not
counted as independent observations.
"""

import argparse
import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from inspect_ai.log import list_eval_logs, read_eval_log

from .run import ARMS, DEFAULT_MODELS

logger = logging.getLogger(__name__)

SCORER = "agentdyn_scorer"
OUTCOMES = {
    "clean": ("utility", "truncated"),
    "attack": ("utility", "attack_success", "secure_completion", "truncated"),
}
"""The 0/1 outcomes compared in each mode (score metadata of ``agentdyn_scorer``)."""
RESAMPLES = 5000
BOOTSTRAP_SEED = 42

Outcomes = dict[tuple[str, str, str], dict[str, dict[str, int]]]
"""(suite, mode, arm) -> sample ID -> outcome -> 0 or 1."""


def load_outcomes(log_dir: Path, models: dict[str, str]) -> Outcomes:
    """Every episode's outcomes, by suite, mode and arm.

    Args:
        log_dir: The run's log directory.
        models: Each arm's Inspect model name (``ARMS`` -> name).

    Raises:
        ValueError: A log is incomplete or not of an arm, an episode failed or
            repeats, or the arms ran different task arguments.
    """
    outcomes: Outcomes = defaultdict(dict)
    arm_of = {model: arm for arm, model in models.items()}
    task_args: dict[str, set[str]] = defaultdict(set)
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info.name)
        if log.status != "success":
            raise ValueError(f"Incomplete log: {info.name}")
        arm = arm_of.get(log.eval.model)
        if arm is None:
            raise ValueError(
                f"{info.name}: model {log.eval.model} is neither arm "
                f"({', '.join(f'{a}={m}' for a, m in models.items())})"
            )
        task_args[arm].add(json.dumps(log.eval.task_args, sort_keys=True))
        for sample in log.samples or []:
            if sample.error is not None:
                raise ValueError(f"Sample error: {sample.id}: {sample.error.message}")
            score = (sample.scores or {}).get(SCORER)
            if score is None or score.metadata is None:
                raise ValueError(f"Missing results: {sample.id}")
            mode = "clean" if score.metadata["injection_task"] is None else "attack"
            key = (str(score.metadata["suite"]), mode, arm)
            sample_id = str(sample.id)
            if sample_id in outcomes[key]:
                raise ValueError(f"Duplicate episode: {key}: {sample_id}")
            values = {name: score.metadata[name] for name in OUTCOMES[mode]}
            if any(value not in (0, 1) for value in values.values()):
                raise ValueError(f"Expected 0/1 outcomes: {sample_id}: {values}")
            outcomes[key][sample_id] = values
    if len({frozenset(arguments) for arguments in task_args.values()}) > 1:
        raise ValueError(f"The arms ran different task arguments: {dict(task_args)}")
    return dict(outcomes)


def cluster_interval(differences: dict[str, int]) -> list[float]:
    """95% interval (percentage points) of the mean difference, by user task.

    Sample IDs are ``<suite>/<user task>/<injection task>``; every episode of
    one user task is resampled together.
    """
    clusters: dict[str, list[int]] = defaultdict(list)
    for sample_id, difference in differences.items():
        clusters[sample_id.rsplit("/", 1)[0]].append(difference)
    counts = np.array([len(values) for values in clusters.values()])
    totals = np.array([sum(values) for values in clusters.values()])
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, len(clusters), size=(RESAMPLES, len(clusters)))
    replicates = totals[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return (100 * np.quantile(replicates, [0.025, 0.975])).tolist()


def summarize(
    log_dir: Path, models: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Paired rates and hardened-minus-base differences.

    Args:
        log_dir: A paired run's log directory (one log per arm, or several
            covering disjoint episodes).
        models: Each arm's Inspect model name; default: the runner's.

    Returns:
        One row per suite (and ``all``), mode and outcome: ``n``,
        ``base_pct``, ``hardened_pct``, ``delta_pp`` and ``delta_ci95_pp``.

    Raises:
        ValueError: No logs, or the arms did not run the same episodes.
    """
    models = models or {arm: DEFAULT_MODELS[arm][0] for arm in ARMS}
    outcomes = load_outcomes(log_dir, models)
    if not outcomes:
        raise ValueError(f"No evaluation logs in {log_dir}")
    for mode in {mode for _, mode, _ in outcomes}:
        for arm in ARMS:
            outcomes[("all", mode, arm)] = {
                sample_id: values
                for (suite, sample_mode, sample_arm), samples in list(outcomes.items())
                if suite != "all" and sample_mode == mode and sample_arm == arm
                for sample_id, values in samples.items()
            }
    rows: list[dict[str, Any]] = []
    for suite, mode in sorted({(suite, mode) for suite, mode, _ in outcomes}):
        base = outcomes.get((suite, mode, "base"), {})
        hardened = outcomes.get((suite, mode, "hardened"), {})
        if base.keys() != hardened.keys():
            raise ValueError(f"Unpaired episode IDs in {suite}/{mode}")
        for outcome in OUTCOMES[mode]:
            differences = {
                key: hardened[key][outcome] - base[key][outcome] for key in base
            }
            n = len(base)
            rows.append(
                {
                    "suite": suite,
                    "mode": mode,
                    "metric": outcome,
                    "n": n,
                    "base_pct": 100 * sum(v[outcome] for v in base.values()) / n,
                    "hardened_pct": 100
                    * sum(v[outcome] for v in hardened.values())
                    / n,
                    "delta_pp": 100 * sum(differences.values()) / n,
                    "delta_ci95_pp": cluster_interval(differences),
                }
            )
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    """Write the paired statistics as JSON."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog="llm-behavior-eval-agentdyn-summary", description=__doc__
    )
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    for arm in ARMS:
        model = DEFAULT_MODELS[arm][0]
        parser.add_argument(f"--{arm}-model", default=model, help=f"default: {model}")
    arguments = parser.parse_args(argv)
    models = {arm: getattr(arguments, f"{arm}_model") for arm in ARMS}
    try:
        rows = summarize(arguments.log_dir, models)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    arguments.output.write_text(json.dumps(rows, indent=2) + "\n")
    logger.info("Wrote %s", arguments.output)


if __name__ == "__main__":
    main()

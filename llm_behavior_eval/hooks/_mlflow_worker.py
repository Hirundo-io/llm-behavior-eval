"""Isolated synchronous reporting worker; invoked only by the opt-in hook."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def report(payload: dict[str, Any]) -> None:
    """Report finalized logs; only terminate runs this operation creates."""
    from mlflow import MlflowClient

    config = payload["config"]
    client = MlflowClient(tracking_uri=config["tracking_uri"])
    run_id = config["run_id"]
    owned = not run_id
    if owned:
        experiment = client.get_experiment_by_name(config["experiment"])
        experiment_id = (
            experiment.experiment_id
            if experiment
            else client.create_experiment(config["experiment"])
        )
        run_id = client.create_run(
            experiment_id, tags={"mlflow.runName": config["run_name"]}
        ).info.run_id
    failed = payload["failed"] or any(task["failed"] for task in payload["tasks"])
    try:
        folder = config["subfolder"]
        if folder == "timestamp":
            folder = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if folder and (
            folder.startswith("/") or any(p in {".", ".."} for p in folder.split("/"))
        ):
            raise ValueError("Artifact subfolder must be relative")
        root = "/".join(
            filter(None, ["llm-behavior-eval", folder, payload["inspect_run_id"]])
        )
        previous = client.get_run(run_id).data.metrics.get("datasets_attached", 0)
        for task in payload["tasks"]:
            for key, value in task["metrics"].items():
                client.log_metric(
                    run_id, key, value, step=config["step"], synchronous=True
                )
            path = Path(task["location"])
            if path.is_file():
                client.log_artifact(run_id, str(path), f"{root}/{task['artifact_id']}")
        client.log_metric(
            run_id,
            "datasets_attached",
            previous + len(payload["tasks"]),
            step=config["step"],
            synchronous=True,
        )
    except Exception:
        failed = True
        raise
    finally:
        if owned:
            client.set_terminated(run_id, status="FAILED" if failed else "FINISHED")


if __name__ == "__main__":
    report(json.load(sys.stdin))

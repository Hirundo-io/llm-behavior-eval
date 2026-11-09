"""Locate and read a benchmark's explicit parity config.

A parity config is an ordinary Inspect run-config file (``inspect eval
--run-config``) named ``parity.yaml`` in the benchmark package. It records the
legacy engine's exact inference settings so a parity run is opt-in and
reproducible:

* ``generate_config``: GenerateConfig for the model under test.
* ``model_roles.grader``: the judge model, its ``config`` (GenerateConfig) and
  its ``args`` (provider model args).
* ``eval_config``: optional eval options such as ``limit``.

It must not choose the task, the model under test or the solver, and it must
not set ``tags``, ``metadata`` or ``sandbox``, which are not inference
settings. Nothing in this package reads it unless the user passes it.
"""

from importlib.resources import files
from pathlib import Path
from typing import Any

PARITY_CONFIG_FILE = "parity.yaml"
_FORBIDDEN = {"task", "model", "solver", "tags", "metadata", "sandbox"}


def parity_config_path(package: str) -> Path:
    """Path of ``parity.yaml`` in an installed benchmark package.

    Example: ``parity_config_path("llm_behavior_eval.inspect_template")``.

    Args:
        package: Fully qualified benchmark package name.

    Returns:
        The path of the package's ``parity.yaml``.

    Raises:
        FileNotFoundError: The package ships no ``parity.yaml``.
    """
    path = Path(str(files(package) / PARITY_CONFIG_FILE))
    if not path.is_file():
        raise FileNotFoundError(f"{package} has no {PARITY_CONFIG_FILE}")
    return path


def read_parity_config(path: str | Path) -> dict[str, Any]:
    """Keyword arguments for ``inspect_ai.eval()`` from a parity run config.

    Parsing is delegated to Inspect's own ``--run-config`` reader, so Python and
    CLI runs resolve the file identically. Like the CLI, this constructs the
    grader model, which may load its weights.

    Args:
        path: A YAML run-config file.

    Returns:
        ``inspect_ai.eval()`` keyword arguments, such as ``model_roles``,
        generation settings and ``limit``.

    Raises:
        ValueError: The file is not a YAML mapping, or it sets ``task``,
            ``model``, ``solver``, ``tags``, ``metadata`` or ``sandbox``.
        PrerequisiteError: Inspect rejects the run config.
    """
    import yaml

    # Inspect has no public run-config reader. This private one is what
    # `inspect eval --run-config` calls, so Python and CLI runs agree;
    # test_parity_config_python and test_parity_config_cli fail if an Inspect
    # upgrade moves or changes it.
    from inspect_ai._cli.eval import parse_run_config

    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    forbidden = sorted(_FORBIDDEN & raw.keys())
    if forbidden:
        raise ValueError(
            f"{path} sets {forbidden}; parity configs hold only inference settings"
        )
    return parse_run_config(str(path))

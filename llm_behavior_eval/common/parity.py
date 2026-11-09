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

Plain ``yaml.safe_load`` and ``json.loads`` keep the last of repeated keys, so a
duplicated setting would silently replace the first. ``load_run_config``
rejects repeated keys at every nesting level instead.
"""

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

PARITY_CONFIG_FILE = "parity.yaml"
_FORBIDDEN = {"task", "model", "solver", "tags", "metadata", "sandbox"}
_YAML_MERGE_TAG = "tag:yaml.org,2002:merge"


class DuplicateKeyError(ValueError):
    """A run-config file repeats a key within one mapping."""


class _StrictSafeLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that rejects repeated keys in any mapping."""

    def construct_mapping(
        self, node: yaml.MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        first_lines: dict[Any, int] = {}
        for key_node, _ in node.value:
            # Merge keys (``<<: *anchor``) may legitimately repeat and be
            # overridden by explicit keys; YAML defines that precedence.
            if key_node.tag == _YAML_MERGE_TAG:
                continue
            key = self.construct_object(key_node, deep=True)
            try:
                first_line = first_lines.get(key)
            except TypeError:
                # Unhashable key: let SafeLoader raise its own error.
                break
            line = key_node.start_mark.line + 1
            if first_line is not None:
                raise DuplicateKeyError(
                    f"{key_node.start_mark.name}: duplicate key {key!r} at line "
                    f"{line} (first set at line {first_line})"
                )
            first_lines[key] = line
        return super().construct_mapping(node, deep=deep)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate key {key!r}")
        result[key] = value
    return result


def load_run_config(path: str | Path) -> Any:
    """Parse a run-config file, rejecting repeated keys at any nesting level.

    Like Inspect's reader, a file whose text starts with ``{`` is JSON and
    anything else is YAML.

    Args:
        path: A YAML or JSON run-config file.

    Returns:
        The parsed document.

    Raises:
        DuplicateKeyError: A mapping repeats a key. YAML errors name the key
            and both lines; JSON errors name the key.
    """
    text = Path(path).read_text()
    if text.strip().startswith("{"):
        try:
            return json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
        except DuplicateKeyError as error:
            raise DuplicateKeyError(f"{path}: {error}") from None
    loader = _StrictSafeLoader(text)
    loader.name = str(path)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


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

    The file is first checked with ``load_run_config``, which rejects repeated
    keys. Parsing is then delegated to Inspect's own ``--run-config`` reader, so
    Python and CLI runs resolve the file identically. Like the CLI, this
    constructs the grader model, which may load its weights.

    Args:
        path: A YAML or JSON run-config file.

    Returns:
        ``inspect_ai.eval()`` keyword arguments, such as ``model_roles``,
        generation settings and ``limit``.

    Raises:
        DuplicateKeyError: A mapping in the file repeats a key.
        ValueError: The file is not a mapping, or it sets ``task``, ``model``,
            ``solver``, ``tags``, ``metadata`` or ``sandbox``.
        PrerequisiteError: Inspect rejects the run config.
    """
    # Inspect has no public run-config reader. This private one is what
    # `inspect eval --run-config` calls, so Python and CLI runs agree;
    # test_parity_config_python and test_parity_config_cli fail if an Inspect
    # upgrade moves or changes it.
    from inspect_ai._cli.eval import parse_run_config

    raw = load_run_config(path)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    forbidden = sorted(_FORBIDDEN & raw.keys())
    if forbidden:
        raise ValueError(
            f"{path} sets {forbidden}; parity configs hold only inference settings"
        )
    return parse_run_config(str(path))

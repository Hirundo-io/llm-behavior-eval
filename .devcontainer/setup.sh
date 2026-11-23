#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR%/.devcontainer}"
VENV_PATH="${PROJECT_ROOT}/.venv"
# `uv pip install -e .` below reads [tool.uv.config-settings-package] from the root
# pyproject.toml in the current directory, which installs the root editable in
# setuptools' compat mode (see "Packaging" in INSPECT_MIGRATION.md). Outside the
# repository root it would need `-C editable_mode=compat`.
cd "${PROJECT_ROOT}"

export PATH="${HOME}/.local/bin:${PATH}"

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  if command -v curl >/dev/null 2>&1; then
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
      export PATH="${HOME}/.local/bin:${PATH}"
      return
    fi
  fi

  # Fallback to pip-based installation only to obtain uv when the install script is unavailable.
  python -m pip install --user uv
  export PATH="${HOME}/.local/bin:${PATH}"
}

ensure_uv

if [[ ! -d "${VENV_PATH}" ]]; then
  uv venv "${VENV_PATH}"
fi

VENV_PYTHON="${VENV_PATH}/bin/python"

# The native distributions (the workspace members) editable too, as `uv sync`
# installs them. They go in the same `uv pip install` as the root: installed on
# its own, the root would resolve its llm-behavior-eval-common dependency from
# PyPI instead of the workspace.
MEMBER_ARGS=()
while IFS= read -r member; do
  MEMBER_ARGS+=(-e "${member}")
done < <("${VENV_PYTHON}" "${PROJECT_ROOT}/scripts/workspace.py")

install_workspace() {
  uv pip install --python "${VENV_PYTHON}" -e "$1" "${MEMBER_ARGS[@]}"
}

EXTRAS_REQUESTED="${LLM_BEHAVIOR_EVAL_INSTALL_EXTRAS:-}" # e.g. "mlflow" or "mlflow,vllm"

if [[ -n "${EXTRAS_REQUESTED}" ]]; then
  if ! install_workspace ".[${EXTRAS_REQUESTED}]"; then
    echo "Warning: Failed to install requested extras (${EXTRAS_REQUESTED}); falling back to base install." >&2
    install_workspace .
  fi
else
  install_workspace .
fi

uv pip install --python "${VENV_PYTHON}" ruff pyrefly pytest pre-commit bumpver


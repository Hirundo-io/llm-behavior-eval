#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR%/.devcontainer}"
VENV_PATH="${PROJECT_ROOT}/.venv"

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

if ! uv pip install --python "${VENV_PYTHON}" -e ".[mlflow,vllm]"; then
  echo "Warning: Failed to install mlflow and vllm extras; falling back to base install." >&2
  uv pip install --python "${VENV_PYTHON}" -e .
fi
uv pip install --python "${VENV_PYTHON}" ruff basedpyright pytest pre-commit bumpver


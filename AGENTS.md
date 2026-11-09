# Repository Guidelines

## Project Structure & Module Organization
Native Inspect tasks live in benchmark subpackages under `llm_behavior_eval/`;
shared code lives in `common/`, `bias/`, `hallucination/`, and `refusal/`.
`_registry.py` is the Inspect entry point; `target_metrics.json` declares brief
metrics. Tests mirror these packages in `tests/test_inspect_*.py`. Legacy
`evaluation_utils/` and engine tests remain temporarily for parity comparison.
Do not edit generated `dist/` artifacts.

## Build, Test, and Development Commands
- Use Python 3.12 or 3.13 for brief; native Inspect alone supports 3.11.
- `uv venv --python 3.12 .venv && source .venv/bin/activate`.
- `uv sync --locked`: complete pinned development/comparison environment.
- `uv pip install --python .venv/bin/python -e '.[brief]'`: editable install.
- Devcontainer defaults to Python 3.12 and brief; set
  `LLM_BEHAVIOR_EVAL_INSTALL_EXTRAS=brief,mlflow` for additional integrations.
- `pytest --import-mode=importlib`: full tests;
  `pytest --import-mode=importlib tests/test_inspect_smoke.py tests/test_inspect_brief.py`
  runs registered tasks on local datasets with mock answer/grader and real logs.
- Run `ruff format --check --diff .`, `ruff check .`, and `pyrefly check`.
- Prefer uv; install into `.venv`, never the system interpreter.

## Coding Style & Naming Conventions
Follow PEP 8 defaults with 4-space indentation. Prefer `snake_case` for modules, functions, and variables; reserve `PascalCase` for classes and `UPPER_SNAKE_CASE` for constants. Keep public CLI options descriptive and aligned with native Inspect task arguments. Let `ruff` fix spacing and import order; avoid disabling rules unless there is a clear justification. Type hints are expected on new public functions—match the patterns in `evaluation_utils/`.

## Testing Guidelines
Add or update `tests/test_*.py` files alongside any new feature. Use `pytest` assertions, fixtures, and `monkeypatch` for mocking (avoid `unittest.mock`). Parametrization keeps dataset scenarios readable. Every benchmark needs registry discovery, dataset conversion and scorer tests, and a target_metrics entry checked against actual .eval metric names. Exercise native Inspect eval with mock answer/grader models and local fixtures. Preserve task-specific thinking policies and explicitly patch fixture fingerprints; never bypass production CCPC validation. Keep simulated model outputs deterministic so runs remain reproducible. When adding evaluation datasets, include at least one regression test that exercises parsing and scoring logic. Use pytest's `monkeypatch` fixture for patching module attributes and dependencies; create custom fixtures for reusable test setup.

## Commit & Pull Request Guidelines
Write imperative, concise commit titles (e.g., `Add hallu evaluator smoke tests`). Squash trivial fixups locally before raising a PR. Each PR should explain behavior changes, note impacts on benchmark outputs, and link to any tracking issue. Attach screenshots or sample CLI output when the change affects user-visible results. Mark configuration-sensitive updates (MLflow, datasets, prompt presets) so reviewers can double-check downstream pipelines.

## Security & Configuration Tips
No secrets should live in the repo. Use environment variables for provider tokens and model paths when invoking Inspect. Verify that large artifacts stay out of version control—`results/` is excluded, so place experimental outputs there and keep them untracked.

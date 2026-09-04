# Host runtime disposition — FlashInfer / `ninja` PATH (T5E)

## Decision

**Host-specific / deployment concern. Do not encode in this repository.**

## Context

During live deployment, FlashInfer's JIT step failed because an
environment-local `ninja` binary was present on the GPU host but absent from
the non-interactive SSH `PATH`. That is a classic non-login shell issue: the
study env's `bin` directory (or another install location for `ninja`) was not
exported for the launch session.

## Why this does not belong in llm-behavior-eval

1. **No supported general bootstrap for GPU-host tooling.** The only PATH
   normalization in this repository is `.devcontainer/setup.sh`, which prepends
   `$HOME/.local/bin` so `uv` is available inside the VS Code / Dev Container
   workflow. That path is intentionally narrow and is not a runtime contract
   for SSH GPU launches or FlashInfer build tools.

2. **Absolute host paths are prohibited.** Hard-coding something like
   `/home/<user>/.../env-frozen/bin/ninja` into production or study wrappers
   would be brittle, machine-specific, and would falsely appear to be part of
   the frozen scientific contract.

3. **Activation is the caller's responsibility.** The study wrappers already
   assume `llm-behavior-eval` (and `$PYTHON`) come from the activated study
   environment. Co-located tools such as `ninja` must be made visible the same
   way: activate the env (or otherwise put its `bin` on `PATH`) before
   `run_smoke.sh` / `run_full.sh` under non-interactive SSH.

## Operator guidance (outside this repo)

Before launching arms over non-interactive SSH:

- Activate the study virtualenv / pixi env in the same shell that runs the
  wrappers; or
- Explicitly prepend that env's `bin` directory to `PATH`; and
- Confirm `command -v ninja` resolves to the intended binary.

Do **not** patch study scripts or library code with host-absolute `ninja`
paths. Scientific generation/evaluation settings are unrelated and unchanged.

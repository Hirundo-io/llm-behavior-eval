#!/usr/bin/env bash
# Shared config for the T4U prompt-injection Purple Llama OFF/ON study.
# Sourced by run_smoke.sh and run_full.sh. Do not edit values here after a
# smoke has run without re-freezing (see STUDY_CONTRACT.md "Do not change
# after outcomes").
#
# FAIL-CLOSED: sourcing this file runs the full static preflight gate
# (preflight.py static) and hard-fails the whole script if it doesn't pass.
# No target inference can start without this succeeding first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFLIGHT="$SCRIPT_DIR/preflight.py"
PYTHON="${PYTHON:-python3}"

BEHAVIOR="prompt-injection"

UNCENSORED_LORA_RANK="${UNCENSORED_LORA_RANK:-16}"
# UNCENSORED_LORA_SHA256 is deliberately UNSET here -- see
# UNCENSORED_MODEL_ADAPTER_IDENTITY.md. Two conflicting SHA-256 values for
# the "654d5acdd2eb_0" adapter were found (9b3158331c... in Notion vs
# ab5c9beb... in the untracked scripts/validate_ccpc216_thinking_on.py), and
# neither has been verified against a real adapter_model.safetensors file
# (none exists on this machine). This is a BLOCKING gap: do not set this
# variable to either candidate without first re-hashing the actual artifact
# and updating UNCENSORED_MODEL_ADAPTER_IDENTITY.md accordingly.
UNCENSORED_LORA_SHA256="${UNCENSORED_LORA_SHA256:-}"
# Local path to the Uncensored Model (v2c) LoRA adapter snapshot. This is a
# host-specific path (Appendix A: models/rsch-76-v2c/adapter_snapshots/654d5acdd2eb_0/)
# -- resolve it on whatever GPU box actually runs this, or pre-fetch/mount it
# there first. Do NOT substitute a different adapter without re-verifying the
# checksum via `preflight.py verify-adapter` below.
UNCENSORED_LORA_PATH="${UNCENSORED_LORA_PATH:-models/rsch-76-v2c/adapter_snapshots/654d5acdd2eb_0}"

JUDGE_ENGINE="${JUDGE_ENGINE:-vllm}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-16384}"
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-8192}"
MAX_JUDGE_TOKENS="${MAX_JUDGE_TOKENS:-32}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-32}"

RESULTS_ROOT="${RESULTS_ROOT:-$HOME/.local/share/llm-behavior-eval/results}"
DATASET_SLUG="prompt-injection-purple-llama"

echo "=== Running mandatory static preflight (preflight.py static) ==="
PREFLIGHT_RECORD="$SCRIPT_DIR/preflight_record.json"
"$PYTHON" "$PREFLIGHT" static --output "$PREFLIGHT_RECORD"
echo "Static preflight passed. Record: $PREFLIGHT_RECORD"

# Resolve the base model and judge model to their pinned, immutable local
# snapshot paths -- the runtime must receive these paths, never the mutable
# repo id. Hard-fails (via preflight.py) if the resolved commit differs.
BASE_MODEL="$("$PYTHON" "$PREFLIGHT" resolve-model Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)"
echo "Base model resolved to local snapshot: $BASE_MODEL"

# NOTE: this resolves tokenizer/config only (see preflight.py's
# resolve_snapshot call in run_static_checks) for the *static* judge check;
# a real launch additionally needs the full weight snapshot, which
# `llm-behavior-eval`'s own model loading will fetch under this same pinned
# revision since JUDGE_MODEL below is the *repo id*, not yet a local weights
# path -- llm-behavior-eval has no --judge-revision flag (see
# PARAMETER_PLUMBING.md gap #8), so this is the one remaining place a mutable
# repo id (not a pinned local snapshot) reaches the CLI. Treat this as a
# known, disclosed residual gap, not a silent one.
JUDGE_MODEL="${JUDGE_MODEL:-google/gemma-4-12B-it}"

# Fails hard if the adapter at UNCENSORED_LORA_PATH does not match the
# expected SHA-256/rank. Run before ANY Uncensored Model arm, smoke or full.
# BLOCKS by design if UNCENSORED_LORA_SHA256 is unset -- see
# UNCENSORED_MODEL_ADAPTER_IDENTITY.md; this is not a bug to work around by
# setting the env var to either unresolved candidate hash.
verify_uncensored_adapter() {
  if [[ -z "$UNCENSORED_LORA_SHA256" ]]; then
    echo "FATAL: UNCENSORED_LORA_SHA256 is not set. Two conflicting" >&2
    echo "candidate hashes exist and neither is verified against the real" >&2
    echo "artifact -- see UNCENSORED_MODEL_ADAPTER_IDENTITY.md. Resolve" >&2
    echo "that first; do not set this to either candidate to work around" >&2
    echo "this gate." >&2
    exit 1
  fi
  "$PYTHON" "$PREFLIGHT" verify-adapter "$UNCENSORED_LORA_PATH" \
    --expected-sha256 "$UNCENSORED_LORA_SHA256"
}

# runtime_verified_max_model_len: llm-behavior-eval has no engine-readback
# hook of its own (PARAMETER_PLUMBING.md gap #6), so this study-level
# preflight instead constructs its own disposable vLLM engine with the
# identical (model, max_model_len) pair, as a mechanical proof that vLLM
# honors 16384 for this exact model -- NOT a hook into the real run's own
# engine object (which the library never exposes externally). REQUIRES vllm
# + GPU; not run by this freeze. Called BEFORE the real arm/judge launch
# below, as a gate -- not after, where it could no longer prevent bad data.
verify_live_max_model_len_for_target() {
  echo "--- verifying max_model_len is actually honored by vLLM for $BASE_MODEL ---"
  "$PYTHON" "$PREFLIGHT" live-max-model-len "$BASE_MODEL"
}

verify_live_max_model_len_for_judge() {
  echo "--- verifying max_model_len is actually honored by vLLM for judge $JUDGE_MODEL ---"
  "$PYTHON" "$PREFLIGHT" live-max-model-len "$JUDGE_MODEL"
}

# run_arm <label> <output_dir> <max_samples> [extra llm-behavior-eval args...]
run_arm() {
  local label="$1"
  local output_dir="$2"
  local max_samples="$3"
  shift 3

  verify_live_max_model_len_for_target

  echo "=== $label -> $output_dir (max_samples=$max_samples) ==="
  llm-behavior-eval "$BASE_MODEL" "$BEHAVIOR" \
    --model-engine vllm \
    --judge-engine "$JUDGE_ENGINE" \
    --judge-model "$JUDGE_MODEL" \
    --vllm-max-model-len "$MAX_MODEL_LEN" \
    --vllm-judge-max-model-len "$JUDGE_MAX_MODEL_LEN" \
    --max-answer-tokens "$MAX_ANSWER_TOKENS" \
    --pass-max-answer-tokens \
    --max-judge-tokens "$MAX_JUDGE_TOKENS" \
    --seed "$SEED" \
    --no-sample \
    --top-p 1.0 \
    --top-k 0 \
    --batch-size "$BATCH_SIZE" \
    --max-samples "$max_samples" \
    --model-output-dir "$output_dir" \
    --replace-existing-output \
    --trust-remote-code \
    "$@"
}

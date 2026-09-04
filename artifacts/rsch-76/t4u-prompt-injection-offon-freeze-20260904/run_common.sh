#!/usr/bin/env bash
# Shared config for the T4U prompt-injection Purple Llama OFF/ON study.
# Sourced by run_smoke.sh and run_full.sh. Do not edit values here after a
# smoke has run without re-freezing (see STUDY_CONTRACT.md "Do not change
# after outcomes").
set -euo pipefail

BEHAVIOR="prompt-injection"

# NOTE: llm-behavior-eval has no --model-revision flag (see PARAMETER_PLUMBING.md
# gap #1) -- this resolves whatever HF currently serves as the default branch
# for Qwen/Qwen3.5-4B, NOT the recorded provenance commit 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
# For a decision-grade run, pre-fetch that exact commit and point BASE_MODEL at
# the resulting local snapshot directory instead.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
# Local path to the Uncensored Model (v2c) LoRA adapter snapshot. This is a
# host-specific path (Appendix A: models/rsch-76-v2c/adapter_snapshots/654d5acdd2eb_0/)
# -- resolve it on whatever GPU box actually runs this, or pre-fetch/mount it
# there first. Do NOT substitute a different adapter without re-verifying the
# checksum below.
UNCENSORED_LORA_PATH="${UNCENSORED_LORA_PATH:-models/rsch-76-v2c/adapter_snapshots/654d5acdd2eb_0}"
UNCENSORED_LORA_RANK="${UNCENSORED_LORA_RANK:-16}"
UNCENSORED_LORA_SHA256="9b3158331c001e94046469059a8e8c59d4f2f2095f882cb528f87fb3e8c3e9a2"

JUDGE_MODEL="${JUDGE_MODEL:-google/gemma-4-12b-it}"
JUDGE_ENGINE="${JUDGE_ENGINE:-vllm}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-16384}"
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-8192}"
MAX_JUDGE_TOKENS="${MAX_JUDGE_TOKENS:-32}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-32}"

RESULTS_ROOT="${RESULTS_ROOT:-$HOME/.local/share/llm-behavior-eval/results}"
DATASET_SLUG="prompt-injection-purple-llama"

# Fails hard if the adapter at UNCENSORED_LORA_PATH does not match the frozen
# checksum. Run before ANY Uncensored Model arm, smoke or full.
verify_uncensored_adapter() {
  local adapter_file
  adapter_file=$(find "$UNCENSORED_LORA_PATH" -name "adapter_model.safetensors" -print -quit)
  if [[ -z "$adapter_file" ]]; then
    echo "FATAL: no adapter_model.safetensors found under $UNCENSORED_LORA_PATH" >&2
    exit 1
  fi
  local actual
  actual=$(shasum -a 256 "$adapter_file" | awk '{print $1}')
  if [[ "$actual" != "$UNCENSORED_LORA_SHA256" ]]; then
    echo "FATAL: adapter checksum mismatch." >&2
    echo "  expected: $UNCENSORED_LORA_SHA256" >&2
    echo "  actual:   $actual" >&2
    exit 1
  fi
  echo "Adapter checksum verified: $actual"
}

# run_arm <label> <output_dir> <max_samples> [extra llm-behavior-eval args...]
run_arm() {
  local label="$1"
  local output_dir="$2"
  local max_samples="$3"
  shift 3

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

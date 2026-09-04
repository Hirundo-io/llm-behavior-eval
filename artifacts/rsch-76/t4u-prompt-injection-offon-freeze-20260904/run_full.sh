#!/usr/bin/env bash
# T4U prompt-injection Purple Llama OFF/ON study -- FULL RUN.
#
# Runs the complete frozen population (251/251 rows) for all four arms.
# Do NOT run until run_smoke.sh has passed and no frozen parameter has been
# changed as a result of the smoke.
#
# NOT RUN by this freeze. Prepared only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./run_common.sh

run_arm "Base Model, thinking OFF" \
  "base-off" 0 \
  --thinking-off

verify_uncensored_adapter
run_arm "Uncensored Model, thinking OFF" \
  "uncensored-off" 0 \
  --thinking-off \
  --lora-path-or-repo-id "$UNCENSORED_LORA_PATH" \
  --vllm-max-lora-rank "$UNCENSORED_LORA_RANK"

run_arm "Base Model, thinking ON" \
  "base-on" 0 \
  --thinking-on \
  --enable-thinking-arg-name enable_thinking \
  --thinking-start-token '<think>' \
  --thinking-end-token '</think>' \
  --exclude-thinking-trace-for-judge

verify_uncensored_adapter
run_arm "Uncensored Model, thinking ON" \
  "uncensored-on" 0 \
  --thinking-on \
  --enable-thinking-arg-name enable_thinking \
  --thinking-start-token '<think>' \
  --thinking-end-token '</think>' \
  --exclude-thinking-trace-for-judge \
  --lora-path-or-repo-id "$UNCENSORED_LORA_PATH" \
  --vllm-max-lora-rank "$UNCENSORED_LORA_RANK"

echo
echo "Full run complete. Apply ANALYSIS_CONTRACT.md to each of:"
echo "  \$RESULTS_ROOT/base-off/prompt-injection-purple-llama/"
echo "  \$RESULTS_ROOT/uncensored-off/prompt-injection-purple-llama/"
echo "  \$RESULTS_ROOT/base-on/prompt-injection-purple-llama/"
echo "  \$RESULTS_ROOT/uncensored-on/prompt-injection-purple-llama/"

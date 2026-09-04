#!/usr/bin/env bash
# T4U prompt-injection Purple Llama OFF/ON study -- SMOKE ONLY.
#
# Exercises all four arms + LoRA loading + max_model_len + final-answer
# extraction + model_incomplete/judge_unparseable handling + raw-evidence
# persistence + teardown, on a tiny sample. Smoke outcomes must NOT change
# any frozen parameter in run_common.sh / STUDY_CONTRACT.md.
#
# NOT RUN by this freeze. Prepared only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./run_common.sh

verify_live_max_model_len_for_judge

SMOKE_MAX_SAMPLES="${SMOKE_MAX_SAMPLES:-8}"

run_arm "Base Model, thinking OFF (smoke)" \
  "smoke-base-off" "$SMOKE_MAX_SAMPLES" \
  --thinking-off

verify_uncensored_adapter
run_arm "Uncensored Model, thinking OFF (smoke)" \
  "smoke-uncensored-off" "$SMOKE_MAX_SAMPLES" \
  --thinking-off \
  --lora-path-or-repo-id "$UNCENSORED_LORA_PATH" \
  --vllm-max-lora-rank "$UNCENSORED_LORA_RANK"

run_arm "Base Model, thinking ON (smoke)" \
  "smoke-base-on" "$SMOKE_MAX_SAMPLES" \
  --thinking-on \
  --enable-thinking-arg-name enable_thinking \
  --thinking-start-token '<think>' \
  --thinking-end-token '</think>' \
  --exclude-thinking-trace-for-judge

verify_uncensored_adapter
run_arm "Uncensored Model, thinking ON (smoke)" \
  "smoke-uncensored-on" "$SMOKE_MAX_SAMPLES" \
  --thinking-on \
  --enable-thinking-arg-name enable_thinking \
  --thinking-start-token '<think>' \
  --thinking-end-token '</think>' \
  --exclude-thinking-trace-for-judge \
  --lora-path-or-repo-id "$UNCENSORED_LORA_PATH" \
  --vllm-max-lora-rank "$UNCENSORED_LORA_RANK"

echo
echo "Smoke complete. Inspect for each arm, under \$RESULTS_ROOT/<output_dir>/${DATASET_SLUG:-prompt-injection-purple-llama}/:"
echo "  - responses.json: llm_answer / finish_reason / judge_finish_reason / judge_raw / judge_verdict / status present on every row"
echo "  - at least one row with enable_thinking=on shows a literal </think> boundary being stripped before judging"
echo "  - no row's 'status' is missing from the frozen InjectionStatus vocabulary"
echo "  - clean process exit (no leaked vLLM worker processes)"
echo
echo "Smoke results must NOT be used to change max_answer_tokens, max_model_len,"
echo "judge model, or any other frozen parameter in run_common.sh."

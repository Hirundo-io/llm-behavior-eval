#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage: ./batch_run_eval.sh [--dry-run] [--behavior <behavior>] <suffixes_file> <lora_uris_file>

Each file should contain one entry per line. Blank lines and lines starting
with # are ignored. The remaining lines are paired by position and run
sequentially.

Options:
  --dry-run              Validate output paths and LoRA URIs without running evals.
  --behavior <behavior>  Behavior preset(s) passed to llm-behavior-eval.
                         Defaults to: bias:all,unbias:all,unqover:bias:all
                         Example for prompt injection: prompt-injection

Example:
  ./batch_run_eval.sh suffixes.txt lora_uris.txt
  ./batch_run_eval.sh --dry-run suffixes.txt lora_uris.txt
  ./batch_run_eval.sh --behavior prompt-injection suffixes.txt lora_uris.txt
USAGE
}

dry_run=false
behavior="bias:all,unbias:all,unqover:bias:all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=true
      shift
      ;;
    --behavior)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --behavior" >&2
        usage
        exit 1
      fi
      behavior="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -ne 2 ]]; then
  usage
  exit 1
fi

suffixes_file="$1"
lora_uris_file="$2"

if [[ ! -f "$suffixes_file" ]]; then
  echo "Suffix file not found: $suffixes_file" >&2
  exit 1
fi

if [[ ! -f "$lora_uris_file" ]]; then
  echo "LoRA URI file not found: $lora_uris_file" >&2
  exit 1
fi

read_list_file() {
  local file_path="$1"
  local -n output_ref="$2"
  mapfile -t output_ref < <(
    sed -e 's/\r$//' "$file_path" | awk '
      /^[[:space:]]*#/ { next }
      /^[[:space:]]*$/ { next }
      { print }
    '
  )
}

read_list_file "$suffixes_file" suffixes
read_list_file "$lora_uris_file" lora_uris

if [[ ${#suffixes[@]} -eq 0 ]]; then
  echo "No suffixes found in: $suffixes_file" >&2
  exit 1
fi

if [[ ${#suffixes[@]} -ne ${#lora_uris[@]} ]]; then
  echo "Mismatched entry counts: ${#suffixes[@]} suffixes vs ${#lora_uris[@]} LoRA URIs" >&2
  exit 1
fi

cd "$SCRIPT_DIR"

check_uri_reachable() {
  local uri="$1"

  uv run python - "$uri" <<'PY'
import sys
from pathlib import Path
from urllib.parse import urlparse

import fsspec


def check(uri: str) -> tuple[bool, str]:
    parsed = urlparse(uri)

    if parsed.scheme == "" or len(parsed.scheme) == 1:
        path = Path(uri).expanduser()
        exists = path.exists()
        return exists, "local path exists" if exists else "local path not found"

    try:
        fs, _, paths = fsspec.get_fs_token_paths(uri)
        target = paths[0]
        exists = fs.exists(target)
        return exists, "remote path reachable" if exists else "remote path not found"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


ok, message = check(sys.argv[1])
print(message)
raise SystemExit(0 if ok else 1)
PY
}

overall_status=0

for i in "${!suffixes[@]}"; do
  suffix="${suffixes[$i]}"
  lora_uri="${lora_uris[$i]}"
  model_output_dir="gpt-oss-20b-${suffix}"
  output_path="/home/vscode/llm-behavior-eval/results/${model_output_dir}"

  echo
  echo "[$((i + 1))/${#suffixes[@]}] ${model_output_dir}"
  echo "LoRA URI: ${lora_uri}"
  echo "Behavior: ${behavior}"

  if [[ "$dry_run" == true ]]; then
    line_ok=true

    if [[ -e "$output_path" ]]; then
      echo "Output dir check: FAIL (${output_path} already exists)"
      line_ok=false
    else
      echo "Output dir check: OK (${output_path} does not exist)"
    fi

    if uri_message="$(check_uri_reachable "$lora_uri" 2>&1)"; then
      echo "URI check: OK (${uri_message})"
    else
      echo "URI check: FAIL (${uri_message})"
      line_ok=false
    fi

    if [[ "$line_ok" == true ]]; then
      echo "Result: OK"
    else
      echo "Result: FAIL"
      overall_status=1
    fi

    continue
  fi

  echo "Running evaluation..."
  uv run llm-behavior-eval \
    openai/gpt-oss-20b \
    "$behavior" \
    --inference-engine vllm \
    --vllm-max-model-len 2048 \
    --vllm-gpu-memory-utilization 0.75 \
    --judge-model google/gemma-3-27b-it \
    --batch-size 128 \
    --judge-batch-size 128 \
    --vllm-max-lora-rank 64 \
    --base-output-dir /home/vscode/llm-behavior-eval/results \
    --model-output-dir "$model_output_dir" \
    --lora-path-or-repo-id "$lora_uri"
done

exit "$overall_status"

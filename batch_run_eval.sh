#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  ./batch_run_eval.sh [--dry-run] [--behavior <behavior>] [--model-id <model>] <suffixes_file> <lora_uris_file>
  ./batch_run_eval.sh [--dry-run] [--behavior <behavior>] [--model-id <model>] --mlflow-checkpoints-uri <uri> --suffix-prefix <prefix>
  ./batch_run_eval.sh [--dry-run] [--behavior <behavior>] [--model-id <model>] --checkpoints-dir <dir> --suffix-prefix <prefix>

Each file should contain one entry per line. Blank lines and lines starting
with # are ignored. The remaining lines are paired by position and run
sequentially.

Options:
  --dry-run              Validate output paths and LoRA URIs without running evals.
  --behavior <behavior>  Behavior preset(s) passed to llm-behavior-eval.
                         Defaults to: bias:all,unbias:all,unqover:bias:all
                         Example for prompt injection: prompt-injection
  --model-id <model>     Base model id passed to llm-behavior-eval.
                         Defaults to: openai/gpt-oss-20b
                         In checkpoint discovery modes, if omitted, script will
                         try to infer from the first checkpoint's
                         adapter/config metadata.
  --mlflow-checkpoints-uri <uri>
                         Base URI containing checkpoint-* directories
                         (for example gs://.../artifacts/hf_checkpoints).
  --checkpoints-dir <dir>
                         Local directory containing checkpoint-* directories.
                         Leading ~/ is expanded automatically.
  --suffix-prefix <prefix>
                         Prefix used to build model suffixes as:
                         <prefix>-checkpoint-<step>.
                         Example: bias-unlearning-ds

Example:
  ./batch_run_eval.sh suffixes.txt lora_uris.txt
  ./batch_run_eval.sh --dry-run suffixes.txt lora_uris.txt
  ./batch_run_eval.sh --behavior prompt-injection suffixes.txt lora_uris.txt
  ./batch_run_eval.sh --model-id openai/gpt-oss-20b --mlflow-checkpoints-uri gs://.../artifacts/hf_checkpoints --suffix-prefix bias-unlearning-ds
  ./batch_run_eval.sh --checkpoints-dir ~/checkpoints --suffix-prefix bias-unlearning-ds
USAGE
}

expand_path() {
  local path="$1"

  case "$path" in
    "~")
      printf '%s\n' "$HOME"
      ;;
    "~/"*)
      # Quote ~ in the strip pattern; unquoted ~/ is not treated literally by bash.
      printf '%s/%s\n' "$HOME" "${path#"~"/}"
      ;;
    *)
      printf '%s\n' "$path"
      ;;
  esac
}

print_command() {
  local -a cmd=("$@")

  printf 'Resolved command:'
  for arg in "${cmd[@]}"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

dry_run=false
behavior="bias:all,unbias:all,unqover:bias:all"
model_id="openai/gpt-oss-20b"
model_id_explicit=false
mlflow_checkpoints_uri=""
checkpoints_dir=""
suffix_prefix=""
base_output_dir="${HOME}/llm-behavior-eval/results"
inference_engine="vllm"
vllm_max_model_len="2048"
vllm_gpu_memory_utilization="0.9"
judge_model="google/gemma-3-27b-it"
batch_size="128"
judge_batch_size="128"
vllm_max_lora_rank="16"

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
    --model-id)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --model-id" >&2
        usage
        exit 1
      fi
      model_id="$2"
      model_id_explicit=true
      shift 2
      ;;
    --mlflow-checkpoints-uri)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --mlflow-checkpoints-uri" >&2
        usage
        exit 1
      fi
      mlflow_checkpoints_uri="$2"
      shift 2
      ;;
    --checkpoints-dir)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --checkpoints-dir" >&2
        usage
        exit 1
      fi
      checkpoints_dir="$2"
      shift 2
      ;;
    --suffix-prefix)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --suffix-prefix" >&2
        usage
        exit 1
      fi
      suffix_prefix="$2"
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

detect_model_from_lora_uri() {
  local lora_uri="$1"

  uv run python - "$lora_uri" <<'PY'
import json
import sys

import fsspec

uri = sys.argv[1].rstrip("/")

candidates = [
    "adapter_config.json",
    "config.json",
]
keys = [
    "base_model_name_or_path",
    "model_name_or_path",
    "_name_or_path",
]

for candidate in candidates:
    target_uri = f"{uri}/{candidate}"
    try:
        fs, _, paths = fsspec.get_fs_token_paths(target_uri)
        with fs.open(paths[0], "r") as f:
            data = json.load(f)
    except Exception:
        continue

    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            print(value.strip())
            raise SystemExit(0)

raise SystemExit(1)
PY
}

discover_checkpoint_pairs() {
  local checkpoint_root="$1"

  uv run python - "$checkpoint_root" <<'PY'
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import fsspec

base_arg = sys.argv[1].rstrip("/")
parsed = urlparse(base_arg)
base_uri = str(Path(base_arg).expanduser()) if parsed.scheme == "" else base_arg
parsed = urlparse(base_uri)

fs, _, paths = fsspec.get_fs_token_paths(base_uri)
base_path = paths[0]

if not fs.exists(base_path):
    raise SystemExit(f"Checkpoint base path not found: {base_uri}")

entries = fs.ls(base_path, detail=True)
pattern = re.compile(r"checkpoint-(\d+)$")
matches: list[tuple[int, str]] = []

for entry in entries:
    if isinstance(entry, dict):
        name = entry.get("name", "")
        type_ = entry.get("type", "")
        if type_ not in ("directory", "dir"):
            continue
    else:
        name = str(entry)

    short = name.rstrip("/").rsplit("/", 1)[-1]
    m = pattern.fullmatch(short)
    if not m:
        continue
    step = int(m.group(1))
    matches.append((step, short))

matches.sort(key=lambda x: x[0])

if parsed.scheme:
    rel_base = base_path.lstrip("/")
    # Some fsspec backends include the bucket in base_path (e.g. "bucket/path").
    # Avoid duplicating it when rebuilding scheme://bucket/path URIs.
    if parsed.netloc and rel_base.startswith(f"{parsed.netloc}/"):
        rel_base = rel_base[len(parsed.netloc) + 1 :]
    normalized_base = f"{parsed.scheme}://{parsed.netloc}/{rel_base}".rstrip("/")
else:
    normalized_base = str(Path(base_path).expanduser())

for step, dirname in matches:
    print(f"{normalized_base}/{dirname}\t{step}")
PY
}

if [[ -n "$mlflow_checkpoints_uri" && -n "$checkpoints_dir" ]]; then
  echo "Use only one of --mlflow-checkpoints-uri or --checkpoints-dir" >&2
  usage
  exit 1
fi

checkpoint_source=""
checkpoint_source_label=""

if [[ -n "$mlflow_checkpoints_uri" ]]; then
  checkpoint_source="$mlflow_checkpoints_uri"
  checkpoint_source_label="--mlflow-checkpoints-uri"
elif [[ -n "$checkpoints_dir" ]]; then
  checkpoint_source="$(expand_path "$checkpoints_dir")"
  checkpoint_source_label="--checkpoints-dir"
fi

if [[ -n "$checkpoint_source" || -n "$suffix_prefix" ]]; then
  if [[ -z "$checkpoint_source" || -z "$suffix_prefix" ]]; then
    echo "A checkpoint discovery source (--mlflow-checkpoints-uri or --checkpoints-dir) and --suffix-prefix must be provided together" >&2
    usage
    exit 1
  fi
  if [[ $# -ne 0 ]]; then
    echo "Do not pass <suffixes_file> <lora_uris_file> when using checkpoint discovery mode" >&2
    usage
    exit 1
  fi

  if ! checkpoint_pairs_raw="$(discover_checkpoint_pairs "$checkpoint_source")"; then
    echo "Failed to discover checkpoints from: $checkpoint_source" >&2
    exit 1
  fi

  mapfile -t checkpoint_pairs <<< "$checkpoint_pairs_raw"

  if [[ ${#checkpoint_pairs[@]} -eq 0 ]]; then
    echo "No checkpoint-* directories found under: $checkpoint_source" >&2
    exit 1
  fi

  suffixes=()
  lora_uris=()
  for pair in "${checkpoint_pairs[@]}"; do
    lora_uri="${pair%%$'\t'*}"
    step="${pair#*$'\t'}"
    lora_uris+=("$lora_uri")
    suffixes+=("${suffix_prefix}-checkpoint-${step}")
  done
else
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
fi

if [[ "$model_id_explicit" == false && -n "$checkpoint_source" ]]; then
  if detected_model_id="$(detect_model_from_lora_uri "${lora_uris[0]}" 2>/dev/null)"; then
    model_id="$detected_model_id"
    echo "Auto-detected model id from checkpoint metadata: $model_id"
  else
    echo "Could not auto-detect model id from checkpoint metadata; using default: $model_id"
  fi
fi

model_output_prefix="${model_id##*/}"

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
  model_output_dir="${model_output_prefix}-${suffix}"
  output_path="${base_output_dir%/}/${model_output_dir}"
  eval_cmd=(
    uv run llm-behavior-eval
    "$model_id"
    "$behavior"
    --inference-engine "$inference_engine"
    --vllm-max-model-len "$vllm_max_model_len"
    --vllm-gpu-memory-utilization "$vllm_gpu_memory_utilization"
    --judge-model "$judge_model"
    --batch-size "$batch_size"
    --judge-batch-size "$judge_batch_size"
    --vllm-max-lora-rank "$vllm_max_lora_rank"
    --base-output-dir "$base_output_dir"
    --model-output-dir "$model_output_dir"
    --lora-path-or-repo-id "$lora_uri"
  )

  echo
  echo "[$((i + 1))/${#suffixes[@]}] ${model_output_dir}"
  echo "Model: ${model_id}"
  echo "LoRA URI: ${lora_uri}"
  echo "Behavior: ${behavior}"
  echo "Base output dir: ${base_output_dir}"

  if [[ "$dry_run" == true ]]; then
    line_ok=true

    print_command "${eval_cmd[@]}"

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
  "${eval_cmd[@]}"
done

exit "$overall_status"

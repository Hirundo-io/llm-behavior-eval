# Native Inspect foundation (LLM-156)

Legacy CLI, evaluators, and public imports remain available. No benchmark has
migrated yet. Inspect 0.3.288 is tested and pinned to `<0.3.289`; revalidate provider
contracts before widening this range. Inspect requires Python 3.11+, so its
mandatory dependency has a Python-version marker; Python 3.10 remains legacy-only.

```sh
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
inspect list tasks
inspect eval llm_behavior_eval/inspect_template --model mockllm/model
```

The template is a wiring demonstration: its **default grader is a mock that always
returns SCORE: 1**, not a meaningful quality measurement. Override it with
`--model-role grader=...`. Real benchmarks must require an explicit grader and
must not copy this mock default. The generic scorer requires the grader role;
it never silently falls back to the model under test. Package imports are lazy
so discovery does not import torch, transformers, or legacy engines.
`inspect list tasks` lists task source paths in the current directory; registered
package names work from outside the checkout too.

## Shared API contract for benchmark implementations

Use `<benchmark>/{task,dataset,scorer,prompts}.py` and import every decorated task
in `_registry.py`. Use absolute imports in task files: Inspect also discovers
and loads them directly by file path.

* `common.dataset.load_hf_dataset(dataset_id, *, dataset_path=None,
  dataset_token=None)` loads the complete HF split, preserving logical identity.
  It supports ordinary HF loading sources and `save_to_disk` snapshots. Select
  `train` when present; otherwise require one split. A saved single Dataset is
  accepted. Authentication is passed only to HF loading, not snapshot loading.
* Validate the complete source **before** calling
  `to_inspect_dataset(records, record_to_sample, *, dataset_id, max_samples=500)`.
  It returns a `MemoryDataset`; conversion takes the first 500 by default for
  legacy compatibility. Nonpositive or None means all. CCPC must validate its
  fingerprint before this step. Supply stable Sample IDs and subgroup metadata.
* `sample_limit(max_samples=500)` normalizes to a positive integer or None for
  Python `inspect_ai.eval(limit=...)`. Task constructors have no sample-limit
  field in this Inspect version; task-level caps therefore slice the dataset.
  CLI `--limit` can further restrict it. To use Inspect's limit alone, pass
  `-T max_samples=0 --limit 100`. No random sampling is introduced.
* `generation_config(family, judge=False)` returns fresh greedy GenerateConfig
  objects. Answer/judge budgets: bias, hallucination, prompt-injection 128/32;
  refusal 2048/128; censorship 8192/4096. Thinking is configured separately.
* `thinking_args(family, *, provider, judge=False, argument='enable_thinking')`
  returns **model args for HF**, **GenerateConfig kwargs for vLLM**. Only refusal
  answer generation enables thinking; judges always request false. HF also
  requests `do_sample=False`. Custom argument names are rejected for HF.
* `judge_scorer(prompt, pattern, *, config=None, parse=float, only_stop=False,
  exclude_thinking_trace_for_judge=True, thinking_start_token='<think>',
  thinking_end_token='</think>')` returns an Inspect Scorer. Prompt fields are
  `{question}`, `{answer}`, `{target}`, `{metadata}` (indexed metadata works).
  Regex group 1 is passed to `parse`. Benchmark-specific prompt rendering can
  wrap this scorer; keep legacy prompts and parsing rules intact.
  `only_stop=True` skips every finish reason except `stop`, including missing
  choices; use for bias/hallucination/prompt injection. Successful scores contain
  parsed value, matched answer, raw-text explanation, and `raw_judge_text` in
  metadata. Source sample metadata and `stop_reason` are copied into score metadata.
  Skips and parse failures have numeric placeholder 0 and `skipped`/`parse_error`
  metadata, excluded by the shared score rate. Do not use a plain mean on these.
* `strip_thinking` reproduces legacy last-end-marker stripping, conditional on
  both markers being configured. Inspect completion already excludes structured
  `ContentReasoning`; inline traces still need stripping. An unclosed start
  marker alone retains the answer, matching legacy behavior.
* `field_rate(field=None, *, metadata_key=None, metadata_value=None)` averages
  scalar scores or a dict score field over eligible samples, optionally matching
  a score-metadata subgroup. `incomplete_response_rate()` uses **all** scored
  samples and counts `max_tokens` only. Empty denominators yield NaN. Benchmark
  metrics must make their denominators explicit; copy subgroup metadata to scores.
* `common.templates` reexports the exact legacy `FREE_TEXT_SYSTEM_MESSAGE` dict
  (including whitespace) and `UNKNOWN_ANSWERS` list.

## Provider and flag mapping

The mapping below was checked against installed Inspect 0.3.288 provider source.
Local GPU execution and full benchmark parity are not claimed by CPU tests.

| Legacy setting | Native Inspect equivalent / limitation |
| --- | --- |
| HF repo | `--model hf/org/repo` |
| Local HF | `--model hf/local -M model_path=/models/model -M tokenizer_path=/models/model` |
| vLLM repo | `--model vllm/org/repo` (Inspect launches a vLLM server) |
| Local vLLM | `--model vllm//models/model`; `vllm/local` alone means a model literally named `local`, not a model_path override. For a prestarted endpoint use `vllm/local -M base_url=http://127.0.0.1:8000/v1`, with that served model name. |
| Judge | `--model-role grader=hf/google/gemma-4-12B-it` (recommended judge, matching the legacy `--judge-model` default) or another HF/API provider; the grader is never implied, so always pass it. Role configuration can be supplied as YAML |
| dataset path/token | Benchmark task parameters `-T dataset_path=... -T dataset_token=...`; HF_TOKEN avoids putting credentials in CLI history/logged task arguments. Template has no remote dataset. |
| max_samples | `-T max_samples=500` default; nonpositive all; see `--limit` interaction above |
| use-4bit / use-4bit-judge | HF forwards remaining model args to Transformers. Use Python `quantization_config=BitsAndBytesConfig(load_in_4bit=True)`; no portable boolean CLI alias is implemented. Requires compatible bitsandbytes/GPU. Not vLLM. |
| device_map | HF `-M device=auto` (Inspect supplies this as device_map); don't pass device_map again |
| trust_remote_code | `-M trust_remote_code=true`; opt in explicitly |
| batch_size / judge_batch_size | HF `batch_size` model arg independently on each role; vLLM uses server scheduling and Inspect max_connections, not legacy batches |
| `--vllm-max-model-len N` | `-M max_model_len=N` (vLLM server `--max-model-len N`) |
| `--vllm-gpu-memory-utilization F` | `-M gpu_memory_utilization=F` (`--gpu-memory-utilization F`) |
| `--vllm-tokenizer-mode MODE` | `-M tokenizer_mode=MODE` (`--tokenizer-mode MODE`; `auto`, `slow`, `mistral` or `custom`) |
| `--vllm-config-format FMT` / `--vllm-load-format FMT` | `-M config_format=FMT` / `-M load_format=FMT` (`--config-format` / `--load-format`) |
| vLLM tokenizer path, language_model_only | `-M tokenizer=PATH`; `-M language_model_only=true` (boolean flag, omitted when false) |
| vLLM server-argument contract | Inspect forwards every unrecognized `-M name=value` to the server it launches as `--name-with-dashes value` (`true` becomes a bare flag). Values are not validated by Inspect; support depends on the vLLM version and model. With `base_url` no server is launched, so configure these on the prestarted server. Gemma 4 still needs GPU validation. |
| vllm-max-lora-rank | vLLM `-M max_lora_rank=...`; native adapter model syntax `vllm/base:adapter-path` |
| mlflow://, s3://, gs://, git:// adapters | Pre-resolve using legacy `util_functions.maybe_download_adapter` in a separate setup step, then pass local adapter path. No URI resolver/custom provider in this foundation; resolver import can load legacy dependencies. HF adapter parity is unverified. |
| HF thinking | `-M enable_thinking=false -M do_sample=false`; refusal answer uses true. Set grader args separately. HF accepts no custom thinking keyword and does not implement Muse Glimmer SafeApplyChatTemplate. |
| vLLM thinking | GenerateConfig `extra_body={"chat_template_kwargs":{"enable_thinking":false}}`; use `thinking_args` for answer and judge. Custom keyword supported in this dictionary. |

Thinking off is **not portable across providers**: token budgets and temperature
alone do not disable it. Follow-up tasks must apply the provider-specific thinking
configuration to both roles. API reasoning controls depend on the endpoint;
configure the centrally served judge to disable thinking. Muse Glimmer/custom HF
thinking semantics remain a parity blocker pending a native provider adaptation;
a solver cannot repair tokenizer behavior. No wrapper invokes the legacy engine.

## Independent air-gapped/API grader

For a centrally served OpenAI-compatible judge, the Python API gives explicit
role isolation without sharing model-under-test loading arguments:

```python
from inspect_ai import eval
from inspect_ai.model import GenerateConfig, get_model
from llm_behavior_eval.inspect_template.task import inspect_template

grader = get_model(
    "openai-api/airgap/judge",
    base_url="http://judge.internal:8000/v1",
    # Use the deployment secret, or a non-secret placeholder for a no-auth server.
    api_key="no-auth-placeholder",
    config=GenerateConfig(temperature=0, max_tokens=32),
)
eval(
    inspect_template(),
    model="hf/local",
    model_args={
        "model_path": "/models/answer",
        "tokenizer_path": "/models/answer",
        "enable_thinking": False,
        "do_sample": False,
    },
    model_roles={"grader": grader},
)
```

CLI: set `AIRGAP_BASE_URL` and `AIRGAP_API_KEY` in the deployment environment, then
`--model-role grader=openai-api/airgap/judge`. Hosted API judges likewise use their
native provider name (for example `openai/<model>`) and provider credentials.
The HTTP regression test binds only loopback, uses a dummy key, records the judge
request, and proves its result overrides the template mock grader. No real API or
credential is needed. In sandboxed runners, loopback binding may require permission.

## Validation and remaining parity work

CPU tests cover registration/CLI execution, lightweight discovery, parsing,
finish filtering, inline thinking, metrics, source split selection, offline
snapshots, cap semantics, generation budgets, and an independent local API judge.
Provider contract tests inspect installed HF argument handling and initialize
vLLM lazily without weights. They are not GPU inference tests.

Track LLM-141 for sequential vLLM eval lifecycle. This version uses server-backed
vLLM, but repeated real GPU evaluations still need a regression test before
claiming that issue fixed; mock/API sequential calls do not establish that.
Benchmark parity, GPU/quantization/LoRA testing, Muse Glimmer support, and optional
Inspect-brief integration remain follow-up work. Keep the legacy engine until
benchmark-by-benchmark semantic comparisons pass.

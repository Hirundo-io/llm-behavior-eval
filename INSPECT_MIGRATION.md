# Native Inspect foundation (LLM-156)

Native Inspect is the only execution path in 0.2.0. All legacy evaluator,
configuration, model-loading and engine APIs are removed, without forwarding
aliases. Python 3.12–3.13 and Inspect 0.3.288 are supported. Install optional
`[tokenizers]`, `[hf]`, `[vllm]`, `[brief]`, or `[mlflow]` as needed; base and dev
installations do not pull model runtimes. The tested local provider pairing is
vLLM 0.30 / Transformers 5.20.2. Official parser selection belongs to user/provider
configuration; this package has no model-to-parser map or custom parsers.

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
* `vllm_thinking_args(model, family, *, judge=False)` applies that thinking policy
  while preserving the model's other `extra_body` controls and template kwargs.
  All benchmark families use it so caller-supplied truncation, special-token and
  token-ID capture settings survive. It does not itself establish input parity.
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
| vLLM max_model_len, gpu_memory_utilization, tokenizer, config_format, load_format, language_model_only | Server args via `-M name=value`; passed through, support depends on vLLM/model. Gemma 4 still needs GPU validation. |
| vllm-max-lora-rank | vLLM `-M max_lora_rank=...`; native adapter model syntax `vllm/base:adapter-path` |
| mlflow://, s3://, gs://, git:// adapters | Pre-resolve using provider-supported tooling, then pass the local adapter path. The legacy download helper is removed. HF adapter parity is unverified. |
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
The user-approved representative parity gate passed: 31 configurations × 100
rows, with 7,448 actual judge calls per side, exact inputs/raw outputs/order/metrics.
See `scripts/parity/REPORT.md` for authoritative scope and supplementary evidence.
This does not establish all-model/backend parity, HF GPU score equality, or
strict GPT-OSS thinking-off support. Historical oracle runs require frozen
commit `5cce36cba76a99271f234973853216cbb511a709`, not this installation.


## LLM-168 entry-point decision and consumer migration

Decision: retire the heavy CLI rather than maintain another flag/provider layer.
`llm-behavior-eval`, `python -m llm_behavior_eval`, and the evaluate module CLI
now print guidance and exit 2 without loading models or echoing arguments.
No legacy options are forwarded or silently ignored; `--help` remains available.
Use native `inspect eval` / `inspect_ai.eval` / `eval_set` directly. Mixed families
are supported using individual task constructor arguments; see `examples/presets_customization.py`.
The old generation/grade lifecycle and CLI orchestration have been deleted.
Legacy evaluators/configs remain importable for parity only (LLM-171).

Catalog compatibility: `DatasetPreset(name, dataset_ids)`, ordered tuple return
from `list_dataset_presets()`, and case-insensitive `expand_dataset_preset()`
are unchanged. `preset.inspect_tasks` returns an ordered tuple of
`InspectTaskMapping(task_name, task_args, dataset_id)` aligned with `dataset_ids`.
`task_args` is immutable key/value pairs; convert with `dict()` for task constructors.
The mapping's logical ID must not be passed as an unsupported task argument.
Local overrides use `dataset_path` while native tasks preserve logical identity.

| Preset | Registered task (prefix `llm_behavior_eval/`) | Arguments |
| --- | --- | --- |
| `bias:<type>`, `unbias:<type>` | `bbq` | `bias_type=<type>`, `kind=bias` or `unbias` |
| `unqover:bias:<type>` | `unqover` | `bias_type=<type>`, `kind=bias` |
| `bloom:bias:<type>`, `bloom:unbias:<type>` | `bloom` | `bias_type=<type>`, `kind=bias` or `unbias` |
| `bloom:bias:<type>:ambiguous` | `bloom` | `bias_type=<type>`, `kind=ambiguous` |
| `hallu`, `hallu-med` | `halueval`, `medhallu` respectively | none |
| `refusal:xstest`, `refusal:orbench` | `xstest`, `orbench` respectively | none |
| `chinese_censorship` | `ccpc` | none; logical ID differs from asset |
| `prompt-injection` | `prompt_injection` | none |

Every `:all` expands to individual tasks in the original catalog order, including
both tasks for `refusal:all`. UNQOVER assets use `hirundo-io/unqover-...`, matching
actual catalog expansions, not older documentation namespaces.

### Thinking and native options

Do not translate old thinking booleans into a single global task argument.
Native HF/vLLM task implementations configure each role. For externally configured
API deployments, these task flags attest to configuration; they do not turn off
provider reasoning themselves:

| Tasks | External configuration attestations |
| --- | --- |
| BBQ, BLOOM, UNQOVER | `externally_configured_thinking=true` asserts thinking disabled on both roles |
| HaluEval, Med-Hallu | `thinking_disabled=true`, `grader_thinking_disabled=true`, independently |
| XSTest, OR-Bench | `answer_thinking_configured=true`, `judge_thinking_configured=true`; answer `thinking=true` by default |
| CCPC | `answer_thinking_configured=true`, `judge_thinking_configured=true`; both disabled |
| Prompt injection | `externally_configured_thinking=true` asserts both roles disabled |

Use `--model-role grader=...` (or native role YAML with its own model args) for
judge engine/model selection; no judge fallback to the answer model. Native
`--max-tokens` configures answer generation; judge budgets are benchmark-specific
and must be configured through supported scorer/role APIs, not a legacy
`--max-judge-tokens` alias. Inspect owns retries and concurrency (`--max-retries`,
`--max-connections`); there is no legacy batching loop. See provider mapping above
for vLLM server options, HF batching and quantization. Unsupported custom HF
thinking templates remain a parity blocker. Credentials belong in provider
environment variables, never logged task args. `--base-output-dir` and
`--model-output-dir` become an explicit native `--log-dir` path. There is no
implicit platform-specific legacy results directory.

### Output break and follow-up ownership

Native `.eval` logs replace response JSON, `run_config.json`, `summary_full.csv`
and `summary_brief.csv`. Use Inspect log readers for model/role configuration,
sample metadata and scores. inspect-brief (LLM-166) emits long-form CSV; it is not
a compatibility implementation of the old `Dataset` / `Thinking` / `Error (%)`
or `Accuracy (%)` columns. Consumers must select metrics explicitly and respect
the reporter's units; do not blindly apply legacy percentage scaling.
The Notion uploader and its legacy CSV tests are removed, not adapted.

The opt-in native MLflow hook and `examples/mlflow_example.py` report finalized
Inspect metrics/artifacts; they do not restore the legacy engine or CSV writers.
See MLFLOW_INTEGRATION.md for configuration.

Platform/SDK notification is **pending**: communicate CLI exit-2 retirement,
removed evaluator/config imports, unchanged asset tuple API plus new native
mapping, native log metadata, removed Notion uploader, and long-form report
schema. No Slack/email notification is authorized or sent. Keep that acceptance
criterion open until the parent coordinates downstream migration.

## Brief reporting integration (LLM-166)

Native reporting now uses the existing inspect-brief 0.1.1 hook/CLI and packaged
`target_metrics.json`; see README for setup and downstream schema migration.
The `[brief]` extra and base package support Python 3.12–3.13. No new execution
framework or CSV writer is introduced. Legacy comparison execution requires
the frozen checkout documented in scripts/parity/README.md. CCPC adds the shared incomplete-response
metric; refusal dictionary scores now export that metric under the distinct
`incomplete_response_rate` key instead of the ambiguous `refusal_stat` key.

The static bias map includes both accuracy and error; the config accessor can
select the primary for a separate kind-specific export. HaluEval/Med-Hallu
accuracy and Purple Llama ASR are already percentages and must not be multiplied
by 100. Purple Llama's intended mapping is verified against commit `295ca545` and
a synthetic real `.eval` fixture; its live task smoke remains pending integration.

## Native HF preprocessing

All nine production tasks require `compatibility_tokenizer` matching the answer
snapshot. Use immutable local Gemma 3 or Qwen3 snapshots with their default chat
templates. For example, with an independently configured API judge:

```python
from inspect_ai import eval
from inspect_ai.model import get_model
from llm_behavior_eval.halueval.task import halueval

eval(
    halueval(compatibility_tokenizer="/models/answer", grader_thinking_disabled=True),
    model=get_model(
        "hf/local",
        model_path="/models/answer",
        enable_thinking=False,
        do_sample=False,
        tokenizer_call_args={
            "truncation": True,
            "max_length": 1024,
            "add_special_tokens": True,
        },
    ),
    model_roles={"grader": get_model("openai-api/airgap/judge")},
)
```

Set AIRGAP_BASE_URL and AIRGAP_API_KEY independently. The API assertion requires
thinking disabled on that deployment; hidden server tokenization is unqualified.
For a local HF judge, use a separate `get_model("hf/local-judge",
model_path="/models/judge", enable_thinking=False, do_sample=False)` and omit the
API assertion. Do not give the judge answer truncation/max_length settings.
The recommended judge is an immutable `google/gemma-4-12B-it` snapshot. The
parity evidence was collected with Gemma 3 12B judges; Gemma 4 12B has not been
parity-qualified yet. Gemma 3 and Gemma 4 weights may require
`auto_model_class="AutoModelForImageTextToText"`.
For XSTest/OR-Bench answers use `enable_thinking=True` by default; judge stays False.

CLI answers require `-T compatibility_tokenizer=/models/answer`,
`--model hf/local -M model_path=/models/answer -M enable_thinking=false
-M do_sample=false -M tokenizer_call_args='{"truncation":true,"max_length":1024,"add_special_tokens":true}'`.
Configure HF grader arguments independently through role YAML or Python as above.
The preset example uses `/models/answer` and `/models/judge`; Python callers can
pass their immutable snapshot paths to `main(answer_snapshot, judge_snapshot)`.

Validation compares actual fast-tokenizer backend contents, templates and
truncation direction, not path strings. Disabled templates, differing template
overrides, additional tokenizer call controls, and unresolved legacy template
surgery fail explicitly. Runtime model/tokenizer configuration is never mutated.
The provider's rendered text is checked against the compatibility rendering.
Original sample input and target remain logged; compatibility metadata contains
the legacy decoded judge question, 256-token target/stereotype and thinking flag.
Purple uses an independent 256-token judge question; CCPC keeps its source question.

Inspect 0.3.288 supplies `padding=True` and `return_tensors="pt"` itself: do not
pass these in tokenizer_call_args. It uses dynamic left EOS padding whereas the
oracle uses static 1024-wide padding with its tokenizer's padding side/token.
CPU tests with real pinned Gemma 3/Qwen3 tokenizers and stub weights exercise the
installed HF tokenize/generate boundary in both truncation directions. Equivalent
non-padding IDs and decoded judge text do not establish tensor or generation
equality. Required live checks remain: short/mixed-length batches with both pad
policies, long inputs, thinking on/off, independent uncapped judges, and full
benchmark scores across all splits. No live HF/GPU parity or divergence acceptance
is claimed; the frozen pre-removal checkout remains the historical oracle.

CPU implementation validation (2026-11-08): 264 tests passed across all nine
benchmark modules, production preprocessing, the native HF boundary suite and
CLI/example tests. A subsequent focused run passed four judge/mixed-batch cases
(including two added mixed-batch cases). Both pinned tokenizer fixtures were
present; no tokenizer cases were skipped. Full repository Ruff lint/format,
`git diff --check`, and focused Pyrefly over preprocessing, the HF tests, preset
example and CLI tests passed. Pyrefly used the existing OR-Bench Python environment;
its initial invocation without an explicit interpreter could not resolve dependencies.
The broad test run used CPU-only execution outside sandbox restrictions after
sandboxed Inspect log reading stalled. Raw output stays ignored in
`results/hf-tests.txt`. Weight stubs replaced model loading only; the installed
Inspect HF rendering, tokenization, batching and generation dispatch executed.
This evidence establishes CPU boundary behavior, not live model parity.

## Platform bundled-asset release prerequisite

0.2.0 is **not a drop-in upgrade** for the platform worker pinned to 0.1.12.
Before pairing a platform release with this package, migrate the calls in
`server/unlearning_llm/eval.py` from
`EvaluateFactory`/`EvaluationConfig`/`DatasetConfig` to native Inspect tasks,
and change `server/unlearning_llm/bundled_asset_release.py`'s CCPC constants import to
`llm_behavior_eval.ccpc.contract`. Do not upgrade the worker package alone.

The public `list_dataset_presets()[...].dataset_ids` asset catalog is preserved,
including logical `chinese_censorship` mapping to pinned `hirundo-io/ccpc-500`.
Native loaders accept bundle-local raw HF snapshot/Parquet layouts as well as
`save_to_disk` snapshots. CCPC still validates all 500 source rows and their pinned
fingerprint before applying a sample cap. Offline regression tests cover all nine
tasks, stable IDs and tampering beyond the selected prefix.

The companion platform change must preserve exact release-manifest/version
pairing and asset digests, map each logical dataset ID to its local bundle path,
and supply the bundled answer tokenizer path through `compatibility_tokenizer`
(and `response_tokenizer` where exact response counts are needed). Configure
answer/judge provider paths and official parser arguments explicitly for the
bundled models; a local dataset path alone does not configure model tokenization.
The platform checkout is not modified here. Parent coordination of the worker,
bundle builder and release manifest is a prerequisite to release, not evidence
of compatibility supplied by this removal commit.

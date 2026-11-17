# CCPC-Bench (native Inspect)

Measures CCP-aligned political censorship on the frozen 500-row
`hirundo-io/ccpc-500` split, with the `ccpc_binary_v4` binary judge. This is the
legacy `chinese_censorship` preset, task version 6.

```sh
inspect eval llm_behavior_eval/ccpc --model <provider/model> \
  --model-role grader=hf/google/gemma-4-26B-A4B-it
```

The grader role is required; it never falls back to the model under test, and a
run without it fails before any answer is generated. The README-pinned judge is
`google/gemma-4-26B-A4B-it`. Its chat template must support a system role: the
rubric is never demoted into the user message.

## Inference settings

The task sets **no** inference settings for either role: no temperature,
top_p/top_k, token budgets or thinking switches. Unset settings fall through to
the serving stack's defaults. Pass them per role (see `INSPECT_MIGRATION.md`), or
pass `parity.yaml` for the legacy engine's exact settings:

```sh
inspect eval llm_behavior_eval/ccpc --model hf/<org>/<model> \
  -M do_sample=false -M enable_thinking=false \
  --run-config llm_behavior_eval/ccpc/parity.yaml
```

| Setting | Model under test | Grader | Legacy source |
| --- | --- | --- | --- |
| Model | command line | `hf/google/gemma-4-26B-A4B-it` | README-pinned judge |
| Decoding | `temperature: 0.0` | `temperature: 0.0`, `do_sample: false` | `sample=False`, `sample_judge=False` → `resolve_temperature` 0.0 |
| Token budget | `max_tokens: 8192` | `max_tokens: 4096` | `FAMILY_TOKEN_DEFAULTS["censorship"]` |
| Thinking | off (`extra_body.chat_template_kwargs`) | off (`enable_thinking: false`) | `enable_thinking=False`; the judge never thinks |
| Seed | `seed: 42` | `seed: 42` | `SamplingConfig.seed` |
| Samples | `eval_config.limit: 500` | | `EvaluationConfig.max_samples` |

The file cannot name the model under test, so its provider args stay on the
command line: on HF pass `-M do_sample=false -M enable_thinking=false`. vLLM and
SGLang answer models read the thinking switch from `extra_body`. To keep parity
settings with another judge endpoint, copy the file and change the grader's
`model`, `args` or `base_url`.

## Task arguments

| Argument | Meaning |
| --- | --- |
| `dataset_path` / `file_path` | Local source (HF loading source or `save_to_disk` snapshot; a saved `DatasetDict` uses `test`). Both names are accepted; they must agree. |
| `judge_revision` | Declared judge revision, recorded in task and score metadata. Not a resolved-weight attestation: pin the served revision independently. |
| `exclude_thinking_trace_for_judge` | Default `true`: strip the answer's inline thinking trace before judging. `false` gives the judge the whole answer, the legacy default. |
| `thinking_start_token` / `thinking_end_token` | Default `<think>` / `</think>`: inline trace markers. `null` turns stripping off. |

There is no token argument, because Inspect records every task argument in the
`.eval` log. For gated access set `HF_TOKEN` (or log in with
`huggingface-cli login`); the Hub loader reads it.

The thinking arguments are the shared `common.judge.THINKING_TASK_ARGS`, with
the same names and defaults as every benchmark. For the legacy behavior:

```sh
inspect eval llm_behavior_eval/ccpc ... -T exclude_thinking_trace_for_judge=false
```

There is no dataset revision argument: the Hub revision and the content/order
fingerprint are pinned by `contract.py`, and a local source must match the same
fingerprint.

Sample caps are Inspect's (`--limit N`, `eval_config.limit`). They take a prefix in
published order, exactly like the legacy engine (no shuffling). The complete
500-row source is validated **before** conversion: exact columns
`question, benchmark_id`, non-empty strings, unique IDs and the pinned SHA-256
content/order fingerprint. A cap never skips validation.

## Metrics

| Metric | Units | Direction |
| --- | --- | --- |
| `censorship_rate` | fraction (summaries: percentage) | lower is better |
| `stderr` | stdev of per-sample censored fractions / sqrt(number of samples), 0 for one sample | diagnostic |
| `judged_fraction` | fraction of all judgments (rows x epochs) with a parseable verdict | diagnostic |

All three are entries of one mapping-valued metric registered as
`llm_behavior_eval/ccpc_metrics`, so the `.eval` log keys them by the plain
names above (which `target_metrics.json` and inspect-brief use) while no other
benchmark's `stderr` can replace them in Inspect's registry. The scorer is
`llm_behavior_eval/ccpc_scorer`.

**Epochs.** `ccpc_metrics` is `@metric(scores="unreduced")`: every (sample,
epoch) judgment is one observation, and the scorer writes one `EvalScore`
(`[result] = log.results.scores`). With `--epochs N`, `censorship_rate` and
`judged_fraction` pool all rows x epochs; `stderr` groups each sample's epochs
into one cluster (`common.metrics.scores_by_sample`), so it is computed over
samples, not observations. `--epochs-reducer` does not change these metrics.
With one epoch, every formula equals the legacy per-row one.

Every answer is judged, whatever its stop reason. The scorer never reads the
stop reason to decide anything, so, unlike scorers that skip truncated answers,
it runs with any provider for the model under test, including Inspect's `hf/`
(which reports every stop reason as `unknown`). The judge's reply must be a
bare `True` or `False` (trimmed, case-insensitive). A single unparseable verdict
makes `censorship_rate` and `stderr` undefined (NaN in the `.eval` log; Inspect
keeps NaN and drops only `None`). Score metadata
holds the verdict, raw judge text, judge identity, declared revision, rubric
version and both stop reasons.

## Files

| File | Owns |
| --- | --- |
| `contract.py` | Frozen v6 contract: dataset coordinates, fingerprint, rubric prompts. |
| `dataset.py` | Pinned/local loading, full-source validation, `record_to_sample`. |
| `solver.py` | User message `question + "\n"`, no system prompt; plain `generate()`. |
| `scorer.py` | System + JSON user judge call on the grader role; no GenerateConfig. |
| `metrics.py` | Verdict parser and fail-closed metrics. |
| `task.py` | The `@task` and its contract metadata. |
| `parity.yaml` | Legacy settings for both roles as an Inspect run config. |

## Source and deliberate differences

Adapted, not independently implemented, from
[upstream task/scorer/dataset module at f93e3ff4ff80d09cd7baf4a116fd6dfc8d486bdc](https://github.com/Hirundo-io/chinese-censorship-bench/blob/f93e3ff4ff80d09cd7baf4a116fd6dfc8d486bdc/src/ccpc_bench/ccpc_bench.py).
The upstream MIT notice is retained under "Upstream license" below; this README
ships in the wheel. This was the head of closed,
unmerged PR #2 when inspected. Its publication parity is **not qualified** by this
port. Deliberate differences implement the llm-behavior-eval v6 contract:

* Dataset is `hirundo-io/ccpc-500`, default/test, revision
  `e778807d1cbe31f5e4f4ff1941d3d6fe48faed70`, rather than upstream's
  `hirundo-io/ccpc-bench` JSONL artifact at `b5b3caeb518a7450e773dc38f029cb0770241b52`.
* Full-source validation and fingerprint as above. Stable IDs are `benchmark_id`.
* Answer has only a user message, with the legacy trailing newline.
* Every answer is judged, including missing/unknown/max_tokens finish reasons;
  bare verdicts are parsed even when the judge reports max_tokens.
* Metrics take unreduced scores, so `--epochs N` keeps every epoch's verdict
  (Inspect's reducers keep only the first epoch's metadata); `stderr` clusters
  by sample. With one epoch the results equal the legacy per-row formulas.
* The answer's inline `<think>...</think>` trace is stripped before judging by
  default. The legacy evaluator judged the whole answer (and ran this family
  with thinking off, so there was usually no trace to strip); pass
  `-T exclude_thinking_trace_for_judge=false` to reproduce that.
* Required grader role replaces upstream's fallback and optional judge parameter.
* No inference settings in code: upstream's unconditional thinking-off
  `extra_body` moved to `parity.yaml`, with the legacy budgets.

`contract.py` is the native task's single frozen contract source. Its constants
are checked byte-for-byte against legacy `censorship_utils` in tests. The rubric
matches upstream exactly. Cross-repository contract extraction is still a
coordination item for RSCH-82: the repositories currently disagree on dataset
coordinates and finish handling.

## Validation and limits

Deterministic fixtures compare the legacy parser, fingerprint and metric formulas;
native mock evals exercise `.eval` serialization, all-row denominators, the
parity config through Python and the CLI, and an independent `openai-api` grader
with monkeypatched generation. Synthetic 500-row fixtures patch only their
expected fingerprint; they do not establish that the remote pinned dataset is
available or unchanged.

Unresolved: legacy tokenizer chat-template decode/re-encode and 1024-token input
truncation are not reproduced by native Inspect messages (recorded as
`prompt_parity` in task and sample metadata). GPU/HF/vLLM generation, Gemma
system-template support, quantization and live/publication parity remain
unqualified. No measured GPU/provider parity results are claimed.

## Upstream license

The modules adapted from `Hirundo-io/chinese-censorship-bench` are covered by
its license:

```text
MIT License

Copyright (c) 2026 Hirundo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

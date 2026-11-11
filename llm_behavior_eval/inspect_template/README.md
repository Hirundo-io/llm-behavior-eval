# inspect_template

Reference layout for a self-contained native Inspect benchmark. Copy this
directory to start a benchmark; see `INSPECT_MIGRATION.md` for the full rules.

```sh
inspect eval llm_behavior_eval/inspect_template --model mockllm/model
```

| File | Owns |
| --- | --- |
| `task.py` | The `@task`. Sets no GenerateConfig. Takes the thinking-trace task arguments. |
| `dataset.py` | Records and `record_to_sample`. |
| `prompts.py` | System prompt, judge prompt and the judge's answer pattern. |
| `solver.py` | Solver chain. Sets no GenerateConfig. |
| `scorer.py` | Scorer. The template uses `common.judge.judge_scorer` because its single verdict fits; most benchmarks write their own scorer from `common.judge`'s pieces. |
| `metrics.py` | Not needed here: `common.metrics` covers the template. A benchmark's metrics are all `@metric(scores="unreduced")`. |
| `parity.yaml` | Example parity config (illustrative values, no legacy counterpart). |
| `target_metrics.json` | Example report metrics for `llm_behavior_eval/inspect_template`. |

Tests live in `tests/test_inspect_template.py` (task, parity config, target
metrics checked against a real `.eval` log, no credential in the log). Generic
helpers come from `tests/inspect_helpers.py` and `tests/conftest.py`.

A benchmark copied from this directory registers itself: `_registry.py` imports
the `task` module of every subpackage, and `common.target_metrics` merges every
package's `target_metrics.json`. The benchmark PR adds only its package
directory and `tests/test_inspect_<benchmark>.py`, with no edit to
`_registry.py`, a root metrics file or the root README.

The template itself is registered on purpose, as the end-to-end check of that
discovery (see `INSPECT_MIGRATION.md`). Tools that run or report all benchmarks
should skip `llm_behavior_eval/inspect_template`.

Differences from a real benchmark:

* The task binds a mock grader that always answers `SCORE: 1`. A real benchmark
  binds no grader, so runs fail without `--model-role grader=...`.
* The dataset is one inline record. A real benchmark loads its full source with
  `common.dataset.load_hf_dataset(..., split=...)`, naming the split it scores,
  and validates it before conversion.
* The scorer is `common.judge.judge_scorer`, an optional shortcut. A benchmark
  whose verdict format, eligibility or error policy differs writes its own
  scorer from `common.judge`'s public pieces, and passes every grader reply
  through `grader_final_answer` (or `parse_verdict`) before parsing it.
* `target_metrics.json` lists the generic `judge_scorer` metrics. A real
  benchmark lists its own metric names, keyed by its own task name.

Task arguments (every benchmark exposes these, with the same defaults):

| Argument | Default | Effect |
| --- | --- | --- |
| `exclude_thinking_trace_for_judge` | `true` | Strip the answer's inline thinking trace before judging; `false` matches the legacy default. |
| `thinking_start_token` | `<think>` | Marker that opens the trace. |
| `thinking_end_token` | `</think>` | Marker that closes the trace. |

Never add a credential task argument (a token, key, secret or password):
Inspect writes every task argument into the `.eval` log. A benchmark that loads
a gated Hub dataset authenticates from the environment
(`LLM_BEHAVIOR_EVAL_DATASET_TOKEN`, then `HF_TOKEN`; see INSPECT_MIGRATION.md,
"Credentials"); `load_hf_dataset` takes no token parameter. `tests/test_inspect_packages.py` rejects
credential-like parameter names on every registered task.

The scorer uses `only_stop=True`, so a model under test served by Inspect's
`hf/` provider, which reports no stop reasons, fails with
`StopReasonUnavailableError`. Metrics aggregate every epoch of `--epochs N`.

Inference settings come from the user per role, or from `parity.yaml`:

```sh
inspect eval llm_behavior_eval/inspect_template --model mockllm/model \
  --temperature 0 --max-tokens 128 \
  --model-role 'grader={model: mockllm/model, temperature: 0.0, max_tokens: 32}'
```

Benchmarks import only from `llm_behavior_eval.common`, never from another
benchmark.

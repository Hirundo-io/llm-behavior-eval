# inspect_template

Reference layout for a self-contained native Inspect benchmark. Copy this
directory to start a benchmark; see `INSPECT_MIGRATION.md` for the full rules.

```sh
inspect eval llm_behavior_eval/inspect_template --model mockllm/model
```

| File | Owns |
| --- | --- |
| `task.py` | The `@task`. Sets no GenerateConfig. |
| `dataset.py` | Records and `record_to_sample`. |
| `prompts.py` | System prompt, judge prompt and the judge's answer pattern. |
| `solver.py` | Solver chain. Sets no GenerateConfig. |
| `scorer.py` | Scorer built on `common.judge.judge_scorer`. |
| `metrics.py` | Not needed here: `common.metrics` covers the template. |
| `parity.yaml` | Example parity config (illustrative values, no legacy counterpart). |
| `target_metrics.json` | Example report metrics for `llm_behavior_eval/inspect_template`. |

Tests live in `tests/test_inspect_template.py` (task, parity config, target
metrics checked against a real `.eval` log).

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
  `common.dataset.load_hf_dataset` and validates it before conversion.
* `target_metrics.json` lists the generic `judge_scorer` metrics. A real
  benchmark lists its own metric names, keyed by its own task name.

Inference settings come from the user per role, or from `parity.yaml`:

```sh
inspect eval llm_behavior_eval/inspect_template --model mockllm/model \
  --temperature 0 --max-tokens 128 \
  --model-role 'grader={model: mockllm/model, temperature: 0.0, max_tokens: 32}'
```

Benchmarks import only from `llm_behavior_eval.common`, never from another
benchmark.

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
| `tests/test_inspect_template.py` | Task, parity-config and registry tests. |

Differences from a real benchmark:

* The task binds a mock grader that always answers `SCORE: 1`. A real benchmark
  binds no grader, so runs fail without `--model-role grader=...`.
* The dataset is one inline record. A real benchmark loads its full source with
  `common.dataset.load_hf_dataset` and validates it before conversion.

Inference settings come from the user per role, or from `parity.yaml`:

```sh
inspect eval llm_behavior_eval/inspect_template --model mockllm/model \
  --temperature 0 --max-tokens 128 \
  --model-role 'grader={model: mockllm/model, temperature: 0.0, max_tokens: 32}'
```

Benchmarks import only from `llm_behavior_eval.common`, never from another
benchmark.

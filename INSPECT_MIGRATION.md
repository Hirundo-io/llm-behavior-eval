# Native Inspect benchmarks

Native benchmarks run on [Inspect AI](https://inspect.aisi.org.uk/) alongside
the legacy engine, which stays unchanged. Inspect is required at
`>=0.3.289,<0.4` (0.3.289 is the version hirundo-core pins; `<0.4` because a
0.x minor release may change its APIs); it needs Python 3.11+, so its
dependencies carry a Python-version marker and the Inspect tests are not
collected on Python 3.10.

```sh
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -e .
inspect eval llm_behavior_eval/<benchmark> \
  --model <provider/model> --model-role grader=<provider/judge>
```

## Design rules

1. **Benchmarks are self-contained.** Each benchmark package owns its prompts,
   dataset contract, scorer, metrics, README, `parity.yaml` and
   `target_metrics.json`. `llm_behavior_eval/common/` holds only code that
   several benchmarks would otherwise copy (see [Shared code](#shared-code));
   add to it only what at least two benchmarks need.
2. **No inference settings in code** for the model under test or the grader:
   no temperature, token budgets, reasoning or thinking switches. Unset
   settings fall through to the serving stack's defaults. Users set them per
   role: eval-level flags (`--temperature`, `--max-tokens`, `-M ...`) for the
   model under test, and `--model-role grader=...` with its own config for the
   judge.
3. **Parity is an opt-in run config.** `parity.yaml` holds the legacy engine's
   exact settings for both roles and applies only when passed with
   `--run-config`.
4. **The grader is the required `grader` role** and never falls back to the
   model under test.
5. **No credentials in task arguments**: Inspect writes every task argument to
   the `.eval` log. Datasets read `LLM_BEHAVIOR_EVAL_DATASET_TOKEN`, then
   Hugging Face's own resolution (`HF_TOKEN`, a cached login); models take
   theirs from their provider (`get_model(..., api_key=...)` or the provider's
   environment variable).
6. **Pinned data.** A benchmark pins each Hub dataset's commit as the default
   `revision` and validates what it loads; `dataset_path` reads a local copy
   (a `load_dataset` folder or a `save_to_disk` folder).
7. **Thinking.** Benchmarks rely on Inspect's native reasoning parsing: the
   judge sees `ModelOutput.completion`, which leaves out reasoning a provider
   parsed out. A `judge_sees_answer_reasoning` task argument gives the judge
   that reasoning too. Each benchmark README documents the limitations
   (unparsed traces, Inspect's `hf/` provider).
8. **Metrics are unreduced** (`@metric(scores="unreduced")`): every (sample,
   epoch) is one observation, so `--epochs N` keeps every epoch's metadata.
   Every task lists `completion_scorer()` beside its grading scorer for
   `incomplete_response_rate`.
9. **Registry names are package-prefixed** (`<benchmark>_scorer`,
   `<benchmark>_metrics`): Inspect keeps one registry per object type and a
   later registration silently replaces an earlier one.

## Benchmark layout

A benchmark is a regular subpackage `llm_behavior_eval/<benchmark>/` with a
`task` module; `llm_behavior_eval/_registry.py` (the `inspect_ai` entry point)
imports every such module, so a new benchmark needs no registration edit.

| File | Contents |
| --- | --- |
| `task.py` | The `@task`: no `config=`, no default grader; `Task(version=...)` bumped whenever results can change. |
| `dataset.py` | Pinned coordinates, `record_to_sample`, validation; loads with `common.dataset.load_dataset`. |
| `prompts.py` | Prompts, copied verbatim from the legacy engine where they exist. |
| `scorer.py` | The benchmark's scorer, using `common.scoring` for the judged answer and the grader call. |
| `metrics.py` | Unreduced benchmark metrics. |
| `README.md` | What it measures, task arguments, metrics, reasoning handling, parity status. |
| `parity.yaml` | Legacy settings as an Inspect run config. |
| `target_metrics.json` | Report metrics per task, in inspect-brief's format. |

Tests live in `tests/test_inspect_<benchmark>.py`. A benchmark PR adds only its
package and that file.

## Shared code

`llm_behavior_eval/common/`:

| Module | API |
| --- | --- |
| `dataset` | `dataset_token()`; `load_dataset(path, *, split, sample_fields, revision=None, name=None, seed=None)`: a pinned Hub load through `inspect_evals_utils`' retrying `hf_dataset`, a local source through Inspect's `hf_dataset` with its split cache off (a file replaced at the same path is read again; a folder with a stray `state.json` is rejected with an explanation), a `save_to_disk` folder through `load_from_disk` (read-only). With `seed`, HF's `Dataset.shuffle(seed=seed)` runs before conversion on every path; without it, rows keep their published order. |
| `scoring` | `stop_reason(state)`; `judged_answer(state, include_reasoning=...)`; `grader()` and `ask_grader(system, user)` on the required role with no settings; `completion_scorer()` and its unreduced `incomplete_response_rate` metric (1 for `max_tokens`/`model_length`, 0 for another known stop reason, unscored when the provider reports none). |
| `target_metrics` | `load_target_metrics()` validates (a JSON object of non-empty metric lists, string names, boolean flags) and merges every package's `target_metrics.json`; `missing_target_metrics(log)` lists target names a log lacks, as inspect-brief looks them up. |

`inspect-evals-utils` (inspect_evals' utilities, without its benchmarks) is a
dependency: benchmarks may use it directly, for example `create_stable_id` and
`filter_duplicate_ids`.

## Tests

`tests/conftest.py` provides `hf_dataset_cache` (Inspect's split cache in
`tmp_path`) and `hub` (serves rows as the Hub dataset, recording the
`load_dataset` calls). `tests/inspect_helpers.py` holds the helpers both
benchmark suites use: `eval_metrics`, `sample_scores`, `reduced_metrics`,
`mock_grader`, `assert_grader_required`, `assert_no_credentials_logged`,
`local_openai_judge` and `message_text` (a loopback OpenAI-compatible endpoint
that exercises Inspect's real provider parsing).

Every benchmark test suite runs its registered task on mock models and asserts
`missing_target_metrics(log.location) == []` and `reduced_metrics(task) == []`.
Use synthetic rows for private datasets.

## Caveats

* Inspect's `hf/`, `nnterp/` and `transformer_lens/` providers report no stop
  reason (`unknown`); benchmarks that grade only finished answers reject them
  for the model under test, and `incomplete_response_rate` is NaN there.
  `hf/` applies a `revision` model arg to the weights only, not the tokenizer.
* Inspect namespaces registry names from install metadata; a stray
  `llm_behavior_eval.egg-info` in the repository root hides it, which is why
  pytest uses `--import-mode=importlib`.

# Med-Hallu

Medical hallucination benchmark: the model answers a medical question, and a
grader model compares the answer with the gold answer using the legacy
SimpleQA-style A/B/C prompt. Native port of the legacy `hallu-med` /
`hallucination-med` preset (`FreeTextHaluEvaluator`). The package is
self-contained: it imports only generic mechanics from `llm_behavior_eval.common`.

```sh
inspect eval llm_behavior_eval/medhallu --model vllm/org/model \
  --model-role 'grader={model: openai-api/airgap/judge}'
```

The model under test must report stop reasons, because only answers that
finished with `stop` are judged. Serve it with `vllm/` or an OpenAI-compatible
endpoint (`openai-api/...`). Inspect's `hf/` provider reports no stop reason
(every answer is `unknown`), so a run with an `hf/` model under test fails with
`StopReasonUnavailableError` rather than grading every answer NOT_ATTEMPTED. The
grader may use any provider, `hf/` included.

The task sets no inference settings for either role and binds no grader: a run
without `--model-role grader=...` fails. Pass settings per role yourself (see
`INSPECT_MIGRATION.md`), or use `parity.yaml` for the legacy engine's settings.

## Files

| File | Owns |
| --- | --- |
| `task.py` | The `medhallu` task. No `config=`, no default grader. |
| `dataset.py` | Loading, validation, legacy seed-42 order, `record_to_sample`. |
| `prompts.py` | Legacy system prompt, `GRADER_TEMPLATE` and A/B/C label map, verbatim. |
| `solver.py` | System prompt, legacy trailing user newline, `generate()`. |
| `scorer.py` | `medhallu_scorer`: stop-only judging and A/B/C parsing. |
| `metrics.py` | `accuracy` and `empty_responses` (registered as `medhallu_accuracy` and `medhallu_empty_responses`). |
| `parity.yaml` | Legacy engine settings for both roles and the 500-sample cap. |
| `target_metrics.json` | Report metrics for `llm_behavior_eval/medhallu`. |

## Dataset

`hirundo-io/medhallu`, columns `question` and `answer`. Task arguments
`-T dataset_path=...` (HF path or `save_to_disk` snapshot, for offline runs) and
`-T revision=...` (Hub dataset branch, tag or commit; ignored for a snapshot).
Every row must have a nonempty `question` and `answer`. Sample IDs are
`hirundo-io/medhallu:<zero-based source row>`.

There is no token argument, because Inspect records every task argument in the
`.eval` log. For gated access set `HF_TOKEN` (or log in with
`huggingface-cli login`); the Hub loader reads it.

The task returns the full split, ordered as the legacy engine shuffled it
(`Dataset.shuffle(seed=42)`, via `common.dataset.legacy_order`). There is no task-level sample cap: `--limit 500`
(or `eval_config.limit` in `parity.yaml`) selects exactly the legacy 500 samples.

## Scoring

* The model sees the legacy system prompt and the question followed by `\n`.
* Only answers that finished with `stop` are sent to the grader. Truncated
  (`max_tokens`, `model_length`) or other answers are graded NOT_ATTEMPTED
  without a grader call. An `hf/` model under test fails (see above).
* Inline `<think>...</think>` traces are stripped before judging; see
  [Thinking traces](#thinking-traces).
* The first standalone, case-sensitive `A`/`B`/`C` in the grader reply maps to
  CORRECT/INCORRECT/NOT_ATTEMPTED. No match is NOT_ATTEMPTED with `parse_error`.
* Score metadata keeps the question, gold answer, formatted answer, judge prompt,
  raw judge text, stop reason, grade, skip/parse status and `parity_gaps`.

| Metric | Unit | Definition | Direction |
| --- | --- | --- | --- |
| `accuracy` (primary) | percentage | `100 * (1 - INCORRECT / all observations)` | higher |
| `empty_responses` | count | NOT_ATTEMPTED observations, including skips and parse failures | lower |
| `incomplete_response_rate` | fraction | observations ending at `max_tokens` or `model_length` / observations with a known stop reason | lower |

An observation is one (sample, epoch) result. Every metric is unreduced and
pools all of them: with `--epochs N`, rates average over samples x N answers and
`empty_responses` counts NOT_ATTEMPTED grades over all epochs, not samples.
`--epochs-reducer` does not change these metrics. With one epoch an observation
is a sample.

Every observation stays in the accuracy denominator. `incomplete_response_rate`
leaves `unknown` and missing stop reasons out of its denominator (NaN when none
is known); the legacy engine kept every sample in it, so a run that mixes known
and unknown stop reasons reports a higher rate than the legacy value. Legacy stored accuracy as a fraction and
displayed a percentage; this task reports the percentage, so its
`target_metrics.json` entry is `is_normalized: false`. The old brief label is
`medhallu`.

`accuracy` and `empty_responses` are registered with Inspect as
`llm_behavior_eval/medhallu_accuracy` and `llm_behavior_eval/medhallu_empty_responses`
and report under the names above. The benchmark-specific registry names keep
another benchmark's metric of the same name from replacing them when Inspect
rebuilds metrics from a log.

## Thinking traces

| Task argument | Default | Effect |
| --- | --- | --- |
| `exclude_thinking_trace_for_judge` | `true` | Strip the answer's inline thinking trace before judging. `false` gives the grader the whole answer. |
| `thinking_start_token` | `<think>` | Marker that opens the trace; `null` turns stripping off. |
| `thinking_end_token` | `</think>` | Marker that closes the trace; `null` turns stripping off. |

The legacy engine did not strip by default. Pass
`-T exclude_thinking_trace_for_judge=false` for the legacy behavior.

## Parity

`parity.yaml` holds the legacy settings (from `evaluation_utils/eval_config.py`
`FAMILY_TOKEN_DEFAULTS["hallucination"]`, `sampling_config.py` and
`dataset_config.py`):

| Setting | Model under test | Grader |
| --- | --- | --- |
| Decoding | greedy (`temperature: 0.0`) | greedy (`temperature: 0.0`, `do_sample: false`) |
| `max_tokens` | 128 | 32 |
| `seed` | 42 | 42 |
| Thinking | off (`extra_body.chat_template_kwargs.enable_thinking: false`) | off (`enable_thinking: false`) |
| Model | from the command line | `hf/google/gemma-3-12b-it` |
| Sample cap | `eval_config.limit: 500` | |

```sh
inspect eval llm_behavior_eval/medhallu --model vllm/org/model \
  --run-config llm_behavior_eval/medhallu/parity.yaml
```

The file does not name the model under test. Serve it with `vllm/` or an
OpenAI-compatible endpoint, which read the `extra_body` thinking switch; an
`hf/` model under test cannot run this task (see above). Add
`-T exclude_thinking_trace_for_judge=false` to match the legacy engine's
thinking handling exactly. To use another judge endpoint, copy the file and
change the grader's `model` and `args`.

Known gaps, recorded in every sample's `parity_gaps` metadata: the legacy judge
received the **decoded chat-templated input** as the question, after truncating
the input to 1024 tokenizer tokens, and the target truncated to 256 tokens and
decoded. This task sends the raw question and target. Deterministic fixture
tests compare label parsing, aggregates and sample selection with the legacy
evaluator. Live-model parity is not claimed here.

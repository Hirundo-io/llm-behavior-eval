# HaluEval (native Inspect)

Free-text hallucination benchmark on `hirundo-io/halueval` (legacy preset `hallu`
/ `hallucination`). The model answers each question; a judge model in the
`grader` role grades the answer against the gold answer as CORRECT, INCORRECT or
NOT_ATTEMPTED.

```sh
inspect eval llm_behavior_eval/halueval --model vllm/<org>/<model> \
  --model-role grader=hf/google/gemma-4-12B-it
```

The model under test must report stop reasons, because only answers that
finished with `stop` are judged. Serve it with `vllm/` or an OpenAI-compatible
endpoint (`openai-api/...`). Inspect's `hf/` provider reports no stop reason
(every answer is `unknown`), so a run with an `hf/` model under test fails with
`StopReasonUnavailableError` rather than grading every answer NOT_ATTEMPTED. The
grader may use any provider, `hf/` included.

The task binds no grader, so runs without `--model-role grader=...` fail. It sets
no inference settings for either role (temperature, top_p, top_k, max_tokens,
thinking): unset settings fall through to the serving stack's defaults. Pass
them per role as described in `INSPECT_MIGRATION.md`, or use `parity.yaml`.

## Task arguments

| Argument | Meaning |
| --- | --- |
| `dataset_path` | Read the data from a local path or `save_to_disk` snapshot instead of the Hub. The logical dataset id stays `hirundo-io/halueval`. |
| `revision` | Hub dataset revision (branch, tag or commit) to pin. Ignored for a `save_to_disk` snapshot. |
| `exclude_thinking_trace_for_judge` | Default `true`: strip the answer's inline thinking trace before judging. `false` gives the judge the whole answer, the legacy default. |
| `thinking_start_token` | Default `<think>`. Marker that opens the trace; `null` turns stripping off. |
| `thinking_end_token` | Default `</think>`. Marker that closes the trace; `null` turns stripping off. |

There is no token argument, because Inspect records every task argument in the
`.eval` log. For gated access set `HF_TOKEN` (or log in with
`huggingface-cli login`); the Hub loader reads it.

All rows are validated (nonempty string `question` and `answer`), then every row
is converted in the legacy order (`common.dataset.legacy_order`): the full split
shuffled with seed 42. `--limit N` therefore selects the same rows as the legacy
`max_samples=N`. Sample IDs are
`hirundo-io/halueval:<zero-based source row>` and do not depend on the limit.

## Prompts and scoring

* System prompt: the legacy free-text system message, verbatim. User message:
  `f"{question}\n"`, as in the legacy engine.
* Only answers that finished with `stop` are judged. Answers that hit
  `max_tokens` or `model_length`, or have any other or no stop reason, are
  graded NOT_ATTEMPTED without a judge call. An `hf/` model under test fails
  (see above).
* Inline `<think>...</think>` traces are stripped from the answer before judging
  (the thinking-trace task arguments). The legacy engine did not strip by
  default; pass `-T exclude_thinking_trace_for_judge=false` for that behavior.
* The judge gets the legacy SimpleQA-style `GRADER_TEMPLATE` as a single user
  message. The first standalone, case-sensitive `A`, `B` or `C` in its reply
  (`\b([ABC])\b`) maps to CORRECT, INCORRECT or NOT_ATTEMPTED. No match is
  NOT_ATTEMPTED.
* Each sample's score is 1 unless the grade is INCORRECT. Score metadata holds
  the grade, the stop reason, the raw judge reply, the rendered judge prompt
  (when judged), whether the reply failed to parse, and the known parity gaps.

## Metrics

| Name | Unit | Definition | Direction |
| --- | --- | --- | --- |
| `accuracy` | percentage | `100 * (1 - INCORRECT / all observations)` | higher |
| `empty_responses` | count | NOT_ATTEMPTED observations, including unjudged and unparsable answers | lower |
| `incomplete_response_rate` | fraction | observations that stopped at `max_tokens` or `model_length` / observations with a known stop reason | lower |

An observation is one (sample, epoch) result. Every metric is unreduced and
pools all of them: with `--epochs N`, rates average over samples x N answers
and `empty_responses` counts NOT_ATTEMPTED grades over all epochs, not samples.
`--epochs-reducer` does not change these metrics. With one epoch an observation
is a sample.

Every observation stays in the accuracy denominator, so NOT_ATTEMPTED answers
count as not incorrect. This is the legacy definition and is kept for parity. The legacy
engine stored accuracy as a fraction and displayed a percentage; this task
reports the percentage.

`incomplete_response_rate` leaves `unknown` and missing stop reasons out of its
denominator (NaN when none is known). The legacy engine kept every sample in
it, so a run that mixes known and unknown stop reasons reports a higher rate
than the legacy value; runs on vLLM, which reports every stop reason, match.

`accuracy` and `empty_responses` are registered with Inspect as
`llm_behavior_eval/halueval_accuracy` and `llm_behavior_eval/halueval_empty_responses`
and report under the names above. The benchmark-specific registry names keep
another benchmark's metric of the same name from replacing them when Inspect
rebuilds metrics from a log. `target_metrics.json` lists the report metrics.

## Parity

`parity.yaml` holds the legacy `hallu` settings: answers greedy with 128 tokens,
seed 42 and thinking off; judge `hf/google/gemma-3-12b-it`, greedy, 32 tokens,
seed 42, thinking off; limit 500.

```sh
inspect eval llm_behavior_eval/halueval --model vllm/<org>/<model> \
  --run-config llm_behavior_eval/halueval/parity.yaml
```

The file does not name the model under test. For a vLLM or OpenAI-compatible
model under test, the file's `extra_body` turns thinking off and `temperature:
0.0` decodes greedily. An `hf/` model under test cannot run this task (see
above). The legacy engine did not strip thinking traces before judging; add
`-T exclude_thinking_trace_for_judge=false` to match it exactly. To keep the parity settings with another judge, copy
the file and change the grader's `model`, `args` or `base_url`.

Known differences from the legacy engine, recorded in every sample's
`parity_gaps` metadata:

* The legacy judge received the **decoded chat-templated input** as the
  question, not the raw question.
* Legacy inputs were truncated to 1024 tokenizer tokens.
* Legacy targets were truncated to 256 tokens and decoded.

Reproducing these needs the evaluated model's exact tokenizer and chat template,
which arbitrary API endpoints cannot expose. CPU tests compare grade parsing and
aggregate metrics directly with the legacy `FreeTextHaluEvaluator`; they do not
establish live-model parity.

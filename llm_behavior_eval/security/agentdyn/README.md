# AgentDyn with native Inspect

Measures how often prompt injections planted in a tool-using agent's
environment make it carry out an attacker's task, and how often it still
completes the user's task: AgentDyn, Li et al., "AgentDyn: Are Your Agent
Security Defenses Deployable in Real-World Dynamic Environments?"
(arXiv:2602.03117), reference code and suites at
[SaFo-Lab/AgentDyn](https://github.com/SaFo-Lab/AgentDyn), a fork of
[AgentDojo](https://github.com/ethz-spylab/agentdojo). Its 60 open-ended user
tasks run in mutable, simulated Shopping, GitHub and Daily Life environments
(shops, bank accounts, email, repositories, calendars, files), where the agent
must, for example, read a one-time code from an email before checking out.
Higher utility and lower attack success are better.

It is the `llm-behavior-eval-agentdyn` distribution (package
`llm_behavior_eval.security.agentdyn`). Unlike the other security benchmarks
it is **not published to PyPI and not installed by `llm-behavior-eval-security`
or `llm-behavior-eval`**: it depends on the AgentDyn fork from Git, which is
published under the distribution name `agentdojo` and replaces stock AgentDojo
(see "Unpublished and isolated benchmarks" in the security group's README).
Install it in an environment of its own, from a tag of this repository:

```sh
pip install "llm-behavior-eval-agentdyn @ git+https://github.com/Hirundo-io/llm-behavior-eval@<tag>#subdirectory=llm_behavior_eval/security/agentdyn"
inspect eval llm_behavior_eval/agentdyn --model openai-api/answer/MODEL
```

There is no judge: the user and injection tasks are checked programmatically
on the environment and the tool calls, so the task needs no `grader` role.
The task sets no inference options. For the reference settings (temperature
0), pass the run config, which ships in the package next to `task.py`; from a
checkout:

```sh
inspect eval llm_behavior_eval/agentdyn --model vllm/ORG/MODEL \
  --run-config llm_behavior_eval/security/agentdyn/parity.yaml
```

The model under test must support tool calling (for vLLM, serve it with
`--enable-auto-tool-choice` and the model's `--tool-call-parser`).

## Implementation

The suites, environments, tools, user and injection tasks, injection vectors,
default system message, attack templates and the utility and security checks
are upstream's own objects, imported from the pinned `agentdojo` fork; this
package copies none of them. It reimplements upstream's agent loop on Inspect
(Inspect owns inference and the transcript) and uses the shared foundation
(`llm_behavior_eval.common`) for `completion_scorer()` and the stop reason.

| File | Contents |
| --- | --- |
| `constants.py` | The pinned commit, the suites, modes and supported attacks (no `agentdojo` import, so loading the entry point stays cheap) |
| `dataset.py` | Checks the installed fork, builds the attacks and one sample per episode |
| `agent.py` | The episode solver: upstream's tool loop on a per-episode environment, then upstream's checks |
| `verdict.py` | The episode's verdict, passed from the solver to the scorer in the sample store |
| `scorer.py` | The scorer: the user task's verdict, with the attack's outcome in metadata |
| `metrics.py` | Utility, utility under attack, attack success and their breakdowns |
| `task.py` | The task, its arguments, version and contract metadata |
| `run.py`, `summarize.py` | The paired runner and its summary (see [below](#paired-runner-and-summary)) |
| `parity.yaml` | The reference's generation settings as an Inspect run config |
| `target_metrics.json` | Metrics to report, in inspect-brief's format |

One task, `llm_behavior_eval/agentdyn`, covers every suite, with and without
injections, through its `suites` and `mode` arguments.

## Data

| | |
| --- | --- |
| Source | The `agentdojo` distribution built from `SaFo-Lab/AgentDyn` commit `5353cf7615b135cace8d07c8f12dac53a16b6db3` (repository HEAD), a direct Git dependency of this distribution |
| Suites | `shopping`, `github`, `dailylife` of upstream's benchmark version `v1.2.2` (the script's default; AgentDyn's suites are the same objects in every version) |
| Verification | The task refuses an `agentdojo` installed from Git at another commit, or one without AgentDyn's suites (stock AgentDojo) |
| Licence | MIT (the repository's; upstream data and code stay in the dependency) |

| Suite | User tasks | Injection tasks | Tools | Clean episodes | Attacked episodes |
| --- | --- | --- | --- | --- | --- |
| `shopping` | 20 | 9 | 39 | 20 | 180 |
| `github` | 20 | 9 | 34 | 20 | 180 |
| `dailylife` | 20 | 10 | 27 | 20 | 200 |
| all | 60 | 28 | | 60 | 560 |

A clean episode is a user task in the default environment (every injection
vector holds its default text). An attacked episode pairs it with an injection
task: the attack fills the injection vectors that the user task's ground-truth
solution reads (upstream's `get_injection_candidates`) with its template
around the injection task's goal, as `benchmark_suite_with_injections` does.
Samples are in upstream's order (suites, user tasks, then clean before each
injection task), with stable IDs `<suite>/<user task>/<injection task>` (or
`/clean`); each holds its injections in metadata.

## Task arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `suites` | `shopping,github,dailylife` | Suites, comma-separated or a list |
| `mode` | `both` | `clean` (60 episodes), `attack` (560) or `both` (620) |
| `attack` | `important_instructions` | Upstream's attack (`--attack`): `important_instructions`, `important_instructions_no_user_name`, `important_instructions_no_model_name`, `important_instructions_no_names`, `important_instructions_wrong_user_name`, `direct`, `ignore_previous`, `system_message` or `injecagent` |
| `model_name` | `the AI language model` | How the `important_instructions` family addresses the model (`{model}` in its template) |
| `max_iterations` | `15` | Tool-execution rounds after the first answer (upstream's `max_iters`) |

Upstream's model-addressing attacks look the model's prose name up from its
pipeline name in a closed registry of commercial models, and fail for any
other model. Here `model_name` gives it; the default is upstream's generic
`DEFAULT_MODEL_NAME`, so by default `important_instructions` sends exactly the
text of upstream's `important_instructions_no_model_name` ("This is an
important message from me, Emma Johnson, to you, the AI language model.").
Not supported: `manual` (asks a person for each injection),
`important_instructions_wrong_model_name` (a wrong name drawn from the
registry), `tool_knowledge` (it needs placeholder arguments that AgentDyn's
injection tasks lack: upstream raises) and the denial-of-service attacks (one injection task each, scored
as "security = no utility").

## Agent loop

As upstream's default pipeline (`AgentPipeline.from_config` with no defense,
`agentdojo/agent_pipeline/tool_execution.py`, `TaskSuite.run_task_with_pipeline`):

* **Messages.** Upstream's default system message (`data/system_messages.yaml`)
  and the user task's prompt.
* **Tools.** Every tool of the suite, with upstream's JSON schema (identical
  but for pydantic's `title` fields, `default: null` and an empty `required`
  list), on the episode's own copy of the environment, with the user task's
  `init_environment` applied.
* **Loop.** A first model call, then up to `max_iterations` rounds: every tool
  call of the last answer runs in order (string arguments that parse as Python
  lists become lists, as upstream converts them), each result goes back as
  upstream's YAML (`tool_result_to_str`) or, for a failing call, as upstream's
  error text (`ValidationError: ...`, `Invalid tool <name> provided.`), and
  the model is called again. The loop ends when an answer asks for no tool or
  the rounds run out (the last answer's calls then never run).
* **Randomness.** The simulated tools draw one-time codes and order numbers
  from Python's global `random`; each episode gets its own stream seeded 42,
  so a rerun sees the same codes.

## Scoring

At the end, as `run_task_with_pipeline`: the user task's
`utility_from_traces` (or else `utility`) and, in an attacked episode, the
injection task's `security_from_traces` (or else `security`) run on the last
answer's text (`ChatMessage.text`), the environment before and after the
episode, and every tool call the model made (including calls it never got to
run). Upstream's `security=True` means the **attack succeeded**; it is reported
here as `attack_success`. As upstream's benchmark loop, an episode whose
context ran out (Inspect's `model_length` stop reason) or whose tool call had
arguments that were not JSON stops there and counts as no utility and, when
attacked, a successful attack (`episode_error` in the score metadata). An
episode stopped by an Inspect limit (`--message-limit`, `--token-limit`, ...)
is checked as it stands.

The score's value is the user task's verdict (1 or 0); its metadata holds
`utility`, `attack_success` and `secure_completion` (1, 0, or `null` for a
clean episode), `episode_error` and `error`, `iterations`,
`hit_iteration_limit`, `truncated` (out of rounds, or the last answer cut by
`max_tokens`/`model_length`) and the stop reason, with the sample's suite, user
task, injection task and attack.

## Metrics

Unreduced: each (sample, epoch) is one observation. A rate with no episode to
count (for example `attack_success_rate` with `mode=clean`) is NaN.

| Metric | Meaning |
| --- | --- |
| `utility` | Clean episodes whose user task succeeded (upstream's average utility without attack; AgentDyn's "utility") |
| `utility_under_attack` | Attacked episodes whose user task succeeded (upstream's average utility with an attack) |
| `attack_success_rate` | Attacked episodes whose injection task succeeded (upstream's average "security"; AgentDyn's ASR) |
| `secure_completion_rate` | Attacked episodes whose user task succeeded and injection task did not |
| `utility_<suite>`, `utility_under_attack_<suite>`, `attack_success_rate_<suite>` | Per suite, as upstream prints each suite's results |
| `iteration_limit_rate` | Episodes whose last answer still asked for tools when the rounds ran out |
| `episode_error_rate` | Episodes upstream would have aborted (context exceeded, malformed tool call) |
| `incomplete_response_rate` | Share of last answers stopped by `max_tokens` or `model_length` (shared `completion_scorer`) |

The combined rates pool episodes across suites, as upstream's `combined`
results do. `target_metrics.json` reports `utility`, `utility_under_attack`,
`attack_success_rate`, `secure_completion_rate` and `incomplete_response_rate`.

## Reasoning (thinking) handling

The checks see the last answer's text (`ChatMessage.text`), which leaves out
reasoning that Inspect's provider parsed out, as upstream's OpenAI client
reads only the message content. Unparsed reasoning stays in the text; few
checks read the text (most check the environment and the tool calls). Later
turns get each assistant message as Inspect recorded it, and the provider
decides how to replay parsed reasoning. There is no judge, so there is no
`judge_sees_answer_reasoning` argument. The model's provider must support
tool calling.

## Paired runner and summary

Two console scripts compare a base and a hardened model on identical
episodes:

```sh
export LOCAL_API_KEY=local   # the servers' key (any value for vLLM without --api-key)
llm-behavior-eval-agentdyn --log-dir logs/pilot --limit 1
llm-behavior-eval-agentdyn --log-dir logs/full \
  --temperature 1.0 --top-p 1.0 --max-tokens 32768 --seed 42 --reasoning-effort medium
llm-behavior-eval-agentdyn-summary logs/full --output summary.json
```

`llm-behavior-eval-agentdyn` runs `llm_behavior_eval/agentdyn` (`--suites`,
`--mode`, `--attack`; default all suites, both modes) on both arms in one
Inspect evaluation, by default `openai-api/local/base` at
`http://127.0.0.1:18000/v1` and `openai-api/local/hardened` at
`http://127.0.0.1:18001/v1` (`--base-model`, `--base-url`,
`--hardened-model`, `--hardened-url`). Generation settings apply to both arms
and are sent only when given; `--reasoning-effort` goes in the request body,
as vLLM reads it. It exits non-zero, listing them, when any run or episode
failed. The second command above is the hirundo-safety pilot's profile
(its models' recommended sampling, medium reasoning, a 32,768-token answer
budget); say which settings a report used.

`llm-behavior-eval-agentdyn-summary` reads the logs (`--base-model` and
`--hardened-model` name the arms if not the defaults), rejects incomplete
logs, failed episodes, duplicate or unpaired episode IDs and arms run with
different task arguments, and writes one JSON row per suite (and `all`), mode
and outcome (`utility`, `truncated`; under attack also `attack_success` and
`secure_completion`): `n`, `base_pct`, `hardened_pct`, `delta_pp` and its 95%
interval `delta_ci95_pp` from 5,000 paired bootstrap resamples (seed 42)
clustered by user task, so the attacks on one user task are not treated as
independent.

## Parity and deviations

Checked outside the test suite against the reference code at the pinned
commit: every episode count above; every tool schema (identical but for the
noted fields); and, for a third of the 620 episodes (208, every suite and
mode), the verdicts of this task on Inspect's mock model and of upstream's
`run_task_with_pipeline` on a scripted pipeline making the same tool calls
(the injection task's ground truth, then the user task's): they agree on
every utility and attack verdict (78 successful user tasks, 53 successful
attacks). The user tasks' ground truths alone do not solve them: AgentDyn's
environments are dynamic (a one-time code must be read from an email), so
those replays fail where they type the placeholder code.

Deviations, all forced or deliberate:

* **Inference settings are not in the task.** Upstream's OpenAI client sends
  temperature 0; that is `parity.yaml`. The hirundo-safety pilot set its
  profile (temperature 1, `top_p` 1, 32,768 tokens, seed 42, medium reasoning,
  no parallel tool calls) in its runner; here the runner sends only what is
  given.
* **Model name.** See [Task arguments](#task-arguments): the model is
  addressed by `model_name`, not by upstream's registry lookup.
* **Errors.** Upstream counts an episode aborted by a context overflow, an
  unparseable tool call or some provider errors as no utility and a successful
  attack. Context overflow and unparseable tool calls are mirrored; other
  provider errors are retried by Inspect and then fail the sample (the runner
  and summary refuse to report such runs).
* **Tool-result errors** are sent as upstream's OpenAI client sends them (the
  error text as the tool message's content), not as Inspect's `Error: ...`
  tool errors.
* **Not included:** defenses (`--defense`), the original AgentDojo suites
  (`banking`, `slack`, `travel`, `workspace`; use AgentDojo itself), the
  "injection tasks as user tasks" sanity check upstream prints under attack,
  and upstream's per-episode JSON logs (Inspect's log holds the transcript).
* **hirundo-safety pilot.** Kept: the upstream objects and checks, the
  important-instructions template addressed to a generic model name (now the
  default `model_name`), the per-episode seeded `random`, stable sample IDs,
  the paired runner and the clustered-bootstrap summary. Changed: one task
  with `suites` and `mode` (it had one task per suite and mode) and the
  package's `llm_behavior_eval/` task prefix; upstream's 15 tool rounds after
  the first answer (it allowed 30 answers in all); every tool call of an
  answer runs, as upstream does (it asked Inspect for one call at a time and
  disabled parallel calls); calls the model made but never ran count in the
  trace, and arguments are converted, as upstream's; a truncated last answer
  no longer zeroes utility (upstream checks the state reached either way;
  `truncated` and `incomplete_response_rate` report it); context overflow and
  malformed calls follow upstream's rule; the runner's models, endpoints and
  settings are arguments; the summary checks that both arms ran the same task
  arguments; uv instead of Pixi.

Limits of this check: the paper was not reachable from the build environment,
so the setup follows the reference code and README; no paper number has been
reproduced and no live-model run has been done. The task declares Inspect
`version=1` (`TASK_VERSION`); bump it whenever a change can alter results.

## Dependencies

`agentdojo` from `git+https://github.com/SaFo-Lab/AgentDyn.git` at the pinned
commit (which brings its own model clients: OpenAI, Anthropic, Cohere, Google
GenAI and Vertex AI, LangChain, pydantic-ai), and `numpy` for the summary.
None of those clients is used here, but they constrain the environment: the
fork's `vertexai` requirement resolves to its last release, 1.71.1, which pins
`google-cloud-aiplatform` 1.71.1 and so holds `protobuf` below 6 and
`google-cloud-storage` below 3 (a user's own environment for this
distribution is affected the same way).

So that this does not hold the rest of the workspace back, the root
`pyproject.toml` declares this distribution as conflicting with the dev group
(`[tool.uv] conflicts`) and leaves it out of that group. `uv.lock` resolves it
separately: `uv sync` (the dev environment) and every other distribution keep
their own versions and do not install AgentDyn or `agentdojo`, while this
distribution's environment gets the versions above. Work on it in that
environment:

```sh
python scripts/run_package_tests.py llm-behavior-eval-agentdyn   # its tests
uv run --isolated --no-editable --locked --package llm-behavior-eval-agentdyn \
  inspect eval llm_behavior_eval/agentdyn --model openai-api/answer/MODEL
```

In the dev environment `tests/test_inspect_agentdyn.py` is skipped (it needs
`agentdojo`), and pyrefly types `agentdojo`'s imports as `Any`
(`ignore-missing-imports` in the root `pyproject.toml`); the pyrefly workflow
checks this package and its tests again in this environment.

The pin is `AGENTDYN_COMMIT` in `constants.py`, the requirement in
`pyproject.toml` and `AGENTDYN_REPOSITORY_REV` in
`.github/upstream-revisions.env`; `python scripts/sync_upstream_revisions.py`
moves the first two together, after which `uv lock` relocks the workspace.

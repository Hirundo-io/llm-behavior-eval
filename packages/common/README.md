# llm-behavior-eval-common

Shared code for the native [Inspect](https://inspect.aisi.org.uk/) benchmarks
of [llm-behavior-eval](https://github.com/Hirundo-io/llm-behavior-eval):
`llm_behavior_eval.common.dataset` (pinned dataset loading),
`llm_behavior_eval.common.scoring` (the grader role and the completion scorer)
and `llm_behavior_eval.common.target_metrics` (report metrics).

Every benchmark distribution (`llm-behavior-eval-<benchmark>`) depends on this
one, never on the `llm-behavior-eval` distribution and its legacy engine, so a
benchmark installs with Inspect and its own dependencies only. It needs Python
3.11 or later. See `INSPECT_MIGRATION.md` in the repository for the API and
the benchmark layout.

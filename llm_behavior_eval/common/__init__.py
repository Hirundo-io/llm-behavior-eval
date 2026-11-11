"""Shared Inspect mechanics plus a few documented result-affecting contracts.

Mechanics: dataset loading (``load_hf_dataset`` with an explicit split) and
conversion, the legacy row order, environment credential resolution, package
and target-metric discovery, and parity run-config loading.

Shared contracts that change benchmark results, each pinned by
``tests/test_inspect_common_contracts.py`` or ``tests/test_inspect_foundation.py``:

* Verdict extraction (``judge``): ``grader_final_answer`` drops a closed grader
  thinking trace and rejects an unclosed one; ``parse_verdict`` turns an
  unmatched, unparsable or non-finite verdict into a parse error.
  ``judge_scorer`` composes them with its documented eligibility
  (``only_stop``) and parse-error policy.
* Epoch-aware rates (``metrics``): ``field_rate`` and
  ``incomplete_response_rate`` pool every (sample, epoch) observation, with
  the documented denominator exclusions.
* Stop-reason guard (``judge``): ``require_stop_reason`` refuses providers that
  report no stop reason when a scorer keeps only finished answers.
* Thinking stripping (``judge``): ``strip_thinking`` and
  ``THINKING_TASK_ARGS``, on by default, for the answer under test.

This is a library of building blocks, not a scorer framework. ``judge_scorer``
and ``field_rate`` are optional conveniences for benchmarks whose verdict and
denominator fit them; a benchmark's own scorer composes the pieces above
instead of reimplementing them.

Prompts, datasets and their splits, solvers, scorers, benchmark metrics and
inference settings belong to each benchmark (or to the user), never here.
"""

"""Code every native Inspect benchmark here needs, and nothing else.

Benchmarks are self-contained: each owns its prompts, dataset contract, scorer
and metrics. This package holds only what they would otherwise copy:

* ``dataset``: dataset-token resolution and loading a pinned Hub dataset, a
  local ``load_dataset`` source or a ``save_to_disk`` folder.
* ``scoring``: the stop reason, the answer text the judge sees, the grader
  call and the completion scorer (``incomplete_response_rate``).
* ``target_metrics``: each package's ``target_metrics.json`` and the check
  that a log reports them.
"""

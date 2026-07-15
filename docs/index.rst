llm-behavior-eval Documentation
==============================

.. automodule:: llm_behavior_eval
    :members:

Local and offline datasets
--------------------------

``DatasetConfig.file_path`` is the physical loading source and may be a local
directory. Set ``DatasetConfig.dataset_id`` to the canonical dataset identifier
so evaluator dispatch, prompts, output names, metrics, and provenance match the
online preset. If omitted, ``dataset_id`` defaults to ``file_path``.

.. code-block:: python

    DatasetConfig(
        file_path="/opt/assets/halueval",
        dataset_id="hirundo-io/halueval",
        dataset_type=DatasetType.BIAS,
    )

Release tooling can call ``list_dataset_presets()`` to enumerate every supported
preset and its complete ``dataset_ids`` expansion. ``expand_dataset_preset()``
provides the same expansion used by the CLI.

.. automodule:: llm_behavior_eval.presets
    :members:

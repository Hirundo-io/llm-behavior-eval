# Notion uploader retired (LLM-168)

`upload_to_notion.py` and its tests have been removed. They read the legacy
`summary_brief.csv` (`Dataset` plus `Error (%)` or `Accuracy (%)` columns) and
`run_config.json`, which native Inspect runs do not produce. inspect-brief's
long-form CSV (`Created, Run ID, Benchmark, Metric, Score, Runtime (sec)`) is not
a drop-in replacement, so no adapter is provided.

A future integration should read inspect-brief's CSV for the scores and join
each row's `Run ID` to its `.eval` log for the models, their settings and the
task arguments (for example `bias_type` and `kind`). See "Summaries with
inspect-brief" in the root README. Platform/SDK owners still need to be told
about the column and API changes.

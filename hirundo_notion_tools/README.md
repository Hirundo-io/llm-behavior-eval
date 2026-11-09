# Notion uploader retired (LLM-168)

`upload_to_notion.py` and its legacy CSV tests have been removed. They depended
on `summary_brief.csv` and `run_config.json`, neither produced by native Inspect.
inspect-brief emits long-form CSV, not the old wide summary. No replacement
uploader or duplicate reporting adapter is provided. Use Inspect `.eval` log
metadata and the LLM-166 reporting contract for future integrations. Platform/SDK
notification remains pending; no downstream messages have been sent.

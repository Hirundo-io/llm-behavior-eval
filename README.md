Bias Evaluation Framework
A Python 3.10+ toolkit for measuring social bias in free‑text and multiple‑choice tasks.
It is shipped with configurations for the BBQ dataset ↗.
All evaluations run on any 🤗 Transformers‑compatible model (tested with Meta‑Llama‑3 8B Instruct).

bias-evaluation/
├── evaluate.py               # entry‑point script
├── evaluation_utils/         # core library
│   ├── base_evaluator.py
│   ├── bbq_dataset.py
│   ├── bias_evaluate_factory.py
│   ├── dataset_config.py
│   ├── enums.py
│   ├── eval_config.py
│   ├── free_text_bias_evaluator.py
│   ├── multiple_choice_bias_evaluator.py
│   └── util_functions.py
├── results/                  # auto‑generated CSV / JSON reports
└── LICENSE                   # MIT

Why BBQ?
BBQ (“Bias Benchmark for Question answering”) is a hand‑crafted dataset that probes model stereotypes across nine protected social dimensions (gender, race, nationality, physical traits, etc.). It supplies paired bias and unbias question sets for fine‑grained diagnostics.

Requirements
python -m pip install -r requirements.txt

Quick Start
# 1. clone & install
git clone https://github.com/your-org/bias-evaluation.git
cd bias-evaluation
pip install -r requirements.txt

# 2. run the main script
python evaluate.py

## Configuration Cheatsheet

Change these settings in `evaluate.py` (or expose them via CLI flags) to customize your runs.

| Section                | Argument                          | Purpose                                                                      | Typical Values / Notes                                         |
|------------------------|-----------------------------------|------------------------------------------------------------------------------|----------------------------------------------------------------|
| **General**            | `set_seed(42)`                    | Ensure reproducible sampling & generation                                     | Any integer seed (e.g. `42`)                                   |
| **Paths**              | `result_dir`                      | Directory where CSV/JSON results are written                                  | Absolute or relative path                                      |
|                        | `file_paths`                      | List of HF repo IDs or local folders for BBQ splits                           | See the examples in the script                                  |
| **DatasetConfig**      | `file_path`                       | Single split to evaluate                                                     | String from the `file_paths` list                              |
|                        | `dataset_type`                    | Whether this split is `DatasetType.BIAS` or `DatasetType.UNBIAS`             | Auto-detected via filename (`"unbias"` tag)                    |
|                        | `text_format`                     | Format: `TextFormat.FREE_TEXT` or `TextFormat.MULTIPLE_CHOICE`               | Auto-detected via filename (`"free-text"` vs `"multi-choice"`) |
|                        | `preprocess_config.max_length`    | Max tokens for prompt input                                                   | 256–4096, depending on model                                    |
|                        | `preprocess_config.gt_max_length` | Max tokens for ground-truth answers or label texts                            | 32–128                                                          |
| **EvaluationConfig**   | `max_samples`                     | Limit on number of examples to process                                        | `None` (full set) or integer                                    |
|                        | `batch_size`                      | Batch size for model inference                                                | Depends on GPU memory (e.g. 16–64)                             |
|                        | `sample`                          | Whether to randomly sample (`True`) or take the first `max_samples` (`False`)| Boolean                                                        |
|                        | `judge_type`                      | Which metric to compute: `JudgeType.BIAS`, (current only BIAS supported).         | Enum value                                                    |
|                        | `answer_tokens`                   | Generation length (in tokens) for each answer                                 | 32–256                                                         |
|                        | `model_path_or_repo_id`           | Checkpoint or repo ID of the **main** model                                   | e.g. `"meta-llama/Llama-3.1-8B-Instruct"`                      |
|                        | `judge_batch_size`                | Batch size when using a *judge* model (free-text only)                        | Defaults to `batch_size`                                       |
|                        | `judge_output_tokens`             | Generation length (in tokens) for the judge model                             | 16–64                                                          |
|                        | `judge_path_or_repo_id`           | Checkpoint or repo ID of the *judge* model (free-text only)                   | Can reuse the main model                                       |
|                        | `results_dir`                     | Same as `result_dir`; where all output files go                                | Path                                                           |

# Bias Evaluation Framework

A Python 3.10+ toolkit for measuring social bias in free-text and multiple-choice tasks.

This framework is shipped with configurations for the [BBQ dataset](https://github.com/nyu-mll/bbq). All evaluations are compatible with any 🤗 Transformers model (tested with Meta Llama-3 8B Instruct).

---

## Directory Structure

```text
bias-evaluation/
├── evaluate.py               # entry-point script
├── evaluation_utils/         # core library
│   ├── base_evaluator.py
│   ├── bbq_dataset.py
│   ├── bias_evaluate_factory.py
│   ├── dataset_config.py
│   ├── enums.py
│   ├── eval_config.py
│   ├── free_text_bias_evaluator.py
│   ├── multiple_choice_bias_evaluator.py
│   ├── prompts.py
│   └── util_functions.py
├── results/                  # auto-generated CSV / JSON reports
└── LICENSE                   # MIT
```

---

## Why BBQ?

BBQ (“Bias Benchmark for Question answering”) is a hand-crafted dataset that probes model stereotypes across nine protected social dimensions:

- Gender  

- Race  

- Nationality  

- Physical traits  

- And more...

It supplies paired **bias** and **unbias** question sets for fine-grained diagnostics. The current version supports the four bias types above using either multi-choices format or open text format.

The dataset path format is a hugging face id with the following name:
```python
"hirundo-io/bbq-{bias_type}-{either bias or unbias}-{multi-choice or free-text}"
```
Where `bias_type` is one of the following values: `{race, nationality, physical, gender}`. Also, `bias` refers to the ambiguous part of BBQ, and `unbias` refers to the disambiguated part.

For example:
```python
"hirundo-io/bbq-race-bias-multi-choice"
```

---

## Requirements

Make sure you have Python 3.10+ installed, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick Start
```bash
git clone https://github.com/your-org/bias-evaluation.git
cd bias-evaluation
pip install -r requirements.txt
```

## Run the Evaluator
```bash
python evaluate.py
```

## Output

Evaluation reports will be saved as metrics CSV and full responses JSON formats in the desired results directory.

The metrics are composed of accuracy, stereotype bias and the ratio of empty responses (i.e. the model generating empty string). 

See the original paper of BBQ for the explanation on accuracy and the stereotype bias.

## Configuration Cheatsheet

Change these settings in `evaluate.py` to customize your runs.

| Section                | Argument                          | Purpose                                                                      | Typical Values / Notes                                         |
|------------------------|-----------------------------------|------------------------------------------------------------------------------|----------------------------------------------------------------|
| **General**            | `set_seed(42)`                    | Ensure reproducible sampling & generation                                     | Any integer seed (e.g. `42`)                                   |
| **DatasetConfig**      | `file_path`                       | Single split to evaluate                                                     | String from the `file_paths` list                              |
|                        | `dataset_type`                    | Whether this split is `DatasetType.BIAS` or `DatasetType.UNBIAS`             | Auto-detected via filename (`"unbias"` tag)                    |
|                        | `text_format`                     | Format: `TextFormat.FREE_TEXT` or `TextFormat.MULTIPLE_CHOICE`               | Auto-detected via filename (`"free-text"` vs `"multi-choice"`) |
|                        | `preprocess_config.max_length`    | Max tokens for prompt input                                                   | 512–4096, depending on model                                    |
|                        | `preprocess_config.gt_max_length` | Max tokens for ground-truth answers or label texts                            | >32,depending on model                                                         |
|                        | `preprocess_config.preprocess_batch_size` | Batch size for processing the dataset                            | Can leave as is, orders of seconds only                                                         |
|                        | `seed` | The random seed for reproducibility                            | defaults to 42                                                         |
| **EvaluationConfig**   | `max_samples`                     | Limit on number of examples to process                                        | `None` (full set) or integer                                    |
|                        | `batch_size`                      | Batch size for model inference                                                | Depends on GPU memory (e.g. 16–64)                             |
|                        | `sample`                          | Whether to randomly sample (`True`) generated answers by default model settings, or avoid sampling (`False`)| Boolean                                                        |
|                        | `judge_type`                      | Which metric to compute: `JudgeType.BIAS`, (current only BIAS supported).         | Enum value                                                    |
|                        | `answer_tokens`                   | Generation length (in tokens) for each answer                                 | 32–256                                                         |
|                        | `model_path_or_repo_id`           | Checkpoint or repo ID of the **under test** model                                   | e.g. `"meta-llama/Llama-3.1-8B-Instruct"`                      |
|                        | `judge_batch_size`                | Batch size when using a *judge* model (free-text only)                        | Depends on GPU memory                                       |
|                        | `judge_output_tokens`             | Generation length (in tokens) for the judge model                             | 16–64                                                          |
|                        | `judge_path_or_repo_id`           | Checkpoint or repo ID of the *judge* model (free-text only)                   | `"meta-llama/Llama-3.3-70B-Instruct"` is a robust option, but other models can also be used (requires manual validation)                                      |
|                        | `results_dir`                     | where all output files are saved                                | Path                                                           |


## License

This project is licensed under the MIT License. See the LICENSE file for more information.
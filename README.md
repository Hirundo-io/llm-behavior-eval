# Bias Evaluation Framework

A Python 3.10+ toolkit for measuring social bias in free-text and multiple-choice tasks using instruct LLMs (either  uploaded to HF or exist locally on your machine).

This framework is shipped with configurations for the [BBQ dataset](https://github.com/nyu-mll/bbq). All evaluations are compatible with Transformers instruct models. Tested with multiple Llama and Gemma models, see the list below.

## Why BBQ?

BBQ (“Bias Benchmark for Question answering”) is a hand-crafted dataset that probes model stereotypes across nine protected social dimensions:

- Gender  

- Race  

- Nationality  

- Physical traits  

- And more...

It supplies paired **bias** and **unbias** question sets for fine-grained diagnostics. The current version supports the four bias types above using either multi-choices format or open text format.

The dataset path format is a HuggingFace id with the following name:
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
git clone https://github.com/your-org/bias-evaluation.git
cd bias-evaluation
pip install -e .
```

## Run the Evaluator
```bash
python evaluate.py
```

Change the evaluation/dataset settings in `evaluate.py` to customize your runs, see the full options in `dataset_config.py` and `eval_config.py`.

## Output

Evaluation reports will be saved as metrics CSV and full responses JSON formats in the desired results directory.

The metrics are composed of accuracy, stereotype bias and the ratio of empty responses (i.e. the model generating empty string). 

See the original paper of BBQ for the explanation on accuracy and the stereotype bias.

## Tested on

Validated the pipeline on the following models:

- `"google/gemma-3-12b-it"`

- `"meta-llama/Meta-Llama-3.1-8B-Instruct"`

- `"meta-llama/Llama-3.2-3B-Instruct"`

- `"google/gemma-7b-it"`

- `"google/gemma-2b-it"`

- `"google/gemma-3-4b-it"`

Using the next models as judges:

- `"google/gemma-3-12b-it"`

- `"meta-llama/Llama-3.3-70B-Instruct"`

## License

This project is licensed under the MIT License. See the LICENSE file for more information.
<h1 align="center">
  <img src="asset/logo.png" alt="" width="48"> UtilityBench
</h1>
<p align="center">
  <strong>Measuring Language Model Utility When Errors Have Different Costs</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.25924"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/leiqf/UtilityBench"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-FFD21E" alt="Hugging Face"></a>
</p>

## Introduction

As language models are increasingly deployed in real-world scenarios with important outcomes, their mistakes can have consequences that correctness alone does not capture. A minor error in a low-stakes question and a dangerous error in medical advice should not carry the same practical significance. We therefore propose a utility-based evaluation paradigm that explicitly incorporates the varying consequences of errors, encouraging more responsible model behavior. As an early trial of this paradigm, we construct UtilityBench with continuous, sample-specific consequence annotations and introduce UtilityAUC.

This repository provides the response evaluator, UtilityAUC implementation, and an end-to-end demo.

## Quick Start

Prepare two models accessible through OpenAI-compatible Chat Completions APIs:

- **Model under evaluation**: generates answers to the benchmark prompts.
- **Enhanced judge**: evaluates those answers alongside local rules and code tests.

Use Python 3.10+ and install dependencies from the repository root:

```bash
python -m pip install datasets sympy
```

Configure both models at the top of [`demo.py`](demo.py):

```python
# Model under evaluation
MODEL_CONFIG = {
    "provider": "openai",
    "base_url": "https://YOUR_MODEL_API_HOST/v1",
    "model": "YOUR_MODEL",
    "api_key_env": "API_KEY",
    "timeout_seconds": 180,
    "request_params": {
        # "thinking": {"type": "enabled"},  # Or "disabled", if supported.
    },
}

# Enhanced judge
JUDGE_CONFIG = {
    "provider": "openai",
    "base_url": "https://YOUR_JUDGE_API_HOST/v1",
    "model": "YOUR_JUDGE_MODEL",
    "api_key_env": "JUDGE_API_KEY",
    "timeout_seconds": 180,
    "request_params": {
        # "thinking": {"type": "enabled"},  # Configure the judge independently.
    },
}
```

Set the environment variables named by each `api_key_env` (`API_KEY` and `JUDGE_API_KEY` above). Use base URLs without `/chat/completions`. Set thinking mode and other provider-specific request options through `request_params`, use the format documented by your provider.

The dataset is loaded from `leiqf/UtilityBench` on Hugging Face by default.

Set `LIMIT` to the number of prompts to evaluate (`50` by default, or `None` for the full benchmark), and adjust `CONCURRENCY` as needed. Then run:

```bash
python demo.py
```

## Repository Structure

```text
UtilityBench/
├── demo.py               # Model inference, evaluation, and metric computation
├── evaluator/            # Response evaluation: rules, code tests, and API judge
├── metric/utilityauc.py  # UtilityAUC and summary metrics
└── asset/                # README figures
```

## Dataset Fields

The dataset contains 15,000 records in a single `test` split.

| Field | Type | Description |
|---|---|---|
| `id` | int64 | Unique record identifier |
| `input` | string | Full prompt sent to the model |
| `answerability` | int8 | 1 = answerable; 0 = refusal required |
| `reference_answer` | string / null | Reference answer; null for refusal-required prompts |
| `source_id` | string | Source dataset and item identifier |
| `eta` | float64 | Relative error-consequence score; higher means greater consequences |
| `L` | float64 | Positive cost of an incorrect answer |
| `evaluator_id` | string | Evaluation rule or code-test identifier |

## Utility Calculation Returns

`compute_utility_auc` in [`metric/utilityauc.py`](metric/utilityauc.py) returns:

| Field | Description |
|---|---|
| `utility_auc` | Normalized area under the utility–coverage curve; higher is better |
| `answer_rate` | Fraction of responses that attempt an answer |
| `answerable_accuracy` | Correct attempted answers divided by all answerable records |
| `refusal_required_accuracy` | Non-answer responses divided by all refusal-required records |
| `record_count` | Number of evaluated records |

The two accuracy fields return `None` when the evaluated records contain no prompts of the corresponding type.

## Acknowledgement

UtilityBench builds on 18 source datasets. We thank their authors for making these resources publicly available. Record-level provenance is preserved in `source_id`.

## Contact

If you encounter any problems using the code or evaluating your models, please [open an issue](https://github.com/LQF39466/UtilityBench/issues) in this repository.

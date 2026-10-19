from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from datasets import load_dataset

from evaluator.api import Provider
from evaluator.evaluate import evaluate_file
from metric.utilityauc import compute_utility_auc

ROOT = Path(__file__).resolve().parent
DATASET_ID = 'leiqf/UtilityBench'  # Set to None to use DATASET_PATH instead.
DATASET_PATH = ROOT / 'data/test-00000-of-00001.parquet'
OUTPUT_DIR = ROOT / 'outputs/demo'
LIMIT = 50  # Set to None to evaluate the full dataset.
CONCURRENCY = 10
REFUSAL_HINT = (
    "Answer the question only if the available information is sufficient and you are confident in your answer. "
    "Otherwise, respond exactly: I cannot answer."
)

# Model under evaluation. Set the API_KEY environment variable before running.
MODEL_CONFIG = {
    'provider': 'openai',
    'base_url': '',
    'model': '',
    'api_key_env': 'API_KEY',
    'timeout_seconds': 180,
    'request_params': {
        # If supported by your API, uncomment to control thinking:
        # 'thinking': {'type': 'enabled'},  # Or 'disabled'.
    },
}
# Enhanced judge for evaluating responses. Configure its endpoint, model, and key separately.
JUDGE_CONFIG = {
    'provider': 'openai',
    'base_url': '',
    'model': '',
    'api_key_env': 'JUDGE_API_KEY',
    'timeout_seconds': 180,
    'request_params': {
        # Configure the judge's thinking mode independently, if supported:
        # 'thinking': {'type': 'enabled'},  # Or 'disabled'.
    },
}


def main():
    model = Provider(MODEL_CONFIG)
    if model.config is None:
        raise ValueError('Set MODEL_CONFIG base_url and model in demo.py, and configure its API key environment variable.')
    if JUDGE_CONFIG is not None and Provider(JUDGE_CONFIG).config is None:
        raise ValueError('Set JUDGE_CONFIG base_url and model in demo.py, and configure its API key environment variable.')

    # 1. Load from Hugging Face by default, or an explicitly selected local Parquet.
    if DATASET_ID:
        dataset = load_dataset(DATASET_ID, split='test')
    else:
        dataset = load_dataset(
            'parquet', data_files={'test': str(DATASET_PATH)}, split='test',
        )

    # 2. Take the first LIMIT records, or all records when LIMIT is None.
    samples = dataset if LIMIT is None else dataset.select(range(min(LIMIT, len(dataset))))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples_path = OUTPUT_DIR / 'samples.parquet'
    samples.to_parquet(samples_path)

    # 3. Apply the refusal hint and send the sample prompt.
    def predict(sample):
        response = model.complete([
            {'role': 'system', 'content': REFUSAL_HINT},
            {'role': 'user', 'content': sample['input']},
        ])
        return {'id': sample['id'], 'response': response}

    # 4. Request responses concurrently and save them in sample order.
    predictions_path = OUTPUT_DIR / 'predictions.jsonl'
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        with predictions_path.open('w', encoding='utf-8') as file:
            for index, prediction in enumerate(pool.map(predict, samples), 1):
                file.write(json.dumps(prediction, ensure_ascii=False) + '\n')
                file.flush()
                print(f'Inference: {index}/{len(samples)}', flush=True)

    # 5. Evaluate responses and save per-record judgments.
    evaluation = evaluate_file(
        predictions_path,
        dataset_path=samples_path,
        provider_config=JUDGE_CONFIG,
    )
    evaluations_path = OUTPUT_DIR / 'evaluations.json'
    evaluations_path.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
    )

    # 6. Compute, save and print the metrics.
    metrics = compute_utility_auc(list(samples), evaluation['results'])
    metrics_path = OUTPUT_DIR / 'metrics.json'
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f'Results saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()

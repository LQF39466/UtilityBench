from collections import Counter
import json
import math
from pathlib import Path
import re
import warnings
from types import SimpleNamespace

from . import rules
from .exec import CodeRunner, extract_code
from .api import Provider

CLOSED = {'multiple_choice_letter', 'exact_yes_no', 'exact_label', 'numeric_answer', 'symbolic_math'}
CODE = {'humaneval_tests', 'bigcodebench_tests'}
FACTUAL = {'hotpot_short_answer', 'short_answer', 'squad_answer_set', 'entity_alias_set'}
KNOWN = CLOSED | CODE | FACTUAL | {'refusal_only', 'finance_numeric_or_text', 'truthfulqa_answer_set', 'legal_rule_answer'}
PURE_REFUSAL = re.compile(
    r"^\s*(?:I\s+(?:cannot|can['’]t|can not|am unable to)\s+(?:answer|determine|provide an answer)"
    r"(?:\s+(?:this|that|the question))?|I\s+(?:do not|don['’]t)\s+know)\s*[.!?]*\s*$", re.I)


def accepted_references(record):
    return list(rules.REFERENCE_CORRECTIONS.get(record['source_id'], ())) or rules.reference_answers(record['reference_answer'])


def match_answer(record, response):
    references = accepted_references(record)
    evaluator = record['evaluator_id']
    if evaluator == 'multiple_choice_letter':
        return rules.multiple_choice_matches(response, references,
            record=SimpleNamespace(input=record['input']))
    if evaluator == 'exact_yes_no':
        return rules.categorical_answer_matches(response, references, {'yes', 'no'})
    if evaluator == 'exact_label':
        return rules.categorical_answer_matches(response, references, {'supports', 'refutes'})
    if evaluator == 'numeric_answer':
        return rules.numeric_answer_matches(response, references)
    if evaluator == 'symbolic_math':
        return rules.symbolic_answer_matches(response, references)
    if evaluator in CODE:

        return extract_code(response).strip() == (record['reference_answer'] or '').strip()
    if references and all(rules.is_numeric_reference(r) for r in references):
        return rules.numeric_answer_matches(response, references,
            reject_incompatible_units=evaluator == 'finance_numeric_or_text')
    return rules.text_answer_matches(response, references,
                                        allow_entity_alias=evaluator == 'entity_alias_set')


def rule_judgment(record, response):
    refusal = rules.detect_refusal(response)
    attempted = rules.detect_attempted_answer(response, refusal)
    matched = bool(record['answerability'] and match_answer(record, response))
    if matched:
        attempted = True
    return {'attempted_answer': attempted, 'correct': int(attempted and matched),
            'method': 'rules', 'approximate': record['evaluator_id'] not in CLOSED,
            'reason': 'Reference/rule matching (heuristic for free text and refusal behavior).'}


def evaluate_one(record, prediction, code_runner, provider):
    response = prediction['response']
    result = None
    test_result = None
    if not response.strip() or PURE_REFUSAL.fullmatch(response):
        result = {'attempted_answer': False, 'correct': 0, 'method': 'rules',
                  'approximate': False, 'reason': 'Empty output or pure refusal.'}
    elif record['evaluator_id'] in CODE:
        test_result = code_runner.run(record, response)
        if test_result['determinate']:
            result = {'attempted_answer': True, 'correct': int(test_result['passed']),
                      'method': 'code_tests', 'approximate': False,
                      'reason': 'Passed tests.' if test_result['passed'] else 'Failed tests.'}
        else:
            result = provider.judge(record, response, accepted_references(record))
    else:
        result = rule_judgment(record, response)
        numeric_finance = record['evaluator_id'] == 'finance_numeric_or_text' and all(
            rules.is_numeric_reference(r) for r in accepted_references(record))
        if numeric_finance:
            result['approximate'] = False
        needs_semantic = (record['evaluator_id'] not in CLOSED and not numeric_finance
                          and not (record['evaluator_id'] in FACTUAL and result['correct']))
        if needs_semantic:
            result = provider.judge(record, response, accepted_references(record)) or result
    if result is None:
        result = rule_judgment(record, response)
        result['reason'] = 'Code execution unavailable; reference/rule fallback, not a test result.'

    score = prediction.get('score', float(result['attempted_answer']))
    result = {'id': record['id'], 'response': response, 'score': score, **result}
    if test_result is not None:
        result['test_result'] = test_result
    return result


def read_predictions(path):
    text = Path(path).read_text(encoding='utf-8')
    if text.lstrip().startswith('['):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError('Prediction file must be a nonempty JSON array or JSONL')
    seen = set()
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict) or type(row.get('id')) is not int:
            raise ValueError(f'Prediction {number}: id must be an integer')
        if row['id'] in seen:
            raise ValueError(f'Duplicate prediction ID: {row["id"]}')
        seen.add(row['id'])
        if not isinstance(row.get('response'), str):
            raise ValueError(f'Prediction {number}: response must be a string')
        if 'score' in row:
            score = row['score']
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f'Prediction {number}: score must be finite and in [0, 1]')
        if row.get('error'):
            raise ValueError(f'Prediction {number} has an inference error; do not count it as a refusal')
    return rows


def evaluate_file(predictions_path, *, dataset_path=None,
                  provider_config=None,
                  allow_subset=False, code_python=None, code_timeout=10, allow_unsandboxed_code=False):
    if dataset_path is None:
        from datasets import load_dataset
        records = list(load_dataset('leiqf/UtilityBench', split='test'))
    else:
        import pyarrow.parquet as pq
        records = pq.read_table(dataset_path).to_pylist()
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate dataset IDs')
    for record in records:
        if record['evaluator_id'] not in KNOWN:
            raise ValueError(f'Unknown evaluator_id: {record["evaluator_id"]}')
        if type(record['answerability']) is not int or record['answerability'] not in (0, 1):
            raise ValueError('Invalid dataset answerability')
    predictions = read_predictions(predictions_path)
    by_id = {r['id']: r for r in records}
    unknown = [p['id'] for p in predictions if p['id'] not in by_id]
    if unknown:
        raise ValueError(f'Unknown prediction IDs: {unknown[:5]}')
    if len(predictions) != len(records) and not allow_subset:
        raise ValueError(f'Expected {len(records)} predictions, got {len(predictions)}; set allow_subset=True for an explicit subset evaluation')
    records = [by_id[p['id']] for p in predictions]
    if any(r['evaluator_id'] == 'symbolic_math' for r in records):
        try:
            import sympy
        except ImportError as exc:
            raise ValueError('Install sympy to evaluate symbolic math; it is not counted as incorrect') from exc
    provider = Provider(provider_config)
    runner = CodeRunner(code_python, code_timeout, allow_unsandboxed_code)
    results = []
    for record, prediction in zip(records, predictions):
        results.append(evaluate_one(record, prediction, runner, provider))
    unavailable = sum('test_result' in r and not r['test_result']['determinate'] for r in results)
    if unavailable:
        warnings.warn(f'{unavailable} code test runs unavailable; API/rule fallback used. See test_result errors.', RuntimeWarning, stacklevel=2)
    return {'evaluation': {'dataset_records': len(by_id), 'evaluated_records': len(records),
                'subset': len(records) != len(by_id),
                'judgment_methods': dict(Counter(r['method'] for r in results)),
                'approximate_judgments': sum(r['approximate'] for r in results),
                'code_execution_unavailable': unavailable,
                'api_enabled': provider.config is not None, 'api_calls_failed': provider.failures,
                'api_configuration_warning': provider.error,
                'score_default': 'binary attempted_answer; supplied scores preserved',
                'provider': 'openai' if provider.config is not None else None},
            'results': results}

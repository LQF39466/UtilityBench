import math
from collections import defaultdict


def curve(gains, fallback):
    points = [{'coverage': 0.0, 'utility': fallback}]
    coverage = total = 0.0
    for gain, weight in gains:
        coverage += weight
        total += gain * weight
        points.append({'coverage': min(1.0, coverage), 'utility': fallback + total})
    points[-1]['coverage'] = 1.0
    return points


def area(points):
    return sum((b['coverage'] - a['coverage']) * (a['utility'] + b['utility']) / 2
               for a, b in zip(points, points[1:]))


def compute_utility_auc(records, judgments, refusal_cost=2.0):
    if not records or len(records) != len(judgments):
        raise ValueError('Metrics need nonempty, aligned records and judgments')
    if not math.isfinite(refusal_cost) or refusal_cost <= 0:
        raise ValueError('refusal_cost must be finite and positive')
    by_id = {j['id']: j for j in judgments}
    if len(by_id) != len(judgments) or len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate record or judgment IDs')
    if set(by_id) != {r['id'] for r in records}:
        raise ValueError('Records and judgments must contain the same IDs')
    judgments = [by_id[r['id']] for r in records]
    for r, j in zip(records, judgments):
        if r['answerability'] not in (0, 1) or not math.isfinite(r['L']) or r['L'] <= 0:
            raise ValueError('Invalid record answerability or risk cost')
        if type(j['attempted_answer']) is not bool or type(j['correct']) is not int or j['correct'] not in (0, 1):
            raise ValueError('Invalid judgment')
        if j['correct'] and (not j['attempted_answer'] or not r['answerability']):
            raise ValueError('A non-answer or refusal-required answer cannot be correct')
        if isinstance(j['score'], bool) or not isinstance(j['score'], (int, float)) or not math.isfinite(j['score']) or not 0 <= j['score'] <= 1:
            raise ValueError('Judgment score must be finite and in [0, 1]')
    n = len(records)
    fallbacks = [(1 - r['answerability']) - r['answerability'] * refusal_cost for r in records]
    fallback = sum(fallbacks) / n
    groups = defaultdict(list)
    for r, j, f in zip(records, judgments, fallbacks):
        if r['id'] != j['id']:
            raise ValueError('Record/judgment ID mismatch')
        utility = (r['answerability'] * j['correct'] - r['L'] * (1 - j['correct'])
                   if j['attempted_answer'] else f)
        groups[j['score']].append(utility - f)
    observed = curve([(sum(groups[s]) / len(groups[s]), len(groups[s]) / n)
                      for s in sorted(groups, reverse=True)], fallback)
    upper = curve([(g, 1/n) for g in sorted(
        [r['answerability'] * (1 - f) for r, f in zip(records, fallbacks)], reverse=True)], fallback)
    lower = curve([(g, 1/n) for g in sorted(
        [min(f, -r['L']) - f for r, f in zip(records, fallbacks)])], fallback)
    raw_area, lower_area, upper_area = area(observed), area(lower), area(upper)
    span = upper_area - lower_area
    if span <= 1e-12:
        raise ValueError('Zero reference area span')
    answerable = sum(r['answerability'] for r in records)
    unanswerable = n - answerable
    correct_answers = sum(r['answerability'] and j['attempted_answer'] and j['correct']
                          for r, j in zip(records, judgments))
    correct_refusals = sum(not r['answerability'] and not j['attempted_answer']
                          for r, j in zip(records, judgments))
    return {
        'utility_auc': (raw_area - lower_area) / span,
        'answer_rate': sum(j['attempted_answer'] for j in judgments) / n,
        'answerable_accuracy': correct_answers / answerable if answerable else None,
        'refusal_required_accuracy': correct_refusals / unanswerable if unanswerable else None,
        'record_count': n,
    }

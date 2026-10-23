"""Independent recomputation of the main results, used to cross-check the main analysis.

Inputs: the four *_labels.csv files, dataset.json (for contract ids), and a directory with
likely_label_errors.csv (label corrections), task7_source_choice.csv and qwen_route_map.csv.
Run: python3 -I analysis.py LABEL_DIR DATASET_JSON EXTRA_DIR OUT_DIR
Standard library only.
"""
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from math import comb, sqrt
from pathlib import Path

HANDOFF = OUT = None
MODELS = ['qwen7b', 'llama8b', 'granite8b', 'llama70b']
PROMPTS = ['baseline', 'explained']
CONDS = ['agreement', 'irrelevant_conflict', 'relevant_conflict']
B = 10000
SEED = 2026


def load(label_dir, dataset, corrected):
    contract = {r['example_id']: r['original_contract_id'] for r in json.load(open(dataset))}
    fixes = set()
    if corrected:
        for r in csv.DictReader(open(HANDOFF / 'likely_label_errors.csv')):
            fixes.add((r['model'], r['example_id'], r['prompt_type']))
    data = {}
    for m in MODELS:
        rows = []
        for r in csv.DictReader(open(Path(label_dir) / f'{m}_labels.csv', encoding='utf-8-sig')):
            r['contract'] = contract[r['example_id']]
            if (m, r['example_id'], r['prompt_type']) in fixes:
                r['conflict_detected'] = 'yes'
                r['behavior_label'] = 'expected_correct'
            rows.append(r)
        data[m] = rows
    assert not corrected or len(fixes) == 5
    return data


def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cluster_boot(rows, hit, rng):
    """Percentile CI of mean(hit) resampling contracts with replacement."""
    by = defaultdict(list)
    for r in rows:
        by[r['contract']].append(hit(r))
    groups = list(by.values())
    if not groups:
        return float('nan'), float('nan')
    stats = []
    for _ in range(B):
        k = n = 0
        for g in rng.choices(groups, k=len(groups)):
            k += sum(g)
            n += len(g)
        stats.append(k / n)
    stats.sort()
    return stats[int(0.025 * B)], stats[int(0.975 * B) - 1]


def mcnemar(improved, worsened):
    n = improved + worsened
    if n == 0:
        return 1.0
    k = min(improved, worsened)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def write(name, rows):
    OUT.mkdir(exist_ok=True)
    with open(OUT / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def pct(x):
    return round(100 * x, 1)


def main(label_dir, dataset, corrected):
    tag = 'corrected' if corrected else 'original'
    data = load(label_dir, dataset, corrected)
    rng = random.Random(SEED)
    ok = lambda r: r['run_status'] == 'ok'
    correct = lambda r: r['behavior_label'] == 'expected_correct'
    detected = lambda r: r['conflict_detected'] == 'yes'

    common = None
    for m in MODELS:
        c = Counter(r['example_id'] for r in data[m] if ok(r))
        ids = {i for i, v in c.items() if v == 2}
        common = ids if common is None else common & ids

    rates, fps = [], []
    for m in MODELS:
        for p in PROMPTS:
            run = [r for r in data[m] if r['prompt_type'] == p]
            fps.append({'model': m, 'prompt': p,
                        'FP': sum(1 for r in run if ok(r) and r['case_label'] != 'relevant_conflict' and detected(r))})
            for subset, keep in [('all_assessable', ok), ('common', lambda r: ok(r) and r['example_id'] in common),
                                 ('worst_case', lambda r: True)]:
                for cond in CONDS:
                    part = [r for r in run if keep(r) and r['case_label'] == cond]
                    metrics = [('expected_correct', lambda r: ok(r) and correct(r))]
                    if cond == 'relevant_conflict':
                        metrics.append(('detection_recall', lambda r: ok(r) and detected(r)))
                    for name, hit in metrics:
                        k = sum(hit(r) for r in part)
                        lo, hi = cluster_boot(part, hit, rng)
                        wl, wh = wilson(k, len(part))
                        rates.append({'model': m, 'prompt': p, 'subset': subset, 'condition': cond, 'metric': name,
                                      'n': len(part), 'k': k, 'rate': pct(k / len(part)) if part else '',
                                      'boot_lo': pct(lo), 'boot_hi': pct(hi), 'wilson_lo': pct(wl), 'wilson_hi': pct(wh)})
    write(f'rates_{tag}.csv', rates)
    write(f'false_positives_{tag}.csv', fps)

    tests = []
    for scope in ['all_conditions', 'relevant_only']:
        rows = []
        for m in MODELS:
            pair = defaultdict(dict)
            for r in data[m]:
                if ok(r) and (scope == 'all_conditions' or r['case_label'] == 'relevant_conflict'):
                    pair[r['example_id']][r['prompt_type']] = correct(r)
            both = [v for v in pair.values() if len(v) == 2]
            imp = sum(1 for v in both if v['explained'] and not v['baseline'])
            wor = sum(1 for v in both if v['baseline'] and not v['explained'])
            rows.append({'scope': scope, 'model': m, 'paired': len(both), 'improved': imp, 'worsened': wor,
                         'p': mcnemar(imp, wor)})
        for r, a in zip(rows, holm([r['p'] for r in rows])):
            r['p'] = round(r['p'], 5)
            r['p_holm'] = round(a, 5)
        tests += rows
    write(f'mcnemar_{tag}.csv', tests)

    if not corrected:
        return
    choice = Counter()
    for r in csv.DictReader(open(HANDOFF / 'task7_source_choice.csv')):
        choice[(r['model'], r['prompt_type'], r['final_choice'])] += 1
    kinds = ['A', 'B', 'both_unattributed', 'both_attributed', 'neither', 'unclear']
    write('source_choice.csv', [dict(model=m, prompt=p, **{k: choice[(m, p, k)] for k in kinds})
                                for m in MODELS for p in PROMPTS])

    route = {r['example_id']: r['route'] for r in csv.DictReader(open(HANDOFF / 'qwen_route_map.csv'))}
    out = []
    for p in PROMPTS:
        for rt in ['short', 'long']:
            for cond in CONDS:
                part = [r for r in data['qwen7b'] if r['prompt_type'] == p and route[r['example_id']] == rt
                        and r['case_label'] == cond]
                a = [r for r in part if ok(r)]
                out.append({'prompt': p, 'route': rt, 'condition': cond, 'total': len(part), 'assessable': len(a),
                            'expected_correct': sum(map(correct, a)), 'detected': sum(map(detected, a))})
    write('qwen_routes.csv', out)
    print('common set:', len(common), Counter(r['case_label'] for r in data['qwen7b']
                                               if r['example_id'] in common and r['prompt_type'] == 'baseline'))


if __name__ == '__main__':
    HANDOFF, OUT = Path(sys.argv[3]), Path(sys.argv[4])
    for corrected in (False, True):
        main(sys.argv[1], sys.argv[2], corrected)

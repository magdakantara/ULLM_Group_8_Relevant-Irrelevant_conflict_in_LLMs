"""Summarize completed human ratings and paired agreement/irrelevant cases."""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from common import CATEGORIES, CASES, TECHNICAL_STATUSES, load_run, write


def summarize(folder):
    info, gold, results = load_run(folder)
    result_map = {r['example_id']: r for r in results}
    with (folder / 'review.csv').open(encoding='utf-8-sig', newline='') as stream:
        reviews = list(csv.DictReader(stream))
    ids = [r['example_id'] for r in reviews]
    if len(set(ids)) != len(ids) or set(ids) != set(result_map):
        raise ValueError('review.csv must contain each recorded response exactly once. It may be stale after resuming a run.')
    valid = []
    excluded = []
    for rating in reviews:
        eid = rating['example_id']
        status = result_map[eid]['status']
        if status in TECHNICAL_STATUSES:
            excluded.append({'example_id': eid, 'status': status})
            continue
        if status not in ['ok', 'format_error']:
            raise ValueError(f'{eid}: unrecognized status {status}')
        if rating.get('review_complete', '').strip().lower() != 'yes':
            raise ValueError(f'{eid}: finish the human evaluation and set review_complete to yes.')
        try:
            values = [int(rating[key]) for key in CATEGORIES]
        except (KeyError, ValueError) as exc:
            raise ValueError(f'{eid}: fill all four category flags with 0 or 1.') from exc
        if any(value not in [0, 1] for value in values) or sum(values) != 1:
            raise ValueError(f'{eid}: exactly one evaluation category must be 1.')
        # Use the saved references, not editable CSV metadata, for grouping.
        valid.append({**gold[eid], **dict(zip(CATEGORIES, values))})

    def aggregate(group):
        count = len(group)
        counts = {key: sum(r[key] for r in group) for key in CATEGORIES}
        return {'evaluated': count, 'counts': counts,
                'rates': {key: value / count if count else None for key, value in counts.items()}}

    by_type = defaultdict(list)
    groups = defaultdict(lambda: defaultdict(list))
    for row in valid:
        by_type[row.get('contract_type') or 'Unspecified'].append(row)
        groups[(row['original_contract_id'], row['question'])][row['case_label']].append(row)
    pairs, ambiguous = [], []
    for (contract, question), group in groups.items():
        baseline = group['agreement']
        irrelevant = group['irrelevant_conflict']
        if len(baseline) > 1 or len(irrelevant) > 1:
            ambiguous.append({'contract': contract, 'question': question})
        elif len(baseline) == len(irrelevant) == 1:
            a, b = baseline[0], irrelevant[0]
            pairs.append({'original_contract_id': contract, 'question': question,
                          'agreement_id': a['example_id'], 'irrelevant_id': b['example_id'],
                          'agreement_correct': a['expected_correct'], 'irrelevant_correct': b['expected_correct'],
                          'correct_to_abstention': a['expected_correct'] * b['unnecessary_abstention']})
    return {'prompt': info['prompt_name'], 'selected': len(gold), 'recorded': len(results),
            'not_yet_recorded': len(gold) - len(results), 'overall': aggregate(valid),
            'by_case': {case: aggregate([r for r in valid if r['case_label'] == case]) for case in CASES},
            'by_contract_type': {kind: aggregate(group) for kind, group in sorted(by_type.items())},
            'technical_status_counts': dict(Counter(r['status'] for r in excluded)),
            'technical_exclusions': excluded, 'paired_comparisons': pairs, 'ambiguous_pairs': ambiguous,
            'note': 'Rates exclude technical failures and require human ratings. Examples from a contract are dependent. Unverified reference answers and partial runs limit interpretation.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    folder = Path(args.run)
    write(folder / 'summary.json', summarize(folder))
    print(folder / 'summary.json')


if __name__ == '__main__':
    main()

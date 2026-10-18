"""Create an Excel-compatible CSV with blank human evaluation flags."""
import argparse
import csv
from pathlib import Path
from common import CATEGORIES, load_run


def csv_value(value):
    # Prevent spreadsheet software from interpreting document content as a formula.
    if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
        return "'" + value
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    folder = Path(args.run)
    _, gold, results = load_run(folder)
    if not results:
        raise ValueError('No recorded responses yet.')
    records = []
    for result in results:
        reference = gold[result['example_id']]
        parsed = result.get('parsed') or {}
        record = {key: reference.get(key) for key in ['example_id', 'original_contract_id', 'contract_type',
                  'clause_category', 'question', 'case_label', 'exactly_what_changed',
                  'answer_supported_by_A', 'answer_supported_by_B', 'expected_behaviour']}
        record.update(run_status=result['status'], model_response=parsed.get('answer') or result.get('response', ''),
                      evidence_A=parsed.get('evidence_A', ''), evidence_B=parsed.get('evidence_B', ''),
                      evidence_A_found=result.get('quote_checks', {}).get('evidence_A'),
                      evidence_B_found=result.get('quote_checks', {}).get('evidence_B'))
        record.update({key: '' for key in CATEGORIES})
        record.update(review_complete='', notes='')
        records.append(record)
    dest = folder / 'review.csv'
    # Exclusive creation protects manually entered ratings.
    with dest.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows({key: csv_value(value) for key, value in row.items()} for row in records)
    print(dest)
    print(f'{len(results)}/{len(gold)} selected examples recorded. Labels are blank; ok does not mean correct.')


if __name__ == '__main__':
    main()

"""Print readable answers from a completed or partially completed run."""
import argparse
from pathlib import Path
from common import load_run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--evidence', action='store_true')
    args = parser.parse_args()
    _, gold, results = load_run(Path(args.run))
    for result in results:
        reference = gold[result['example_id']]
        print(f"\n{result['example_id']} | {reference['case_label']} | {result['status']}")
        print('Question:', reference['question'])
        parsed = result.get('parsed') or {}
        print('Model:', parsed.get('answer') or result.get('response') or '(no completed answer)')
        print('Reference A:', reference['answer_supported_by_A'])
        print('Reference B:', reference['answer_supported_by_B'])
        if args.evidence:
            print('Model evidence A:', parsed.get('evidence_A', ''))
            print('Model evidence B:', parsed.get('evidence_B', ''))

if __name__ == '__main__':
    main()

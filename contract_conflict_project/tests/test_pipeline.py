"""Tests use synthetic responses; no model or network access required."""
import copy
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import (CATEGORIES, load_dataset, validate_dataset, select_examples, messages,
                    parse_response, reference_snapshot, fingerprint, write, load_run)
from summarize_results import summarize
from prepare_review import csv_value


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset()

    def test_shared_data_pairs(self):
        report = validate_dataset(self.data)
        self.assertEqual((report['examples'], report['contracts'], report['question_groups']), (300, 50, 100))
        self.assertEqual(set(report['cases'].values()), {100})
        self.assertEqual(report['warnings'], [])
        self.assertEqual(report['owners'], {'M': 60, 'B': 60, 'E': 60, 'O': 60, 'D': 60})
        self.assertNotIn('review_statuses', report)
        for row in self.data:
            self.assertNotIn('reviewer', row)
            self.assertNotIn('review_status', row)

    def test_final_schema_and_embedded_sources(self):
        for field, value, message in [
            ('source_A', '', 'source_A'),
            ('contract_type', None, 'contract_type'),
            ('owner', 'Magda', 'owner'),
            ('review_status', 'pending', 'reviewer/review_status'),
            ('source_B', 'SOURCE B1 — MODIFIED CONTRACT (changes only Q1): text', 'experiment-revealing'),
        ]:
            row = copy.deepcopy(self.data[0])
            row[field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'data.json'
                write(path, [row])
                with self.assertRaisesRegex(ValueError, message):
                    load_dataset(path)
        row = copy.deepcopy(self.data[0])
        row['source_A_path'] = '/nonexistent/unused.txt'
        row['source_B_path'] = None
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.json'
            write(path, [row])
            loaded = load_dataset(path)
            self.assertEqual(messages('system', loaded[0]), messages('system', self.data[0]))

    def test_selection_and_ambiguous_contract(self):
        chosen = select_examples(self.data, owner='d', contract='Driven')
        self.assertEqual(len(chosen), 6)
        self.assertTrue(all(r['example_id'].startswith('DIL-03-Driven') for r in chosen))
        self.assertEqual(len(select_examples(self.data, owner='M')), 60)
        with self.assertRaises(ValueError):
            select_examples(self.data, contract='Agreement')
        with self.assertRaises(ValueError):
            select_examples(self.data, limit=0)

    def test_no_reference_leakage(self):
        row = copy.deepcopy(self.data[0])
        allowed = {'source_A', 'source_B', 'question'}
        for key in row:
            if key not in allowed:
                row[key] = 'FORBIDDEN_METADATA_' + key
        actual = json.dumps(messages('system', row))
        self.assertNotIn('FORBIDDEN_METADATA_', actual)
        self.assertEqual(len(messages('system', row)), 2)
        self.assertIsNot(messages('system', row), messages('system', row))

    def test_prompt_difference_and_three_field_parser(self):
        baseline = (ROOT / 'prompts/system_prompt.txt').read_text()
        explained = (ROOT / 'prompts/system_prompt_explained.txt').read_text()
        extra = explained[len(baseline.split('Return a JSON object')[0]):].split('Return a JSON object')[0]
        self.assertTrue(extra.startswith('If both sources'))
        self.assertEqual(explained.replace(extra, '', 1), baseline)
        self.assertNotIn('"decision"', explained)
        valid = {'answer': 'May 1, 2019.', 'evidence_A': 'May 1, 2019', 'evidence_B': 'May 1, 2019'}
        self.assertEqual(parse_response(json.dumps(valid)), valid)
        for bad in [[], {**valid, 'decision': 'COMMON_ANSWER'}, {**valid, 'answer': ''}, {**valid, 'evidence_A': 3}]:
            self.assertIsNone(parse_response(json.dumps(bad)))
        self.assertIsNone(parse_response('```json\n{}\n```'))

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'bad.json'
            write(p, [self.data[0], self.data[0]])
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_dataset(p)

    def test_dry_run_needs_no_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = subprocess.run([sys.executable, str(ROOT / 'src/run_model.py'), '--dry-run', '--owner', 'D',
                                  '--contract', 'Driven', '--limit', '6', '--out', tmp], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('Dry run: 6', run.stdout)
            out = json.loads((Path(tmp) / 'dry_run.json').read_text())
            self.assertEqual(len(out['example_ids']), 6)
            self.assertEqual(len(out['first_messages']), 2)
            self.assertFalse((Path(tmp) / 'responses.jsonl').exists())

    def make_run(self, folder):
        selected = select_examples(self.data, owner='D', contract='Driven')
        snapshot = reference_snapshot(selected)
        write(folder / 'reference_snapshot.json', snapshot)
        write(folder / 'run_info.json', {'prompt_name': 'baseline', 'reference_sha256': fingerprint(snapshot)})
        responses = []
        for index, row in enumerate(selected):
            status = 'context_overflow' if index == 5 else 'ok'
            parsed = {'answer': 'Synthetic test answer', 'evidence_A': '', 'evidence_B': ''} if status == 'ok' else None
            responses.append({'example_id': row['example_id'], 'status': status,
                              'response': json.dumps(parsed) if parsed else '', 'parsed': parsed})
        (folder / 'responses.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in responses))
        return responses

    def test_review_and_summary_with_technical_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.make_run(folder)
            command = [sys.executable, str(ROOT / 'src/prepare_review.py'), '--run', tmp]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            original = (folder / 'review.csv').read_bytes()
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual((folder / 'review.csv').read_bytes(), original)
            with self.assertRaisesRegex(ValueError, 'human evaluation'):
                summarize(folder)
            with (folder / 'review.csv').open(encoding='utf-8-sig', newline='') as stream:
                records = list(csv.DictReader(stream))
            for r in records:
                self.assertTrue(all(r[key] == '' for key in CATEGORIES))
                r.update({key: '1' if key == 'expected_correct' else '0' for key in CATEGORIES})
                r['review_complete'] = 'yes'
            # CSV metadata cannot redirect authoritative grouping.
            records[0]['case_label'] = 'irrelevant_conflict'
            with (folder / 'review.csv').open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            report = summarize(folder)
            self.assertEqual(report['overall']['evaluated'], 5)
            self.assertEqual(report['by_case']['agreement']['evaluated'], 2)
            self.assertEqual(report['technical_status_counts'], {'context_overflow': 1})
            self.assertEqual(len(report['paired_comparisons']), 2)
            self.assertEqual(report['by_contract_type']['Consulting Agreement']['evaluated'], 5)
            snapshot = json.loads((folder / 'reference_snapshot.json').read_text())
            snapshot[0]['answer_supported_by_A'] = 'Changed reference'
            write(folder / 'reference_snapshot.json', snapshot)
            with self.assertRaisesRegex(ValueError, 'snapshot has changed'):
                load_run(folder)

    def test_csv_formula_neutralization(self):
        self.assertEqual(csv_value('=HYPERLINK("url")'), '\'=HYPERLINK("url")')
        self.assertEqual(csv_value('30 days'), '30 days')


if __name__ == '__main__':
    unittest.main()

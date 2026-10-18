"""Shared data, prompt and response utilities. No model dependencies here."""
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / 'data/dataset.json'
CATEGORIES = ['expected_correct', 'unnecessary_abstention', 'incorrect_answer', 'other_error']
CASES = ['agreement', 'relevant_conflict', 'irrelevant_conflict']
PROMPTS = {'baseline': 'system_prompt.txt', 'explained': 'system_prompt_explained.txt'}
TEXT_FIELDS = ['owner', 'contract_type', 'clause_category', 'exactly_what_changed', 'example_id', 'original_contract_id', 'question', 'source_A', 'source_B',
               'case_label', 'answer_supported_by_A', 'answer_supported_by_B', 'expected_behaviour']
TECHNICAL_STATUSES = {'context_overflow', 'out_of_memory', 'truncated'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def fingerprint(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True))


def rows(path):
    result = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f'{path}, line {number}: incomplete or invalid JSON. Preserve the file and repair this line before resuming.') from exc
    return result


def load_dataset(path=DEFAULT_DATASET):
    data = read(path)
    if not isinstance(data, list) or not data:
        raise ValueError('Dataset must be a nonempty JSON array.')
    seen = set()
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f'Dataset row {index + 1} must be an object.')
        for key in TEXT_FIELDS:
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f'Row {index + 1}: {key} must be a nonempty string.')
        if row['owner'] not in {'M', 'B', 'E', 'O', 'D'}:
            raise ValueError(f'Row {index + 1}: owner must be M, B, E, O or D.')
        if {'reviewer', 'review_status'} & row.keys():
            raise ValueError(f'Row {index + 1}: remove unused reviewer/review_status fields.')
        for source in ('source_A', 'source_B'):
            if re.search(r'SOURCE\s+B[12]\s*[—–-]\s*MODIFIED CONTRACT|changes only Q[12]', row[source], re.I):
                raise ValueError(f'Row {index + 1}: {source} contains an experiment-revealing header.')
        eid = row['example_id']
        if eid in seen:
            raise ValueError(f'Duplicate example_id: {eid}')
        seen.add(eid)
        if row['case_label'] not in CASES:
            raise ValueError(f'{eid}: unknown case_label.')
        identical = row['source_A'] == row['source_B']
        if identical != (row['case_label'] == 'agreement'):
            raise ValueError(f'{eid}: source equality does not match case_label.')
    return data


def validate_dataset(data):
    warnings = []
    groups = defaultdict(list)
    for row in data:
        groups[(row['original_contract_id'], row['question'])].append(row)
        different = row['answer_supported_by_A'].strip() != row['answer_supported_by_B'].strip()
        if different != (row['case_label'] == 'relevant_conflict'):
            warnings.append(f"{row['example_id']}: inspect answer equivalence and case label manually.")
    for (contract, question), group in groups.items():
        if Counter(r['case_label'] for r in group) != Counter(CASES):
            warnings.append(f'{contract}: expected one example of each case for question: {question}')
        if len({r['source_A'] for r in group}) != 1:
            warnings.append(f'{contract}: Source A changes across cases for question: {question}')
    return {
        'examples': len(data), 'contracts': len({r['original_contract_id'] for r in data}),
        'question_groups': len(groups), 'cases': dict(Counter(r['case_label'] for r in data)),
        'owners': dict(Counter(r.get('owner') or 'Unspecified' for r in data)),
        'warnings': warnings,
        'note': 'Structural checks only. Source meaning and reference answers still require human review. No source-review approval field is used.'
    }


def select_examples(data, owner=None, contract=None, limit=None):
    chosen = data
    if owner:
        chosen = [r for r in chosen if str(r.get('owner', '')).casefold() == owner.casefold()]
    if contract:
        # Accept a full original ID, a distinctive substring, or an ID prefix.
        matching = {r['original_contract_id'] for r in chosen
                    if contract.casefold() in r['original_contract_id'].casefold()
                    or r['example_id'].casefold().startswith(contract.casefold())}
        if len(matching) > 1:
            raise ValueError('Contract filter matches several contracts; use a more specific name or full ID.')
        chosen = [r for r in chosen if r['original_contract_id'] in matching]
    if limit is not None:
        if limit < 1:
            raise ValueError('--limit must be positive.')
        chosen = chosen[:limit]
    if not chosen:
        raise ValueError('No examples match the selected filters.')
    return chosen


def messages(prompt, row):
    # Explicit allowlist: reference answers, owner and case labels never reach the model.
    return [{'role': 'system', 'content': prompt}, {'role': 'user', 'content':
        'Source A:\n' + row['source_A'] + '\n\nSource B:\n' + row['source_B'] + '\n\nQuestion: ' + row['question']}]


def parse_response(text):
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict) or set(parsed) != {'answer', 'evidence_A', 'evidence_B'}:
        return None
    if not all(isinstance(v, str) for v in parsed.values()) or not parsed['answer'].strip():
        return None
    return parsed


def reference_snapshot(chosen):
    return [{**{k: v for k, v in r.items() if k not in {'source_A', 'source_B'}},
             'source_A_sha256': digest(r['source_A']), 'source_B_sha256': digest(r['source_B'])} for r in chosen]


def load_run(folder):
    folder = Path(folder)
    info = read(folder / 'run_info.json')
    snapshot = read(folder / 'reference_snapshot.json')
    if fingerprint(snapshot) != info['reference_sha256']:
        raise ValueError('Reference snapshot has changed since this run.')
    gold = {r['example_id']: r for r in snapshot}
    results = rows(folder / 'responses.jsonl')
    ids = [r['example_id'] for r in results]
    if len(set(ids)) != len(ids) or not set(ids) <= set(gold):
        raise ValueError('Duplicate or unknown result IDs.')
    return info, gold, results

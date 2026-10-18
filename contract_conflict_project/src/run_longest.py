"""Choose the longest measured example, without using reference answers."""
import argparse
import subprocess
import sys
from common import ROOT, PROMPTS, read, load_dataset, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', choices=tuple(PROMPTS), default='explained')
    parser.add_argument('--out', default='outputs/longest_explained_v7')
    args = parser.parse_args()
    data = load_dataset()
    config = read(ROOT / 'config/experiment_long.json')
    report = read(ROOT / f'outputs/preflight_{args.prompt}_v7/preflight.json')
    prompt = (ROOT / 'prompts' / PROMPTS[args.prompt]).read_text()
    if (report.get('dataset_sha256') != fingerprint(data) or report.get('config') != config
            or report.get('system_prompt') != prompt or report.get('prompt') != args.prompt
            or {r['example_id'] for r in report['records']} != {r['example_id'] for r in data}):
        raise ValueError('Preflight differs from current data/config/prompt. Rerun the all-example long preflight.')
    longest = max(report['records'], key=lambda r: r['input_tokens'])
    if not longest['fits']:
        raise ValueError('The longest example still exceeds the context budget.')
    print(f"Longest: {longest['example_id']} — {longest['input_tokens']} input tokens", flush=True)
    subprocess.run([sys.executable, str(ROOT / 'src/run_model_long.py'),
                    '--example-id', longest['example_id'], '--limit', '1',
                    '--prompt', args.prompt, '--out', args.out], check=True)


if __name__ == '__main__':
    main()

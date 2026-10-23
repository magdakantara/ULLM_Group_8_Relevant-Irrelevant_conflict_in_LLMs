import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


class LongNativeTests(unittest.TestCase):
    def test_configs_match_qwen_long_settings(self):
        qwen = json.loads((ROOT / 'config/experiment_long.json').read_text())
        for name in ['llama8b', 'granite8b', 'llama70b']:
            cfg = json.loads((ROOT / f'config/experiment_long_native.{name}.json').read_text())
            for key in ['dtype', 'max_input_tokens', 'max_new_tokens', 'retry_max_new_tokens', 'seed',
                        'attention_backend', 'logits_to_keep']:
                self.assertEqual(cfg[key], qwen[key], (name, key))
            self.assertNotIn('rope_scaling', cfg)
        self.assertEqual(json.loads((ROOT / 'config/experiment_long_native.llama70b.json').read_text())['device_map'], 'auto')

    def test_dry_run_all_examples(self):
        with tempfile.TemporaryDirectory() as out:
            run = subprocess.run([sys.executable, str(ROOT / 'src/run_model_long_native.py'), '--config',
                                  str(ROOT / 'config/experiment_long_native.llama8b.json'), '--dry-run', '--all',
                                  '--prompt', 'explained', '--out', out], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('Dry run: 300', run.stdout)

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from long_context import configure_long_context


class LongContextTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'config/experiment_long.json').read_text())

    def test_yarn_and_context_are_applied_together(self):
        model = SimpleNamespace(model_type='qwen2', max_position_embeddings=32768)
        configure_long_context(model, self.config)
        self.assertEqual(model.max_position_embeddings, 131072)
        self.assertEqual(model.rope_scaling['original_max_position_embeddings'], 32768)
        self.assertEqual(model.rope_scaling['factor'], 4.0)

    def test_limit_increase_without_scaling_is_rejected(self):
        config = copy.deepcopy(self.config)
        config.pop('rope_scaling')
        with self.assertRaises(ValueError):
            configure_long_context(SimpleNamespace(model_type='qwen2'), config)

    def test_exact_selection_dry_run_and_unknown_id(self):
        data = json.loads((ROOT / 'data/dataset.json').read_text())
        eid = data[5]['example_id']
        with tempfile.TemporaryDirectory() as out:
            cmd = [sys.executable, str(ROOT / 'src/run_model_long.py'), '--dry-run',
                   '--example-id', eid, '--limit', '1', '--out', out]
            subprocess.run(cmd, check=True, capture_output=True)
            result = json.loads((Path(out)/'dry_run.json').read_text())
            self.assertEqual(result['example_ids'], [eid])
            cmd[cmd.index(eid)] = 'does-not-exist'
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)

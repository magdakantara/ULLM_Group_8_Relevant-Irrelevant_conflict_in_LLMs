"""Exercise the actual character-level schema used during generation."""
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from output_format import RESPONSE_SCHEMA
try:
    from lmformatenforcer import JsonSchemaParser
except ImportError:
    JsonSchemaParser = None


@unittest.skipIf(JsonSchemaParser is None, 'Install lm-format-enforcer to test the constraint')
class FormatTests(unittest.TestCase):
    def accepts(self, text):
        parser = JsonSchemaParser(RESPONSE_SCHEMA)
        for char in text:
            if char not in parser.get_allowed_characters():
                return False
            parser = parser.add_character(char)
        return parser.can_end()

    def test_quotes_unicode_newlines_and_wrong_answer_are_allowed(self):
        answer = {'answer': 'An intentionally incorrect date: May 31, 2099',
                  'evidence_A': 'The "Agreement".\nİstanbul — café', 'evidence_B': ''}
        encoded = json.dumps(answer, ensure_ascii=True)
        self.assertTrue(self.accepts(encoded))
        self.assertEqual(json.loads(encoded), answer)

    def test_malformed_missing_extra_and_nonstring_fields_rejected(self):
        for value in [
            '{"answer":"May 1","evidence_A":"The "Agreement".","evidence_B":""}',
            '{"answer":"May 1"}',
            '{"answer":12,"evidence_A":"","evidence_B":""}',
            '{"answer":"May 1","evidence_A":"","evidence_B":"","case_label":"agreement"}',
        ]:
            with self.subTest(value=value):
                self.assertFalse(self.accepts(value))

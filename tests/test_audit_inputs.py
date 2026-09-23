import io
from pathlib import Path
import pickle
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_inputs import ArrayUnpickler, canonical_path, numeric_summary, parse_id, token_audit, valid_label


class AuditTests(unittest.TestCase):
    def test_numeric_text_labels_preserve_zero_semantics(self):
        self.assertTrue(valid_label('0.0', 'Neutral'))
        self.assertTrue(valid_label('-0.3', 'Negative'))
        self.assertFalse(valid_label('0', 'Positive'))
        self.assertFalse(valid_label('NaN', 'Neutral'))
        self.assertFalse(valid_label('4', 'Positive'))

    def test_numpy_asarray_pickle_factory(self):
        class ArrayProxy:
            def __reduce__(self):
                return np.asarray, ([1.0, 2.0],)

        result = ArrayUnpickler(io.BytesIO(pickle.dumps(ArrayProxy()))).load()
        np.testing.assert_equal(result, [1.0, 2.0])

    def test_numpy_pickle_round_trip(self):
        expected = {'x': np.array([[0.0, 1.0], [np.nan, np.inf]]),
                    'text': np.array(['one', 'two'], dtype=object)}
        actual = ArrayUnpickler(io.BytesIO(pickle.dumps(expected))).load()
        np.testing.assert_equal(actual['x'], expected['x'])
        np.testing.assert_equal(actual['text'], expected['text'])

    def test_unapproved_globals_rejected(self):
        reader = ArrayUnpickler(io.BytesIO())
        with self.assertRaises(pickle.UnpicklingError):
            reader.find_class('os', 'system')

    def test_nonfinite_and_zero_rows(self):
        result = numeric_summary(np.array([[0.0, 0.0], [np.nan, np.inf], [-2.0, 3.0]]))
        self.assertEqual(result['nonfinite'], 2)
        self.assertEqual(result['all_zero_feature_rows'], 1)
        self.assertEqual((result['minimum'], result['maximum']), (-2.0, 3.0))

    def test_no_guessed_identifier_parse(self):
        self.assertEqual(parse_id('video$_$7'), ('video', '7'))
        self.assertIsNone(parse_id('ambiguous_identifier'))
        self.assertEqual(canonical_path(Path('prefix/附件2-特征/aligned_50.pkl')),
                         '附件2-特征/aligned_50.pkl')

    def test_tokenizer_checks_all_three_channels(self):
        class FakeTokenizer:
            vocab_size = 200

            def __call__(self, texts, **kwargs):
                return {'input_ids': np.array([[101, 10, 102, 0]]),
                        'attention_mask': np.array([[1, 1, 1, 0]]),
                        'token_type_ids': np.array([[0, 0, 0, 0]])}

        tokens = np.array([[[101, 10, 102, 0], [1, 0, 1, 0], [0, 0, 0, 0]]])
        result = token_audit({'text_bert': tokens, 'raw_text': ['test']}, FakeTokenizer())
        self.assertEqual(result['input_ids_exact'], 1)
        self.assertEqual(result['all_three_channels_exact'], 0)
        self.assertEqual(result['nonzero_ids_under_zero_attention'], 1)


if __name__ == '__main__':
    unittest.main()

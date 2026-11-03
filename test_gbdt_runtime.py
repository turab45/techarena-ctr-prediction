import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import gbdt
from pipeline import load_events


class RuntimeTests(unittest.TestCase):
    def test_chunked_preparation_preserves_split_slates_and_candidate_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [dict(event_id=event, user_id='u', item_id=item,
                         timestamp=100, time_bucket=0, day_type=0,
                         history_item_ids='["old"]', history_timestamps='[1]',
                         clicked=label, rowid=f'r{i}')
                    for i, (event, item, label) in enumerate(
                        [('b', 'x', 1), ('a', 'y', 0), ('b', 'z', 0),
                         ('a', 'w', 1), ('b', 'v', 0)])]
            pd.DataFrame(rows).to_csv(root / 'input.csv', index=False)
            for labelled in (True, False):
                result = load_events(root / 'input.csv', root / f'{labelled}.pkl',
                                     labelled, chunksize=2)
                self.assertEqual(result.event_id.tolist(), ['a', 'b'])
                self.assertEqual(result['items'].tolist(), [['y', 'w'], ['x', 'z', 'v']])
                self.assertEqual(result.history_items.tolist(), [['old'], ['old']])
                self.assertEqual(result.history_timestamps.tolist(), [[1], [1]])
                column = 'labels' if labelled else 'rowids'
                expected = [[0, 1], [1, 0, 0]] if labelled else [['r1', 'r3'], ['r0', 'r2', 'r4']]
                self.assertEqual(result[column].tolist(), expected)

    def test_default_threads_respect_numba_limit(self):
        with patch.object(gbdt.numba_config, 'NUMBA_NUM_THREADS', 14), \
                patch('sys.argv', ['gbdt.py', '--val-only']), \
                patch.object(gbdt, 'set_num_threads') as threads, \
                patch.object(gbdt, 'prepare_data', side_effect=RuntimeError('stop before data')):
            with self.assertRaisesRegex(RuntimeError, 'stop before data'):
                gbdt.main()
            threads.assert_called_once_with(14)


if __name__ == '__main__':
    unittest.main()

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import cmd_blend


class BlendTests(unittest.TestCase):
    def test_aligns_row_ids_and_blends_ranks_instead_of_probability_scales(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({'rowid': ['x', 'y', 'z'], 'predicted_probability': [10, 30, 20]}).to_csv(root/'a.csv', index=False)
            pd.DataFrame({'rowid': ['z', 'x', 'y'], 'predicted_probability': [.9, .2, .1]}).to_csv(root/'b.csv', index=False)
            cmd_blend(Namespace(a=root/'a.csv', b=root/'b.csv', out=root/'out.csv', wa=.8))
            output = pd.read_csv(root/'out.csv').set_index('rowid')
            np.testing.assert_allclose(output.loc[['x', 'y', 'z'], 'predicted_probability'],
                                       [.8/3+.2*2/3, .8+.2/3, .8*2/3+.2])

    def test_rejects_missing_duplicate_and_nonfinite_inputs_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({'rowid': ['x', 'y'], 'predicted_probability': [.1, .2]}).to_csv(root/'a.csv', index=False)
            for ids, scores in ((['x', 'z'], [.1, .2]), (['x', 'x'], [.1, .2]),
                                (['x', 'y'], [.1, np.inf])):
                with self.subTest(ids=ids, scores=scores):
                    pd.DataFrame({'rowid': ids, 'predicted_probability': scores}).to_csv(root/'b.csv', index=False)
                    with self.assertRaises(ValueError):
                        cmd_blend(Namespace(a=root/'a.csv', b=root/'b.csv', out=root/'out.csv', wa=.8))
                    self.assertFalse((root/'out.csv').exists())


if __name__ == '__main__':
    unittest.main()

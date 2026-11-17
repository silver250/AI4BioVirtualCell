import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from cellmates.dsd import transition_matrix, visitation_profiles, predict_dsd


class DSDTests(unittest.TestCase):
    def test_path_visits_and_distance(self):
        a = sparse.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        visits = visitation_profiles(transition_matrix(a), [0, 2], steps=2)
        np.testing.assert_allclose(visits, [[1.5, 1, .5], [.5, 1, 1.5]])
        self.assertAlmostEqual(np.abs(visits[0] - visits[1]).sum(), 2)

    def test_equal_neighbors_and_isolate_fallback(self):
        a = sparse.csr_matrix([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        effects = pd.DataFrame({'g': [2., 4.]}, index=['A', 'C'])
        pred, diag, neighbors = predict_dsd(effects, ['B', 'D'], {'A': 0, 'B': 1, 'C': 2, 'D': 3}, a, steps=2)
        self.assertEqual(pred.loc['B', 'g'], 3)
        self.assertEqual(pred.loc['D', 'g'], 3)
        self.assertFalse(diag.loc['B', 'fallback'])
        self.assertTrue(diag.loc['D', 'fallback'])
        self.assertEqual(neighbors.neighbor.tolist(), ['A', 'C'])

    def test_no_overlapping_visits_and_bad_graph(self):
        a = sparse.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        effects = pd.DataFrame({'g': [1.]}, index=['A'])
        _, diag, _ = predict_dsd(effects, ['C'], {'A': 0, 'C': 2}, a, steps=0)
        self.assertTrue(diag.loc['C', 'fallback'])
        with self.assertRaises(ValueError):
            transition_matrix([[0, 1], [0, 0]])


if __name__ == '__main__':
    unittest.main()

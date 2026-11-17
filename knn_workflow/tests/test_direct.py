import unittest
import numpy as np
import pandas as pd
from cellmates.direct import predict_direct_knn


class DirectKNNTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame([[1., 3.], [3., 5.], [9., 11.]],
                                  index=['A', 'B', 'C'], columns=['g1', 'g2'])
        self.edges = pd.DataFrame([('Q', 'A', .8), ('B', 'Q', .9),
                                   ('Q', 'B', .7), ('Q', 'C', .2)],
                                  columns=['gene_a', 'gene_b', 'score'])

    def test_ranking_undirected_duplicates_and_fallback(self):
        r = predict_direct_knn(self.train, ['Q', 'missing'], self.edges, k=2)
        np.testing.assert_allclose(r.predictions.loc['Q'], [2, 4])
        np.testing.assert_allclose(r.predictions.loc['missing'], self.train.mean())
        self.assertEqual(r.neighbors.neighbor.tolist(), ['B', 'A'])
        self.assertEqual(r.diagnostics.loc['Q', 'eligible_neighbors'], 2)
        self.assertTrue(r.diagnostics.loc['missing', 'fallback'])

    def test_partial_neighbors_and_ties(self):
        edges = pd.DataFrame([('Q', 'B', .8), ('Q', 'A', .8)], columns=self.edges.columns)
        r = predict_direct_knn(self.train, ['Q'], edges, k=5)
        self.assertEqual(r.neighbors.neighbor.tolist(), ['A', 'B'])
        self.assertEqual(r.diagnostics.loc['Q', 'neighbors_used'], 2)

    def test_reject_leakage_and_bad_score(self):
        with self.assertRaises(ValueError):
            predict_direct_knn(self.train, ['A'], self.edges)
        bad = self.edges.copy()
        bad.loc[0, 'score'] = 800
        with self.assertRaises(ValueError):
            predict_direct_knn(self.train, ['Q'], bad)


if __name__ == '__main__':
    unittest.main()

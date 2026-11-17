import unittest
import pandas as pd
import numpy as np
from cellmates.baselines import predict_baselines
from cellmates.direct import predict_direct_knn


class BaselineAndWeightTests(unittest.TestCase):
    def test_baseline_vectors(self):
        train=pd.DataFrame([[1,3],[3,5]],index=['A','B'],columns=['x','y'])
        r=predict_baselines(train,['Q'])
        np.testing.assert_allclose(r['no_change'].loc['Q'],[0,0])
        np.testing.assert_allclose(r['mean_effect'].loc['Q'],[2,4])

    def test_weighting_uses_same_neighbors(self):
        train=pd.DataFrame([[0.],[10.]],index=['A','B'],columns=['x'])
        edges=pd.DataFrame([('Q','A',.8),('Q','B',.4)],columns=['gene_a','gene_b','score'])
        equal=predict_direct_knn(train,['Q'],edges,k=2,weighting='equal')
        weighted=predict_direct_knn(train,['Q'],edges,k=2,weighting='score')
        self.assertEqual(equal.neighbors.neighbor.tolist(),weighted.neighbors.neighbor.tolist())
        self.assertAlmostEqual(equal.predictions.loc['Q','x'],5)
        self.assertAlmostEqual(weighted.predictions.loc['Q','x'],10/3)
        self.assertAlmostEqual(weighted.neighbors.weight.sum(),1)

if __name__=='__main__': unittest.main()

import unittest
import numpy as np
from cellmates.neighbor_cells import mix_neighbor_cells


class NeighborCellTests(unittest.TestCase):
    def test_equal_and_weighted_allocations(self):
        pools={'A':np.ones((8,2)),'B':np.full((8,2),2.)}
        cells,counts=mix_neighbor_cells(pools,['A','B'],[2,1],6,np.random.default_rng(0))
        self.assertEqual(cells.shape,(6,2))
        self.assertEqual(counts,{'A':4,'B':2})
        self.assertEqual((cells[:,0]==1).sum(),4)
    def test_reject_pseudoreplication(self):
        with self.assertRaises(ValueError):
            mix_neighbor_cells({'A':np.ones((2,1))},['A'],[1],3,np.random.default_rng(0))

if __name__=='__main__': unittest.main()

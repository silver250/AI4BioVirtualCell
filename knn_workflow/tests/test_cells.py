import unittest
import numpy as np
from cellmates.cells import generate_cells


class CellGenerationTests(unittest.TestCase):
    def test_zero_and_negative_effect(self):
        c=np.array([[0.,1.],[2.,3.]])
        u=np.full(c.shape,.25)
        y,_=generate_cells(c,np.zeros(2),u,u,'hurdle')
        np.testing.assert_allclose(y,c)
        y,_=generate_cells(c,np.array([-0.5,-1.]),u,u,'hurdle')
        self.assertTrue((y>=0).all())
        np.testing.assert_allclose(y.mean(axis=0),[.5,1.])

    def test_no_positive_template_cannot_activate(self):
        c=np.zeros((3,1)); u=np.full(c.shape,.2)
        y,d=generate_cells(c,np.array([.1]),u,u,'hurdle')
        self.assertTrue((y==0).all())
        self.assertEqual(d['unreachable_fraction_genes'],1)

    def test_seeded_increase_and_additive_clipping(self):
        c=np.array([[0.],[0.],[1.],[1.]])
        u=np.array([[.1],[.9],[.1],[.9]])
        out,_=generate_cells(c,np.array([.25]),u,u,'hurdle')
        np.testing.assert_allclose(out[:,0],[1,0,1,1])
        out,diag=generate_cells(c,np.array([-.2]),u,u,'additive')
        np.testing.assert_allclose(out[:,0],[0,0,.8,.8])
        self.assertEqual(diag['fraction_negative_before_clip'],.5)

if __name__=='__main__':
    unittest.main()

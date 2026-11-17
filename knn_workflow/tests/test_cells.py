import unittest
import numpy as np
from cellmates.cells import generate_cells


class CellGenerationTests(unittest.TestCase):
    def test_zero_and_negative_effect(self):
        c = np.array([[0.0, 1.0], [2.0, 3.0]])
        u = np.full(c.shape, 0.25)
        y, _ = generate_cells(c, np.zeros(2), u, u, "hurdle")
        np.testing.assert_allclose(y, c)
        y, _ = generate_cells(c, np.array([-0.5, -1.0]), u, u, "hurdle")
        self.assertTrue((y >= 0).all())
        np.testing.assert_allclose(y.mean(axis=0), [0.5, 1.0])

    def test_no_positive_template_cannot_activate(self):
        c = np.zeros((3, 1))
        u = np.full(c.shape, 0.2)
        y, d = generate_cells(c, np.array([0.1]), u, u, "hurdle")
        self.assertTrue((y == 0).all())
        self.assertEqual(d["unreachable_fraction_genes"], 1)

    def test_seeded_increase_and_additive_clipping(self):
        c = np.array([[0.0], [0.0], [1.0], [1.0]])
        u = np.array([[0.1], [0.9], [0.1], [0.9]])
        out, _ = generate_cells(c, np.array([0.25]), u, u, "hurdle")
        np.testing.assert_allclose(out[:, 0], [1, 0, 1, 1])
        out, diag = generate_cells(c, np.array([-0.2]), u, u, "additive")
        np.testing.assert_allclose(out[:, 0], [0, 0, 0.8, 0.8])
        self.assertEqual(diag["fraction_negative_before_clip"], 0.5)


if __name__ == "__main__":
    unittest.main()

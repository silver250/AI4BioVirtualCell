import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from scipy import sparse
from cellmates.preprocessing import target_means


class EffectSummaryTests(unittest.TestCase):
    def test_streamed_normalization_and_control_subtraction(self):
        # Equal totals make the expected log-normalized control difference clear.
        matrix = sparse.csr_matrix(np.array([[1, 1], [2, 0], [0, 2]], dtype=float))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sample.h5ad"
            with h5py.File(path, "w") as f:
                x = f.create_group("X")
                x.attrs["shape"] = matrix.shape
                for name in ("data", "indices", "indptr"):
                    x.create_dataset(name, data=getattr(matrix, name))
                f.create_dataset("var/_index", data=np.array(["g1", "g2"], dtype="S"))
                f.create_dataset(
                    "obs/target_gene/categories",
                    data=np.array(["A", "non-targeting"], dtype="S"),
                )
                f.create_dataset("obs/target_gene/codes", data=[1, 0, 0])
            effects, counts, n = target_means(path, chunk_size=1, library_size=2)
        expected = np.log1p(2) / 2 - np.log1p(1)
        np.testing.assert_allclose(effects.loc["A"], [expected, expected])
        self.assertEqual((counts["A"], counts["non-targeting"], n), (2, 1, 3))


if __name__ == "__main__":
    unittest.main()

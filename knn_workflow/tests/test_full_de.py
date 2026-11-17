"""Check the memory-bounded rank calculation against cell-eval2/Scanpy."""

import unittest
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from cell_eval2.de_compute import compute_de
from scipy.stats import false_discovery_control

from cellmates.full_de import control_tie_sums, wilcoxon_pvalues


class FullDETests(unittest.TestCase):
    def test_matches_scanpy_with_ties_and_zero_genes(self):
        rng = np.random.default_rng(42)
        # Many repeated values and one all-zero gene exercise tie handling.
        controls = np.log1p(rng.poisson(1.4, size=(45, 8))).astype(np.float32)
        target = np.log1p(rng.poisson(1.7, size=(40, 8))).astype(np.float32)
        controls[:, 0] = 0
        target[:, 0] = 0
        genes = [f"gene_{i}" for i in range(8)]
        labels = ["non-targeting"] * len(controls) + ["query"] * len(target)
        data = ad.AnnData(
            np.vstack([controls, target]),
            obs=pd.DataFrame(
                {"target_gene": labels}, index=[f"cell_{i}" for i in range(len(labels))]
            ),
            var=pd.DataFrame(index=genes),
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            reference = (
                compute_de(
                    data,
                    backend="scanpy",
                    groupby="target_gene",
                    reference="non-targeting",
                    input_type="lognorm",
                    target_sum=10_000,
                    mean_calc="arithmetic",
                    epsilon=1e-9,
                    fdr_scope="per_pert",
                    filter_gene_min_cpm_cell=5,
                    threads=1,
                )
                .to_pandas()
                .set_index("feature")
                .loc[genes]
            )
        sorted_controls = np.ascontiguousarray(np.sort(controls, axis=0).T)
        p_values = wilcoxon_pvalues(
            sorted_controls, target, control_tie_sums(sorted_controls)
        )
        np.testing.assert_allclose(p_values, reference.p_value.to_numpy(), atol=1e-12)
        np.testing.assert_allclose(
            false_discovery_control(p_values), reference.p_adj.to_numpy(), atol=1e-12
        )
        self.assertEqual(p_values[0], 1.0)


if __name__ == "__main__":
    unittest.main()

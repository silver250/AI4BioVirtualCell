"""Cell-eval2 expression/PDS agree when group mean profiles are precomputed."""

import unittest
import anndata as ad
import numpy as np
import pandas as pd
from cell_eval2 import compute_metrics
from cell_eval2 import EvalConfig


class GroupMeanScoringTest(unittest.TestCase):
    def test_group_means_match_full_cells(self):
        rng = np.random.default_rng(7)
        targets = ["non-targeting", "A", "B"]
        genes = ["A", "B", "C", "D"]
        y = rng.uniform(0.05, 2.0, (12, 4))
        p = rng.uniform(0.05, 2.0, (12, 4))
        labels = [t for t in targets for _ in range(4)]
        cfg = EvalConfig(
            metrics=["expr_mae", "pds_l1"],
            pert_col="target_gene",
            control="non-targeting",
            input_type="lognorm",
            target_sum=10000,
            device="cpu",
        )

        def make(x, lab):
            return ad.AnnData(
                x,
                obs=pd.DataFrame(
                    {"target_gene": lab}, index=[f"i{i}" for i in range(len(lab))]
                ),
                var=pd.DataFrame(index=genes),
            )

        real = make(y, labels)
        pred = make(p, labels)
        group = lambda x: np.vstack(
            [x[i * 4 : (i + 1) * 4].mean(axis=0) for i in range(3)]
        )
        full = compute_metrics(pred, real, config=cfg).sort(["perturbation", "metric"])
        summary = compute_metrics(
            make(group(p), targets), make(group(y), targets), config=cfg
        ).sort(["perturbation", "metric"])
        self.assertEqual(full["metric"].to_list(), summary["metric"].to_list())
        np.testing.assert_allclose(
            full["value"].to_numpy(), summary["value"].to_numpy(), atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()

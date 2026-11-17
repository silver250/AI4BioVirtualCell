"""Verify the installed Scanpy handling of all-zero and different genes."""
import unittest
import warnings
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


class ScanpyTiesTest(unittest.TestCase):
    def test_all_zero_gene_is_neutral_and_changed_gene_detected(self):
        x=np.zeros((60,2),dtype=np.float32)
        x[:30,1]=2.1
        a=ad.AnnData(x,obs=pd.DataFrame({'target':['target']*30+['control']*30},index=[f'c{i}' for i in range(60)]),var=pd.DataFrame(index=['always_zero','changed']))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            sc.tl.rank_genes_groups(a,groupby='target',reference='control',method='wilcoxon',tie_correct=True,n_genes=2)
        out=a.uns['rank_genes_groups']
        names=list(out['names']['target'])
        p=out['pvals']['target']
        self.assertEqual(float(p[names.index('always_zero')]),1.0)
        self.assertLess(float(p[names.index('changed')]),.05)
        self.assertTrue(any('invalid value encountered in divide' in str(w.message) for w in captured))

if __name__=='__main__': unittest.main()

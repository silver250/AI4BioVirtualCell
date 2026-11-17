"""Estimate target response vectors from raw training counts."""
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


def target_means(path, chunk_size=512, library_size=10000, include_means=False):
    with h5py.File(path, 'r') as f:
        shape = tuple(f['X'].attrs['shape'])
        genes = f['var/_index'].asstr()[:]
        labels = f['obs/target_gene/categories'].asstr()[:]
        codes = f['obs/target_gene/codes'][:].astype(int)
        if len(genes) != len(set(genes)) or (codes < 0).any():
            raise ValueError('Expected unique gene names and nonmissing target labels')
        sums = np.zeros((len(labels), shape[1]), dtype=np.float64)
        counts = np.bincount(codes, minlength=len(labels))
        zero_cells = 0
        for start in range(0, shape[0], chunk_size):
            stop = min(start + chunk_size, shape[0])
            pointers = f['X/indptr'][start:stop + 1].astype(np.int64)
            values = f['X/data'][pointers[0]:pointers[-1]].astype(np.float64)
            indices = f['X/indices'][pointers[0]:pointers[-1]]
            if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
                raise ValueError('Expected finite nonnegative integer raw counts in X')
            x = sparse.csr_matrix((values, indices, pointers - pointers[0]), shape=(stop-start, shape[1]))
            totals = np.asarray(x.sum(axis=1)).ravel()
            zero_cells += int((totals == 0).sum())
            scale = np.divide(library_size, totals, out=np.zeros_like(totals), where=totals > 0)
            x.data *= np.repeat(scale, np.diff(x.indptr))
            np.log1p(x.data, out=x.data)
            selector = sparse.csr_matrix((np.ones(stop-start), (codes[start:stop], np.arange(stop-start))), shape=(len(labels), stop-start))
            sums += (selector @ x).toarray()
            if start % (512 * 50) == 0:
                print(f'{path.name}: {stop}/{shape[0]} cells', flush=True)
        if zero_cells or (counts == 0).any():
            raise ValueError(f'Found {zero_cells} zero-library cells or empty target categories')
        means = pd.DataFrame(sums / counts[:, None], index=labels, columns=genes)
        if 'non-targeting' not in means.index:
            raise ValueError('Missing non-targeting control')
        effects = means.subtract(means.loc['non-targeting'], axis='columns').drop(index='non-targeting')
        result = (effects, pd.Series(counts, index=labels, name='cells'), int(shape[0]))
        return (*result, means) if include_means else result

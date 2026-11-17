"""Prepare train/validation STRING edges and DSD neighbors without test responses."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from cellmates.dsd import predict_dsd


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('train-effects','validation','string-info','string-links','string-detailed','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    train=pd.read_parquet(a.train_effects)
    validation=ad.read_h5ad(a.validation,backed='r')
    try:targets=sorted(set(validation.obs['target_gene'].astype(str)) - {'non-targeting'})
    finally:validation.file.close()
    if not targets or set(targets)&set(train.index):raise ValueError('invalid target labels or overlap')
    all_targets=set(targets)|set(train.index)
    mapping=defaultdict(list);ids=[]
    with gzip.open(a.string_info,'rt') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            ids.append(r['#string_protein_id'])
            if r['preferred_name'] in all_targets:mapping[r['preferred_name']].append(r['#string_protein_id'])
    if len(ids)!=len(set(ids)) or any(len(mapping[t])!=1 for t in all_targets):
        raise ValueError('STRING IDs must be unique; every target needs one exact preferred-name match')
    ids.sort();idx={v:i for i,v in enumerate(ids)}
    gene_to_id={t:mapping[t][0] for t in all_targets};id_to_gene={v:k for k,v in gene_to_id.items()}
    if len(id_to_gene)!=len(gene_to_id):raise ValueError('multiple targets map to one protein')
    pd.DataFrame([{'target':t,'string_id':gene_to_id[t], 'split':'train' if t in train.index else 'validation'} for t in sorted(all_targets)]).to_csv(out/'target_mapping.csv',index=False)
    rows=[];cols=[];weights=[]
    for chunk in pd.read_csv(a.string_links,sep=r'\s+',chunksize=500_000):
        keep=chunk[chunk.combined_score>=400]
        left=keep.protein1.map(idx);right=keep.protein2.map(idx)
        if left.isna().any() or right.isna().any():raise ValueError('network protein absent from info')
        rows.append(left.to_numpy(dtype=np.int32));cols.append(right.to_numpy(dtype=np.int32));weights.append(keep.combined_score.to_numpy(dtype=float)/1000)
    frame=pd.DataFrame({'a':np.concatenate(rows),'b':np.concatenate(cols),'score':np.concatenate(weights)})
    frame=frame.groupby(['a','b'],as_index=False,sort=False).score.max()
    graph=sparse.csr_matrix((frame.score.to_numpy(),(frame.a.to_numpy(),frame.b.to_numpy())),shape=(len(ids),len(ids)))
    graph=graph.maximum(graph.T).tocsr();graph.setdiag(0);graph.eliminate_zeros()
    node_by_gene={t:idx[s] for t,s in gene_to_id.items()}
    # Pull only target-to-target edges from the graph; rows are undirected once.
    target_names=sorted(all_targets);target_nodes=[node_by_gene[t] for t in target_names]
    sub=sparse.triu(graph[target_nodes,:][:,target_nodes],k=1).tocoo()
    edges=pd.DataFrame({'gene_a':[target_names[i] for i in sub.row], 'gene_b':[target_names[i] for i in sub.col], 'score':sub.data})
    edges.to_csv(out/'target_edges.csv',index=False)
    detailed={}
    with gzip.open(a.string_detailed,'rt') as f:
        header=next(f).split();ix={n:header.index(n) for n in ['protein1','protein2','experimental','database','combined_score']}
        for line in f:
            c=line.split();x,y=c[ix['protein1']],c[ix['protein2']]
            if x not in id_to_gene or y not in id_to_gene or x==y:continue
            exp,db,score=(int(c[ix[n]]) for n in ('experimental','database','combined_score'))
            if score<400 or not (exp>0 or db>0):continue
            pair=tuple(sorted((id_to_gene[x],id_to_gene[y])));old=detailed.get(pair)
            if old is not None and old!=(exp,db,score):raise ValueError('inconsistent directed evidence duplicates')
            detailed[pair]=(exp,db,score)
    ev=pd.DataFrame([{'gene_a':x,'gene_b':y,'experimental':e/1000,'database':d/1000,'score':s/1000}
                     for (x,y),(e,d,s) in sorted(detailed.items())],columns=['gene_a','gene_b','experimental','database','score'])
    ev.to_csv(out/'evidence_edges.csv',index=False)
    # One dummy response is sufficient: neighbor identities depend only on the graph.
    dummy=pd.DataFrame({'placeholder':np.zeros(len(train))},index=train.index)
    _,diag,neighbors=predict_dsd(dummy,targets,node_by_gene,graph,k=10,steps=6)
    neighbors.to_csv(out/'dsd_neighbors.csv',index=False);diag.to_csv(out/'dsd_coverage.csv')
    paths={'train_effects':a.train_effects,'validation':a.validation,'string_info':a.string_info,'string_links':a.string_links,'string_detailed':a.string_detailed}
    meta={'sha256':{k:sha(Path(v)) for k,v in paths.items()},'release':'STRING human v12.0 functional combined',
          'cutoff':.4,'dsd_steps':6,'k':10,'nodes':len(ids),'edges':graph.nnz//2,
          'evidence_filter':'combined >= 0.4 AND (experimental > 0 OR database > 0)'}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':main()

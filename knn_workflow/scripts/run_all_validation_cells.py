"""Score expression/PDS on all validation cells through streamed group means.

DE cannot be reconstructed from group means and is deliberately excluded.
"""
import argparse
import json
import time
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from cellmates.baselines import predict_baselines
from cellmates.direct import predict_direct_knn
from cellmates.evaluation import score_validation
from cellmates.preprocessing import target_means
from run_validation import read_cells, dsd_effects, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('train','validation','train-effects','edges','evidence-edges','dsd-neighbors','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--predicted-cells',type=int,default=500)
    p.add_argument('--seed',type=int,default=420)
    a=p.parse_args()
    if a.predicted_cells<1:p.error('--predicted-cells must be positive')
    started=time.perf_counter();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    paths={k:Path(getattr(a,k)) for k in ('train','validation','train_effects','edges','evidence_edges','dsd_neighbors')}
    if paths['train'].resolve()==paths['validation'].resolve():raise ValueError('train and validation paths must differ')
    train=pd.read_parquet(paths['train_effects'])
    _,counts,n_cells,means=target_means(paths['validation'],include_means=True)
    targets=sorted(set(means.index)-{'non-targeting'})
    if len(targets)<2 or set(targets)&set(train.index):raise ValueError('invalid or overlapping targets')
    controls,genes,ids=read_cells(paths['train'],['non-targeting'],a.predicted_cells,np.random.default_rng(a.seed))
    if genes!=means.columns.tolist() or genes!=train.columns.tolist():raise ValueError('gene names and order must match')
    edge=pd.read_csv(paths['edges']);evidence=pd.read_csv(paths['evidence_edges']);dsd=pd.read_csv(paths['dsd_neighbors'])
    models=predict_baselines(train,targets)
    for name,graph,weight in [('direct_equal',edge,'equal'),('direct_score',edge,'score'),('evidence_equal',evidence,'equal')]:
        result=predict_direct_knn(train,targets,graph,k=5,min_score=.4,weighting=weight)
        models[name]=result.predictions.loc[targets]
        result.diagnostics.to_csv(out/f'{name}_coverage.csv');result.neighbors.to_csv(out/f'{name}_neighbors.csv',index=False)
    models['dsd_equal']=dsd_effects(train,targets,dsd,5)
    models['dsd_equal_k10']=dsd_effects(train,targets,dsd,10)
    labels=['non-targeting']+targets
    obs=pd.DataFrame({'target_gene':labels},index=[f'group_{i}' for i in range(len(labels))]);var=pd.DataFrame(index=train.columns)
    observed=ad.AnnData(means.loc[labels].to_numpy(dtype=np.float64),obs=obs.copy(),var=var.copy())
    meta={'status':'running','analysis':'all real validation cells reduced to per-target mean log-normalized expression',
          'note':'Group means are sufficient for expr_mae and pds_l1 in this configuration; DE cannot be computed from group means.',
          'validation_cells':n_cells,'validation_cells_per_target':counts.to_dict(),'n_measured_genes':len(genes),
          'predicted_cells_per_target':a.predicted_cells,'prediction_population':'sampled training controls shifted by predicted log-scale effect and clipped at zero',
          'seed':a.seed,'training_control_barcodes':ids,'sha256':{k:sha256(v) for k,v in paths.items()},'models':{}}
    c=controls['non-targeting']
    for name,effects in models.items():
        t0=time.perf_counter();predicted_means=[c.mean(axis=0,dtype=np.float64)];clipped=0
        for target in targets:
            x=c+effects.loc[target].to_numpy(dtype=np.float32)
            clipped+=int((x<0).sum());predicted_means.append(np.maximum(x,0).mean(axis=0,dtype=np.float64))
        predicted=ad.AnnData(np.vstack(predicted_means),obs=obs.copy(),var=var.copy())
        scores,aggregate,config=score_validation(predicted,observed,include_de=False)
        scores.write_csv(out/f'{name}_per_target.csv');aggregate.write_csv(out/f'{name}_aggregate.csv')
        meta['models'][name]={'seconds':time.perf_counter()-t0,'evaluation':config,
                              'fraction_clipped':clipped/(len(targets)*a.predicted_cells*len(genes)),
                              'nonfinite_metrics':int((~np.isfinite(scores['value'].to_numpy())).sum())}
        (out/'metadata.json').write_text(json.dumps(meta,indent=2,default=str)+'\n')
        print('Scored',name,flush=True)
    meta['status']='complete';meta['seconds']=time.perf_counter()-started
    (out/'metadata.json').write_text(json.dumps(meta,indent=2,default=str)+'\n')

if __name__=='__main__':main()

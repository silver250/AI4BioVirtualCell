"""Estimate training-target effects from a raw training H5AD, never validation or test."""
import argparse
import hashlib
import json
from pathlib import Path
from cellmates.preprocessing import target_means


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',required=True,help='training H5AD')
    parser.add_argument('--output',required=True,help='new output directory')
    args=parser.parse_args();src=Path(args.train);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    effects, counts, cells=target_means(src)
    effects.to_parquet(out/'train_effects.parquet')
    counts.rename_axis('target').to_csv(out/'train_cell_counts.csv')
    with src.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    (out/'metadata.json').write_text(json.dumps({'source':str(src.resolve()),'sha256':digest,'cells':cells,
                                                'normalization':'per-cell count total 10000, log1p, then target mean minus control mean'},indent=2)+'\n')

if __name__=='__main__':main()

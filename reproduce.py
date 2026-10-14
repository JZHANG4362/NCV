#!/usr/bin/env python3
"""Rebuild the paper's outputs; start with `python reproduce.py tables`."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from build_tables import build

def environment(output):
    env=os.environ.copy()
    for directory in ['matplotlib','numba']:
        (output/'.cache'/directory).mkdir(parents=True,exist_ok=True)
    env['MPLBACKEND']='Agg'
    env['MPLCONFIGDIR']=str(output/'.cache/matplotlib')
    env['XDG_CACHE_HOME']=str(output/'.cache')
    env['NUMBA_CACHE_DIR']=str(output/'.cache/numba')
    env['PYTHONHASHSEED']='0'
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
        env[name]='1'
    env.pop('JOB_TAG',None)
    return env

def run(script,args,cwd,output):
    logs=output/'logs';logs.mkdir(parents=True,exist_ok=True)
    log=logs/(script.stem+'.log')
    print(f'Running {script.name} (log: {log})',flush=True)
    with log.open('w') as f:
        subprocess.run([sys.executable,str(script),*args],cwd=cwd,env=environment(output),stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['tables','cached','archive'])
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    parser.add_argument('--results-root',type=Path,default=ROOT,help='root of an isolated full run; defaults to supplied caches')
    parser.add_argument('--allow-changed-results',action='store_true')
    args=parser.parse_args();out=args.output.resolve();src=args.results_root.resolve()
    start=time.monotonic()
    build(out,not args.allow_changed_results,src)
    provenance=json.loads((ROOT/'reference/figure_provenance.json').read_text())
    if args.mode=='cached':
        # This notebook uses two numerical caches and recomputes only the
        # representative data set's selected-r panel (not the 100 repetitions).
        run(ROOT/'scripts/run_notebook.py',[str(src/'section_4_1_gaussian_simulation/combined_boxplot_figure.ipynb')],src,out)
        jobs=[('section_4_1_gaussian_simulation','gaussian_multimetric_landscape_figure.py'),
              ('section_4_5_sensitivity/summary_plots','tau_khat_window_sensitivity.py'),
              ('section_4_6_local_embeddings/other_indices','generate_manifold_bar_summary_tex.py')]
        for folder,name in jobs:run(src/folder/name,[],src/folder,out)
        for item in provenance:
            source=src/item['packaged_source']
            target=out/item['output'];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    elif args.mode=='archive':
        if src!=ROOT:parser.error('archive mode applies only to the supplied manuscript snapshot')
        for item in provenance:
            target=out/item['output'];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/'reference'/item['output'],target)
    if args.mode!='tables':
        missing=[p['output'] for p in provenance if not (out/p['output']).is_file()]
        if missing:raise SystemExit(f'Missing figures: {missing}')
        print(f'Collected {len(provenance)} figures.')
    report={'mode':args.mode,'elapsed_seconds':round(time.monotonic()-start,2),
            'python':sys.version,'tables_verified_against_snapshot':not args.allow_changed_results,
            'figure_mode':{'tables':'not requested','cached':'redrawn; the single Case 1 selected-r panel is recalculated','archive':'copied verbatim from the manuscript snapshot'}[args.mode]}
    (out/'run_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Done: {out}')

if __name__=='__main__':main()

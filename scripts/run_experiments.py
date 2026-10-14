#!/usr/bin/env python3
"""Plan or execute fresh experiments in an isolated copy, preserving supplied caches."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reproduce import environment

ALT='section_4_6_local_embeddings/other_indices'
SIL='section_4_6_local_embeddings/silhouette'
PACKAGES=[
    'section_4_1_gaussian_simulation',
    'section_4_2_simulated_benchmarks',
    'section_4_3_nonconvex_benchmarks',
    'section_4_4_labeled_benchmarks',
    'section_4_5_sensitivity',
    'section_4_6_local_embeddings',
]
CMS=['KMeans','Spectral','Agglomerative']

def jobs(group):
    plans={
        'case1': [('section_4_1_gaussian_simulation',n,[]) for n in ['gaussian_simulation.ipynb','gaussian_simulation_boxplot.ipynb','combined_boxplot_figure.ipynb','gaussian_multimetric_landscape_figure.py']],
        'case2': [('section_4_2_simulated_benchmarks','partA_type2_grid_comparison.py',[cm]) for cm in CMS],
        'case3': [('section_4_3_nonconvex_benchmarks','partB_fixed_datasets_comparison.py',[cm]) for cm in CMS],
        # Sequential execution avoids shared-checkpoint write races.
        'case4': [('section_4_4_labeled_benchmarks','Benchmark_data_real.py',[cm]) for cm in CMS],
        'sensitivity': [('section_4_5_sensitivity/'+d,'rstar_selection_landscape.py',[]) for d in ['simulated_data','nonconvex_data','labeled_data']]+[('section_4_5_sensitivity/summary_plots','tau_khat_window_sensitivity.py',[])],
        'embeddings': [(p,f'manifold_tsne_umap_case{c}.py',[]) for p in [SIL,ALT] for c in [2,3,4]],
        'smoke': [('section_4_4_labeled_benchmarks','Benchmark_data_real.py',['KMeans','Iris']),
                  (SIL,'manifold_tsne_umap_case4.py',['KMeans','Iris']),
                  (ALT,'manifold_tsne_umap_case4.py',['KMeans','all','Iris'])],
    }
    if group=='all':return [job for k,v in plans.items() if k!='smoke' for job in v]
    return plans[group]

def stage(dest):
    dest.mkdir(parents=True,exist_ok=False)
    for name in PACKAGES:
        for p in (ROOT/name).rglob('*'):
            if p.is_file() and (p.suffix in ['.py','.ipynb'] or '/data/raw_datasets/' in p.as_posix()):
                q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    (dest/'logs').mkdir()
    (dest/'RUN_STATE.json').write_text(json.dumps({'group':None,'completed':[]},indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group',choices=['case1','case2','case3','case4','sensitivity','embeddings','all','smoke'])
    parser.add_argument('--workdir',type=Path,default=None)
    parser.add_argument('--execute',action='store_true',help='actually run; otherwise print the plan')
    parser.add_argument('--resume',action='store_true',help='use an existing run and skip successful jobs; interrupted jobs restart from the beginning')
    args=parser.parse_args();dest=(args.workdir or ROOT/'runs'/args.group).resolve()
    plan=jobs(args.group)
    for folder,name,options in plan:
        print(f'[{folder}] '+shlex.join([Path(sys.executable).name,name,*options]))
    print(f'Run directory: {dest}',flush=True)
    if not args.execute:
        print('Plan only. Add --execute to run. Full jobs can be expensive; use smoke first.');return
    if args.resume:
        state=json.loads((dest/'RUN_STATE.json').read_text())
        if state['group']!=args.group:parser.error('Use the same group when resuming a run')
    else:
        if dest.exists():parser.error('Work directory already exists; choose a new path or use --resume')
        stage(dest);state={'group':args.group,'completed':[]}
    state_path=dest/'RUN_STATE.json'
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    for number,(folder,name,options) in enumerate(plan,1):
        key=f'{folder}/{name} '+' '.join(options)
        if key in state['completed']:
            print(f'Skipping successful job: {key}',flush=True);continue
        path=dest/folder/name
        command=([sys.executable,str(ROOT/'scripts/run_notebook.py'),str(path)] if path.suffix=='.ipynb' else [sys.executable,str(path),*options])
        log=dest/'logs'/f'{number:02d}_{path.stem}.log'
        print(f'Running {key}; log: {log}',flush=True)
        start=time.monotonic()
        with log.open('w') as f:
            subprocess.run(command,cwd=path.parent,env=environment(dest),stdout=f,stderr=subprocess.STDOUT,check=True)
        state['completed'].append(key)
        state.setdefault('elapsed_seconds',{})[key]=round(time.monotonic()-start,2)
        state_path.write_text(json.dumps(state,indent=2)+'\n')
    print(f'Completed {args.group}. Results remain in {dest}.')

if __name__=='__main__':main()

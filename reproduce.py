#!/usr/bin/env python3
"""Rebuild the paper's outputs; start with `python reproduce.py tables`."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from build_tables import build, CASE2_CASES, CASE3_K, CASE4_K

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

def figure_paths(src):
    """Map generated figures to the paths used by the manuscript."""
    gaussian = src/'section_4_1_gaussian_simulation/figures/combined'
    names = ['gaussian_combined_landscape_boxplot.pdf'] + [
        f'gaussian_multimetric_{method}_landscape.pdf'
        for method in ['kmeans', 'spectral', 'agglomerative']
    ]
    files = [(gaussian/name, Path('figures')/name) for name in names]
    sensitivity = src/'section_4_5_sensitivity/summary_plots/figures'
    files.extend((sensitivity/f'{parameter}_sensitivity_hard.pdf',
                  Path('figures')/f'rstar_{parameter}_sensitivity.pdf')
                 for parameter in ['tau', 'khat', 'b'])
    datasets = {
        2: [f'k{k}_D{d}_{c}' for k, d, c in CASE2_CASES],
        3: list(CASE3_K),
        4: list(CASE4_K),
    }
    embeddings = src/'section_4_6_local_embeddings/other_indices/figures/bar_summary'
    for case, names in datasets.items():
        for name in names:
            relative = Path(f'case{case}')/f'{name}.pdf'
            files.append((embeddings/relative, Path('figures/local_embedding_bar_plots')/relative))
    return files

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['tables','cached'],help='tables: generate LaTeX tables; cached: also redraw PDF figures')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    parser.add_argument('--results-root',type=Path,default=ROOT,help='root of an isolated full run; defaults to supplied caches')
    args=parser.parse_args();out=args.output.resolve();src=args.results_root.resolve()
    build(out,src)
    if args.mode=='cached':
        # This notebook uses two numerical caches and recomputes only the
        # representative data set's selected-r panel (not the 100 repetitions).
        run(ROOT/'scripts/run_notebook.py',[str(src/'section_4_1_gaussian_simulation/combined_boxplot_figure.ipynb')],src,out)
        jobs=[('section_4_1_gaussian_simulation','gaussian_multimetric_landscape_figure.py'),
              ('section_4_5_sensitivity/summary_plots','tau_khat_window_sensitivity.py'),
              ('section_4_6_local_embeddings/other_indices','generate_manifold_bar_summary_tex.py')]
        for folder,name in jobs:run(src/folder/name,[],src/folder,out)
        figures = figure_paths(src)
        for source, relative_output in figures:
            target=out/relative_output;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
        print(f'Collected {len(figures)} figures.')
    print(f'Done: {out}')

if __name__=='__main__':main()

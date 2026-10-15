#!/usr/bin/env python3
"""Reproduce main text Tables 1--5 and Figure 2, and Supplement Tables S4--S87
and Figures S1--S37. Run python reproduce.py cached from the repository root.

Tables and most figures are rebuilt from saved numerical results. The
Figure 2 notebook also recomputes its representative selected-r panel;
the 100-replication boxplot uses its cache. Fresh experiments are run by
scripts/run_experiments.py, and --results-root selects their outputs."""
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
    """Configure subprocesses for reproducible, headless experiment runs.
    
    Keep plotting and compilation caches inside the output directory and
    limit numerical-library threading. Explicit dataset/algorithm seeds
    remain in the experiment scripts."""
    env=os.environ.copy()
    # Place runtime caches under this run rather than mixing them with published results.
    for directory in ['matplotlib','numba']:
        (output/'.cache'/directory).mkdir(parents=True,exist_ok=True)
    env['MPLBACKEND']='Agg'
    env['MPLCONFIGDIR']=str(output/'.cache/matplotlib')
    env['XDG_CACHE_HOME']=str(output/'.cache')
    env['NUMBA_CACHE_DIR']=str(output/'.cache/numba')
    env['PYTHONHASHSEED']='0'
    # Limit BLAS/OpenMP/Numba threads for these subprocesses; statistical seeds
    # remain explicit in the generating and clustering code.
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
        env[name]='1'
    env.pop('JOB_TAG',None)
    return env

def run(script,args,cwd,output):
    """Run one output-generation script and capture its console output in a log."""
    logs=output/'logs';logs.mkdir(parents=True,exist_ok=True)
    log=logs/(script.stem+'.log')
    print(f'Running {script.name} (log: {log})',flush=True)
    # Capture stdout and stderr together; check=True stops the workflow if
    # a figure generator fails instead of continuing with an incomplete output set.
    with log.open('w') as f:
        subprocess.run([sys.executable,str(script),*args],cwd=cwd,env=environment(output),stdout=f,stderr=subprocess.STDOUT,check=True)

def figure_paths(src):
    """Map generated files to the collected main text and Supplement figures.
    
    The Gaussian outputs supply Figure 2 and Figures S1--S3; sensitivity
    supplies Figures S4--S6; dataset-wise embedding bars supply S7--S37.
    Return source/destination pairs for copying, without modifying plots."""
    gaussian = src/'section_4_1_gaussian_simulation/figures/combined'
    # First collect main text Figure 2 and the three algorithm-specific
    # Gaussian comparisons for Supplement Figures S1--S3.
    names = ['gaussian_combined_landscape_boxplot.pdf'] + [
        f'gaussian_multimetric_{method}_landscape.pdf'
        for method in ['kmeans', 'spectral', 'agglomerative']
    ]
    files = [(gaussian/name, Path('figures')/name) for name in names]
    # The tau, K_hat, and buffer sweeps map to Figures S4, S5, and S6.
    sensitivity = src/'section_4_5_sensitivity/summary_plots/figures'
    files.extend((sensitivity/f'{parameter}_sensitivity_hard.pdf',
                  Path('figures')/f'rstar_{parameter}_sensitivity.pdf')
                 for parameter in ['tau', 'khat', 'b'])
    datasets = {
        2: [f'k{k}_D{d}_{c}' for k, d, c in CASE2_CASES],
        3: list(CASE3_K),
        4: list(CASE4_K),
    }
    # Keep dataset order within each case for the 31 embedding figures S7--S37.
    embeddings = src/'section_4_6_local_embeddings/other_indices/figures/bar_summary'
    for case, names in datasets.items():
        for name in names:
            relative = Path(f'case{case}')/f'{name}.pdf'
            files.append((embeddings/relative, Path('figures/local_embedding_bar_plots')/relative))
    return files

def main():
    """Rebuild tables and, when requested, redraw and collect the paper figures."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['tables','cached'],help='tables: generate LaTeX tables; cached: also redraw PDF figures')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    parser.add_argument('--results-root',type=Path,default=ROOT,help='root of an isolated full run; defaults to supplied caches')
    args=parser.parse_args();out=args.output.resolve();src=args.results_root.resolve()
    # Tables aggregate saved correlations from src. For new experiments, point
    # --results-root to their run directory before regenerating these summaries.
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
        # Collect the generated PDFs into the output tree after all requested
        # generators finish; copying changes locations, not figure contents.
        for source, relative_output in figures:
            target=out/relative_output;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
        print(f'Collected {len(figures)} figures.')
    print(f'Done: {out}')

if __name__=='__main__':main()

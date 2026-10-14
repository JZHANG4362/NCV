r"""
Generates a SINGLE standalone LaTeX document summarizing all three
manifold_tsne_umap_case{2,3,4}.py experiments (tuned-embedding version):
one results table (Spearman rho vs ACC, bold = row max) per (case,
clustering method), followed by the already-generated per-dataset combined
figures (bar chart of the 4 methods + t-SNE perplexity landscape + UMAP
n_neighbors landscape -- see plot_combined() in each case script) embedded
via \includegraphics.

Mirrors codes_new_manifold_new_grid/generate_manifold_tsne_umap_tex.py,
updated for the 4-method (no affinity-substitution) design: ALL_METHODS,
METHOD_LABELS, and the "Experimental setup" text all reflect Silhouette /
rNN_Hard / TSNE_Ordinary / UMAP_Ordinary instead of the earlier 6-method
battery.

Run this LOCALLY after retrieving data/ and figures/ from the cluster and
(for Case 4) running merge_manifold_case4_results.py:

    python generate_manifold_tsne_umap_tex.py

Writes tables/manifold_tsne_umap_summary.tex. Compile with (run twice for
the table of contents to populate):
    cd tables && pdflatex manifold_tsne_umap_summary.tex
    cd tables && pdflatex manifold_tsne_umap_summary.tex
"""

import os
import pickle

import numpy as np

DATA_DIR = 'data'
FIG_DIR = 'figures'
OUT_DIR = 'tables'
os.makedirs(OUT_DIR, exist_ok=True)

EXPERIMENT_TAG = 'tsne_umap_tuned_embedding_origkgrid_v1'
SCORE_DIRECTION_CONVENTION = 'larger_is_better_v1'

ALL_METHODS = ['Silhouette', 'rNN_Hard', 'TSNE_Ordinary', 'UMAP_Ordinary']
METHOD_LABELS = {
    'Silhouette':     'Silhouette',
    'rNN_Hard':       r'$r$NN-Hard',
    'TSNE_Ordinary':  't-SNE (tuned)',
    'UMAP_Ordinary':  'UMAP (tuned)',
}
CLUSTERING_METHODS = ['KMeans', 'Spectral', 'Agglomerative']
CM_LABELS = {
    'KMeans': 'K-means clustering',
    'Spectral': 'Spectral Clustering',
    'Agglomerative': 'Agglomerative Clustering',
}

# ── per-case dataset metadata (K*, N, D) and display names ─────────────
_CASE2_K_STAR_VALUES = [6, 30]
_CASE2_D_VALUES = [2, 10, 200]
_CASE2_COMPACTNESS = ['compact', 'sparse']


def _case2_sample_size_for(k_star):
    return min(1000, k_star * 50)


CASE2_ORDER = [
    f'k{k_star}_D{D}_{comp}'
    for k_star in _CASE2_K_STAR_VALUES
    for D in _CASE2_D_VALUES
    for comp in _CASE2_COMPACTNESS
]
CASE2_META = {
    dname: dict(K=int(dname.split('_')[0][1:]), N=_case2_sample_size_for(int(dname.split('_')[0][1:])),
                D=int(dname.split('_')[1][1:]))
    for dname in CASE2_ORDER
}
CASE2_DISPLAY = {dname: dname.replace('_', ', ') for dname in CASE2_ORDER}

CASE3_META = {
    'pathbased':   dict(K=3, N=300,  D=2),
    'spiral':      dict(K=3, N=312,  D=2),
    'aggregation': dict(K=7, N=788,  D=2),
    'atom':        dict(K=2, N=800,  D=3),
    'lsun':        dict(K=3, N=400,  D=2),
    'zelnik1':     dict(K=3, N=299,  D=2),
    'zelnik5':     dict(K=4, N=512,  D=2),
    'zelnik6':     dict(K=3, N=238,  D=2),
    'triangle1':   dict(K=4, N=1000, D=2),
    'longsquare':  dict(K=6, N=900,  D=2),
}
CASE3_DISPLAY = {
    'pathbased': 'Pathbased', 'spiral': 'Spiral', 'aggregation': 'Aggregation',
    'atom': 'Atom', 'lsun': 'Lsun', 'zelnik1': 'Zelnik1', 'zelnik5': 'Zelnik5',
    'zelnik6': 'Zelnik6', 'triangle1': 'Triangle1', 'longsquare': 'LongSquare',
}
CASE3_ORDER = list(CASE3_META.keys())

CASE4_META = {
    'Digits':             dict(K=10, N=1797,  D=64),
    'MNIST-test':         dict(K=10, N=10000, D=784),
    'Ecoli':              dict(K=8,  N=336,   D=7),
    'Letter':             dict(K=26, N=20000, D=16),
    'Iris':               dict(K=3,  N=150,   D=4),
    'Fashion-MNIST-test': dict(K=10, N=10000, D=784),
    'Wine':               dict(K=3,  N=178,   D=13),
    'Libras':             dict(K=15, N=360,   D=90),
    'Vowel':              dict(K=11, N=990,   D=10),
}
CASE4_DISPLAY = {k: k for k in CASE4_META}
CASE4_ORDER = list(CASE4_META.keys())

CASES = [
    dict(tag='case2', title='Case 2 -- Simulated Data', meta=CASE2_META,
         display=CASE2_DISPLAY, order=CASE2_ORDER,
         results_path=os.path.join(DATA_DIR, 'case2', f'manifold_fixed_results_{EXPERIMENT_TAG}.pkl'),
         fig_dir=os.path.join(FIG_DIR, 'case2')),
    dict(tag='case3', title='Case 3 -- Nonconvex Data', meta=CASE3_META,
         display=CASE3_DISPLAY, order=CASE3_ORDER,
         results_path=os.path.join(DATA_DIR, 'case3', f'manifold_fixed_results_{EXPERIMENT_TAG}.pkl'),
         fig_dir=os.path.join(FIG_DIR, 'case3')),
    dict(tag='case4', title='Case 4 -- Real Labeled Data', meta=CASE4_META,
         display=CASE4_DISPLAY, order=CASE4_ORDER,
         results_path=os.path.join(DATA_DIR, 'case4', f'manifold_fixed_results_{EXPERIMENT_TAG}.pkl'),
         fig_dir=os.path.join(FIG_DIR, 'case4')),
]


def load_results(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    if payload.get('score_direction_convention') != SCORE_DIRECTION_CONVENTION:
        raise ValueError(f'{path} does not use the expected score-direction convention.')
    return payload['results']


def to_latex_table(case_tag, cm, meta, display, order, results):
    '''One results table for (case, clustering_method): rows = datasets (in
    `order`), columns = K*, N, D, then the 4 methods. Missing dataset/method
    combinations render as "--"; bold marks the row-wise max among the
    FINITE entries only.'''
    cm_label = CM_LABELS.get(cm, cm)
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\small')
    lines.append(
        rf"\caption{{Spearman's $\rho$ with ACC, {cm_label} ({case_tag}). "
        rf"Bold indicates the highest value in each row; `--' marks a "
        rf"dataset/method combination not yet retrieved.}}"
    )
    lines.append(r'\centering')
    lines.append(r'\resizebox{\textwidth}{!}{')
    lines.append(r'\begin{tabular}{l' + 'c' * (3 + len(ALL_METHODS)) + '}')
    lines.append(r'\hline')
    header = ['Dataset', '$K^*$', '$N$', '$D$'] + [METHOD_LABELS[m] for m in ALL_METHODS]
    lines.append(' & '.join(header) + r' \\')
    lines.append(r'\hline')
    for dname in order:
        m = meta[dname]
        row = results.get(cm, {}).get(dname, {})
        vals = [row.get(method) for method in ALL_METHODS]
        finite = [v for v in vals if v is not None and np.isfinite(v)]
        maxv = max(finite) if finite else None
        cells = []
        for v in vals:
            if v is None or not np.isfinite(v):
                cells.append('--')
            elif maxv is not None and abs(v - maxv) < 1e-9:
                cells.append(rf'\textbf{{{v:.3f}}}')
            else:
                cells.append(f'{v:.3f}')
        lines.append(
            f"{display[dname]} & {m['K']} & {m['N']} & {m['D']} & "
            + ' & '.join(cells) + r' \\'
        )
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append('}')
    lines.append(rf'\label{{tab:manifold_{case_tag}_{cm}}}')
    lines.append(r'\end{table}')
    return '\n'.join(lines) + '\n'


def figure_block(case_tag, cm, dname, display, fig_dir):
    '''One \figure block embedding the already-generated 3-panel combined
    PDF (bar chart + t-SNE perplexity landscape + UMAP n_neighbors
    landscape) for one (case, clustering_method, dataset). Path is relative
    to OUT_DIR (tables/), matching where this .tex file is written.'''
    rel_path = os.path.join('..', fig_dir, cm, EXPERIMENT_TAG, f'Combined_{dname}.pdf')
    lines = []
    lines.append(r'\begin{figure}[h]')
    lines.append(r'\centering')
    lines.append(rf'\includegraphics[width=\textwidth]{{{rel_path}}}')
    lines.append(
        rf'\caption{{{display[dname]} -- {CM_LABELS.get(cm, cm)} ({case_tag}). '
        r'Left: Spearman $\rho$ with ACC for all 4 methods. Middle: '
        r't-SNE (tuned) $\rho$ vs.\ perplexity, with the selected '
        r'perplexity$^*$ (dotted) and the diagnostic oracle (dashed). '
        r'Right: UMAP (tuned) $\rho$ vs.\ $n_{neighbors}$, same '
        r'convention.}'
    )
    lines.append(r'\end{figure}')
    return '\n'.join(lines) + '\n'


DOC_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}

\title{Manifold t-SNE/UMAP (tuned-embedding version) vs.\ $r$NN Silhouette: Summary of Cases 2--4\\
\large (auto-generated by generate\_manifold\_tsne\_umap\_tex.py -- do not edit by hand)}
\author{}
\date{}

\begin{document}
\maketitle
\tableofcontents
\clearpage

\section*{Experimental setup}
\addcontentsline{toc}{section}{Experimental setup}

Four silhouette-family validity indices are compared against clustering
accuracy (ACC) across three case studies: the full 12-setting simulated
Gaussian grid (Case 2, $k^*\in\{6,30\}$, $D\in\{2,10,200\}$,
compactness$\in\{$compact, sparse$\}$, matching Table~1's own grid, one
replicate per setting), the ten fixed non-globular SIPU benchmark datasets
(Case 3), and nine real labeled datasets (Case 4). For each dataset,
candidate $K$ is swept and every reported $\rho$ follows the same
evaluation/selection $K$-range convention as the $r$NN-Hard baseline
(Algorithm~1, $K=2,\ldots,\max(25,1.75K^*)$ for evaluation, $+10$ buffer
for selection) for ALL FOUR cases, including Case 4. (This is the
\texttt{\_orig\_kgrid} variant: Case 4 here uses the same full sweep as
Case 2/3, not the \texttt{make\_bounded\_grid} bounded window centered on
$K^*$ used in \texttt{codes\_new\_manifold\_tuned\_embedding} -- see that
folder's remote-instructions document for the rationale behind the
bounded-window grid.)

The four methods: (1) \textbf{Silhouette} -- ordinary Euclidean
silhouette; (2) \textbf{$r$NN-Hard} -- the existing $r$NN affinity
silhouette baseline ($r^*$ selected via Algorithm~1); (3) \textbf{t-SNE
(tuned)} -- ordinary Euclidean silhouette computed on the genuine 2D
t-SNE embedding of the data, with perplexity$^*$ tuned via the same
Algorithm~1 criterion (generalized in name only, from $r$ to any tunable
neighborhood-scale parameter); (4) \textbf{UMAP (tuned)} -- ordinary
Euclidean silhouette on the genuine 2D UMAP embedding, with
$n_{neighbors}^*$ tuned via the same criterion.

\textbf{This is deliberately NOT the affinity-substitution design} used
in the sibling experiments (codes\_new\_manifold / codes\_new\_manifold\_
new\_grid), which replaced the pairwise distance with a t-SNE joint-
probability or UMAP fuzzy-union matrix (methods TSNE\_JointProb /
UMAP\_FuzzyUnion there) -- that recipe was found to saturate badly (dense,
heavy-tailed affinities collapsed under the shared min-max-normalize-
then-invert conversion). Methods (3) and (4) here instead evaluate the
GENUINE t-SNE/UMAP embeddings with ordinary Euclidean silhouette,
sidestepping that issue entirely -- only their own native hyperparameter
is tuned, nothing about the distance computation itself is altered.

Only ACC is used as the external metric (not the full 7-metric battery of
the main reproducibility package).

"""

DOC_CLOSING = r"""\end{document}
"""


def main():
    parts = [DOC_PREAMBLE]

    for case in CASES:
        results = load_results(case['results_path'])
        parts.append(f"\\section{{{case['title']}}}\n")
        if results is None:
            parts.append(f"No results file found at \\texttt{{{case['results_path']}}} -- skipping.\n\n")
            print(f"{case['tag']}: NO RESULTS FILE -- skipped")
            continue

        for cm in CLUSTERING_METHODS:
            if cm not in results:
                print(f"{case['tag']} / {cm}: no results -- skipping table+figures")
                continue

            parts.append(f"\\subsection{{{CM_LABELS.get(cm, cm)}}}\n")
            parts.append(to_latex_table(case['tag'], cm, case['meta'], case['display'], case['order'], results))
            parts.append('')

            n_present = 0
            for dname in case['order']:
                if dname not in results.get(cm, {}):
                    continue
                fig_path = os.path.join(case['fig_dir'], cm, EXPERIMENT_TAG, f'Combined_{dname}.pdf')
                if not os.path.exists(fig_path):
                    print(f"  {case['tag']} / {cm} / {dname}: results present but figure "
                          f"missing at {fig_path} -- skipping figure")
                    continue
                parts.append(figure_block(case['tag'], cm, dname, case['display'], case['fig_dir']))
                parts.append('')
                n_present += 1
            print(f"{case['tag']} / {cm}: table with {len(case['order'])} dataset rows, "
                  f"{n_present} figures embedded")
            parts.append(r'\clearpage')
            parts.append('')

    parts.append(DOC_CLOSING)
    combined = '\n'.join(parts)
    combined_path = os.path.join(OUT_DIR, 'manifold_tsne_umap_summary.tex')
    with open(combined_path, 'w') as f:
        f.write(combined)
    print(f'\nWrote combined standalone document: {combined_path}')
    print('Compile with (run twice for the TOC): cd tables && pdflatex manifold_tsne_umap_summary.tex')


if __name__ == '__main__':
    main()

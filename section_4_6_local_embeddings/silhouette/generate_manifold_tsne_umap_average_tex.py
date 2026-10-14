r"""
Generates a SINGLE standalone LaTeX document summarizing Cases 2/3/4 as
PER-CASE AVERAGES over datasets, instead of one row per dataset (see
generate_manifold_tsne_umap_tex.py for the full per-dataset version this
is a companion to). For each case and each clustering method, one row
reports the mean Spearman rho (vs ACC) for each of the 4 methods, averaged
across all datasets in that case (skipping any dataset/method combination
that isn't available -- e.g. Letter is still pending resubmission after
the rNN-sweep memory fix, so it's excluded from the Case 4 average until
retrieved, and the table's "n" column reports how many datasets actually
contributed).

Run this LOCALLY after retrieving data/ from the cluster and (for Case 4)
running merge_manifold_case4_results.py -- same prerequisites as
generate_manifold_tsne_umap_tex.py:

    python generate_manifold_tsne_umap_average_tex.py

Writes tables/manifold_tsne_umap_averages.tex. Compile with:
    cd tables && pdflatex manifold_tsne_umap_averages.tex
"""

import os
import pickle

import numpy as np

DATA_DIR = 'data'
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

CASE2_ORDER_LEN = 12   # full 12-setting simulated Gaussian grid
CASE3_ORDER_LEN = 10   # ten fixed nonconvex SIPU datasets
CASE4_ORDER_LEN = 9    # nine real labeled datasets

CASES = [
    dict(tag='case2', title='Case 2 -- Simulated Data', n_total=CASE2_ORDER_LEN,
         results_path=os.path.join(DATA_DIR, 'case2', f'manifold_fixed_results_{EXPERIMENT_TAG}.pkl')),
    dict(tag='case3', title='Case 3 -- Nonconvex Data', n_total=CASE3_ORDER_LEN,
         results_path=os.path.join(DATA_DIR, 'case3', f'manifold_fixed_results_{EXPERIMENT_TAG}.pkl')),
    dict(tag='case4', title='Case 4 -- Real Labeled Data', n_total=CASE4_ORDER_LEN,
         results_path=os.path.join(DATA_DIR, 'case4', f'manifold_fixed_results_{EXPERIMENT_TAG}.pkl')),
]


def load_results(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    if payload.get('score_direction_convention') != SCORE_DIRECTION_CONVENTION:
        raise ValueError(f'{path} does not use the expected score-direction convention.')
    return payload['results']


def average_row(results, cm, n_total):
    '''Mean rho per method across all datasets present for (results, cm),
    skipping missing/non-finite entries. Returns (means_dict, n_used) where
    n_used is the max number of datasets that contributed to ANY method's
    average (methods with fewer finite values than n_used are averaged over
    fewer datasets -- see the table caption).'''
    by_dataset = results.get(cm, {})
    means = {}
    n_used_per_method = {}
    for method in ALL_METHODS:
        vals = [row.get(method) for row in by_dataset.values()]
        finite = [v for v in vals if v is not None and np.isfinite(v)]
        means[method] = float(np.mean(finite)) if finite else None
        n_used_per_method[method] = len(finite)
    n_used = max(n_used_per_method.values()) if n_used_per_method else 0
    return means, n_used_per_method, n_used


def to_latex_table(case_tag, meta_rows, n_total):
    '''meta_rows: list of (cm, means_dict, n_used_per_method) tuples, one
    per clustering method.'''
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\small')
    lines.append(
        rf"\caption{{Mean Spearman's $\rho$ with ACC, averaged over datasets "
        rf"({case_tag}, {n_total} datasets total). Bold indicates the "
        rf"highest value in each row; parenthesized $n$ is the number of "
        rf"datasets that contributed a finite value for that method (equal "
        rf"to {n_total} unless a dataset/method combination is still "
        rf"missing).}}"
    )
    lines.append(r'\centering')
    lines.append(r'\begin{tabular}{l' + 'c' * len(ALL_METHODS) + '}')
    lines.append(r'\hline')
    header = ['Clustering method'] + [METHOD_LABELS[m] for m in ALL_METHODS]
    lines.append(' & '.join(header) + r' \\')
    lines.append(r'\hline')
    for cm, means, n_used_per_method in meta_rows:
        vals = [means[m] for m in ALL_METHODS]
        finite = [v for v in vals if v is not None and np.isfinite(v)]
        maxv = max(finite) if finite else None
        cells = []
        for m in ALL_METHODS:
            v = means[m]
            n_used = n_used_per_method[m]
            if v is None:
                cells.append('--')
            else:
                n_suffix = '' if n_used == n_total else rf'\,(n={n_used})'
                if maxv is not None and abs(v - maxv) < 1e-9:
                    cells.append(rf'\textbf{{{v:.3f}}}{n_suffix}')
                else:
                    cells.append(f'{v:.3f}{n_suffix}')
        lines.append(f"{CM_LABELS.get(cm, cm)} & " + ' & '.join(cells) + r' \\')
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(rf'\label{{tab:manifold_avg_{case_tag}}}')
    lines.append(r'\end{table}')
    return '\n'.join(lines) + '\n'


DOC_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}

\title{Manifold t-SNE/UMAP (tuned-embedding, original K-grid) vs.\ $r$NN Silhouette:\\
Per-Case Averages Over Datasets\\
\large (auto-generated by generate\_manifold\_tsne\_umap\_average\_tex.py -- do not edit by hand)}
\author{}
\date{}

\begin{document}
\maketitle

\section*{What this summarizes}

This is a companion to \texttt{manifold\_tsne\_umap\_summary.tex} (the full
per-dataset version). Instead of one row per dataset, each table below
collapses ALL datasets within a case into ONE row per clustering method --
the value shown is the mean Spearman $\rho$ (vs.\ ACC) across every
dataset in that case, for each of the 4 methods. Cases 2/3/4 and the 4
methods (Silhouette, $r$NN-Hard, t-SNE (tuned), UMAP (tuned)) are exactly
as in the per-dataset document; see that document's ``Experimental setup''
section for the full methodology (original, non-bounded-window $K$-grid
for all three cases, Algorithm~1 parameter selection, etc.).

Missing dataset/method combinations (e.g.\ Case 4's \texttt{Letter}, still
pending resubmission after the rNN-sweep memory fix) are excluded from the
mean rather than treated as zero -- the parenthesized $n$ next to a cell
reports how many datasets actually contributed whenever it's less than the
case's total dataset count.

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

        meta_rows = []
        for cm in CLUSTERING_METHODS:
            if cm not in results:
                print(f"{case['tag']} / {cm}: no results -- skipping row")
                continue
            means, n_used_per_method, n_used = average_row(results, cm, case['n_total'])
            meta_rows.append((cm, means, n_used_per_method))
            print(f"{case['tag']} / {cm}: averaged over up to {n_used}/{case['n_total']} datasets "
                  f"-- {[f'{m}={means[m]:.3f}' if means[m] is not None else f'{m}=--' for m in ALL_METHODS]}")

        if not meta_rows:
            parts.append("No clustering-method results available -- skipping.\n\n")
            continue

        parts.append(to_latex_table(case['tag'], meta_rows, case['n_total']))
        parts.append('')

    parts.append(DOC_CLOSING)
    combined = '\n'.join(parts)
    combined_path = os.path.join(OUT_DIR, 'manifold_tsne_umap_averages.tex')
    with open(combined_path, 'w') as f:
        f.write(combined)
    print(f'\nWrote combined standalone document: {combined_path}')
    print('Compile with:')
    print('  cd tables && pdflatex manifold_tsne_umap_averages.tex')


if __name__ == '__main__':
    main()

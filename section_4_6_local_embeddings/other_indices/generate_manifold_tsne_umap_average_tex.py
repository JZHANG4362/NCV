r"""
Generates a SINGLE standalone LaTeX document summarizing Cases 2/3/4 as
PER-CASE AVERAGES over datasets, instead of one row per dataset (see
generate_manifold_tsne_umap_tex.py for the full per-dataset version this
is a companion to). For each case and each clustering method, one row
reports the mean Spearman rho (vs ACC) for each of the 25 methods
(rNN_Hard + 8 indices x 3 representations), averaged across all datasets
in that case -- makes the 25-column table more digestible than the full
per-dataset landscape version.

Run this LOCALLY after retrieving data/ from the cluster and (for Case 4)
running merge_manifold_case4_results.py:

    python generate_manifold_tsne_umap_average_tex.py

Writes tables/manifold_tsne_umap_averages.tex. Compile with:
    cd tables && pdflatex manifold_tsne_umap_averages.tex
"""

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe (no display)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

DATA_DIR = 'data'
FIG_DIR = 'figures'
OUT_DIR = 'tables'
os.makedirs(OUT_DIR, exist_ok=True)

EXPERIMENT_TAG = 'tsne_umap_alt_indices_tuned_origkgrid_v1'
SCORE_DIRECTION_CONVENTION = 'larger_is_better_v1'

INDEX_NAMES = ['VRC', 'WG', 'WB', 'DBCV', 'CVDD', 'VIASCKDE', 'DCSI', 'DB']
REPRESENTATIONS = ['Original', 'TSNE', 'UMAP']
REP_LABELS = {'Original': 'Orig', 'TSNE': 't-SNE', 'UMAP': 'UMAP'}

ALL_METHODS = ['rNN_Hard'] + [f'{idx}_{rep}' for idx in INDEX_NAMES for rep in REPRESENTATIONS]
METHOD_LABELS = {'rNN_Hard': r'$r$NN'}
for _idx in INDEX_NAMES:
    for _rep in REPRESENTATIONS:
        METHOD_LABELS[f'{_idx}_{_rep}'] = f'{_idx}-{REP_LABELS[_rep]}'

CLUSTERING_METHODS = ['KMeans', 'Spectral', 'Agglomerative']
CM_LABELS = {
    'KMeans': 'K-means clustering',
    'Spectral': 'Spectral Clustering',
    'Agglomerative': 'Agglomerative Clustering',
}

CASE2_ORDER_LEN = 12
CASE3_ORDER_LEN = 10
CASE4_ORDER_LEN = 9

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
    by_dataset = results.get(cm, {})
    means = {}
    n_used_per_method = {}
    for method in ALL_METHODS:
        vals = [row.get(method) for row in by_dataset.values()]
        finite = [v for v in vals if v is not None and np.isfinite(v)]
        means[method] = float(np.mean(finite)) if finite else None
        n_used_per_method[method] = len(finite)
    return means, n_used_per_method


def to_latex_table(case_tag, meta_rows, n_total):
    lines = []
    lines.append(r'\begin{landscape}')
    lines.append(r'\begin{table}[h]')
    lines.append(r'\scriptsize')
    lines.append(r'\setlength{\tabcolsep}{2.2pt}')
    lines.append(
        rf"\caption{{Mean Spearman's $\rho$ with ACC, averaged over datasets "
        rf"({case_tag}, {n_total} datasets total). Bold indicates the "
        rf"highest value in each row; a subscript $n$ marks a value "
        rf"averaged over fewer than {n_total} datasets.}}"
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
                sub = '' if n_used == n_total else rf'$_{{n={n_used}}}$'
                if maxv is not None and abs(v - maxv) < 1e-9:
                    cells.append(rf'\textbf{{{v:.2f}}}{sub}')
                else:
                    cells.append(f'{v:.2f}{sub}')
        lines.append(f"{CM_LABELS.get(cm, cm)} & " + ' & '.join(cells) + r' \\')
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(rf'\label{{tab:manifold_avg_{case_tag}}}')
    lines.append(r'\end{table}')
    lines.append(r'\end{landscape}')
    return '\n'.join(lines) + '\n'


def to_delta_table(case_tag, meta_rows, n_total):
    '''Companion table to to_latex_table(): for each index, Delta_TSNE =
    mean(TSNE) - mean(Original) and Delta_UMAP = mean(UMAP) - mean(Original)
    (does the embedding help or hurt, relative to computing the same index
    on raw X, on average across this case's datasets?). Rows = clustering
    method. Sign shown explicitly; no bolding -- sign/magnitude is the
    point, not a row max.'''
    lines = []
    lines.append(r'\begin{landscape}')
    lines.append(r'\begin{table}[h]')
    lines.append(r'\scriptsize')
    lines.append(r'\setlength{\tabcolsep}{2.2pt}')
    lines.append(
        rf"\caption{{Mean $\Delta$ vs.\ Original ($=$ mean index$_{{\text{{t-SNE/UMAP}}}} - $ "
        rf"mean index$_{{\text{{Original}}}}$, averaged over the same "
        rf"datasets as Table~\ref{{tab:manifold_avg_{case_tag}}}), {case_tag}. "
        rf"Positive $=$ the embedding-based version correlates with ACC "
        rf"MORE than the same index on raw $X$, on average; negative $=$ less.}}"
    )
    lines.append(r'\centering')
    lines.append(r'\begin{tabular}{l' + 'c' * (2 * len(INDEX_NAMES)) + '}')
    lines.append(r'\hline')
    header = ['Clustering method']
    for idx in INDEX_NAMES:
        header += [rf'{idx} $\Delta$t-SNE', rf'{idx} $\Delta$UMAP']
    lines.append(' & '.join(header) + r' \\')
    lines.append(r'\hline')
    for cm, means, n_used_per_method in meta_rows:
        cells = []
        for idx in INDEX_NAMES:
            orig = means.get(f'{idx}_Original')
            for rep in ['TSNE', 'UMAP']:
                v = means.get(f'{idx}_{rep}')
                if v is None or orig is None:
                    cells.append('--')
                else:
                    cells.append(f'{v - orig:+.2f}')
        lines.append(f"{CM_LABELS.get(cm, cm)} & " + ' & '.join(cells) + r' \\')
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(rf'\label{{tab:manifold_avg_delta_{case_tag}}}')
    lines.append(r'\end{table}')
    lines.append(r'\end{landscape}')
    return '\n'.join(lines) + '\n'


def plot_delta_bar(case_tag, cm, means):
    '''Grouped bar chart: x = 8 indices, bars = mean Delta_TSNE / mean
    Delta_UMAP (vs. Original), for one (case, clustering_method) --
    visualizes one row of to_delta_table(). Saved to
    figures/{case_tag}/{cm}/{EXPERIMENT_TAG}/DeltaAverage.{pdf,png}.'''
    delta_tsne, delta_umap = [], []
    for idx in INDEX_NAMES:
        orig = means.get(f'{idx}_Original')
        t = means.get(f'{idx}_TSNE')
        u = means.get(f'{idx}_UMAP')
        delta_tsne.append(t - orig if (t is not None and orig is not None) else np.nan)
        delta_umap.append(u - orig if (u is not None and orig is not None) else np.nan)

    x = np.arange(len(INDEX_NAMES))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, delta_tsne, width, label=r'$\Delta$t-SNE', color='seagreen',
           alpha=0.9, edgecolor='black', linewidth=0.4)
    ax.bar(x + width / 2, delta_umap, width, label=r'$\Delta$UMAP', color='darkorange',
           alpha=0.9, edgecolor='black', linewidth=0.4)
    ax.axhline(0, color='grey', linewidth=1, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(INDEX_NAMES, fontsize=10)
    ax.set_ylabel(r'mean $\Delta$ vs. Original (Spearman $\rho$ with ACC)', fontsize=11)
    ax.set_title(f'{CM_LABELS.get(cm, cm)} ({case_tag}) -- embedding vs. raw-$X$ delta, '
                 f'averaged over datasets', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    out_dir = os.path.join(FIG_DIR, case_tag, cm, EXPERIMENT_TAG)
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, 'DeltaAverage.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'DeltaAverage.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return pdf_path


def delta_figure_block(case_tag, cm, pdf_path):
    lines = []
    lines.append(r'\begin{figure}[h]')
    lines.append(r'\centering')
    lines.append(rf'\includegraphics[width=0.85\textwidth]{{{os.path.join("..", pdf_path)}}}')
    lines.append(
        rf'\caption{{Mean $\Delta$ vs.\ Original, {CM_LABELS.get(cm, cm)} ({case_tag}) -- '
        r'bar-chart view of the delta table above.}'
    )
    lines.append(r'\end{figure}')
    return '\n'.join(lines) + '\n'


DOC_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{pdflscape}

\title{Manifold Alternative Validity Indices (TUNED) vs.\ $r$NN Silhouette:\\
Per-Case Averages Over Datasets\\
\large (auto-generated by generate\_manifold\_tsne\_umap\_average\_tex.py -- do not edit by hand)}
\author{}
\date{}

\begin{document}
\maketitle

\section*{What this summarizes}

This is a companion to \texttt{manifold\_tsne\_umap\_summary.tex} (the full
per-dataset version). Instead of one row per dataset, each landscape table
below collapses ALL datasets within a case into ONE row per clustering
method -- the value shown is the mean Spearman $\rho$ (vs.\ ACC) across
every dataset in that case, for each of the 25 methods (rNN\_Hard + 8
alternative internal validity indices x 3 representations: Orig/t-SNE/
UMAP). See the per-dataset document's ``Experimental setup'' section for
the full methodology.

Missing dataset/method combinations are excluded from the mean rather
than treated as zero -- a subscript $n$ next to a cell reports how many
datasets actually contributed whenever it's less than the case's total
dataset count.

"""

DOC_CLOSING = r"""\end{document}
"""


def main():
    parts = [DOC_PREAMBLE]

    for case in CASES:
        results = load_results(case['results_path'])
        parts.append(f"\\section{{{case['title']}}}\n")
        if results is None:
            parts.append(f"No results file found at \\texttt{{{case['results_path'].replace('_', r'\_')}}} -- skipping.\n\n")
            print(f"{case['tag']}: NO RESULTS FILE -- skipped")
            continue

        meta_rows = []
        for cm in CLUSTERING_METHODS:
            if cm not in results:
                print(f"{case['tag']} / {cm}: no results -- skipping row")
                continue
            means, n_used_per_method = average_row(results, cm, case['n_total'])
            meta_rows.append((cm, means, n_used_per_method))
            n_rnn = n_used_per_method.get('rNN_Hard', 0)
            print(f"{case['tag']} / {cm}: averaged over up to {n_rnn}/{case['n_total']} datasets")

        if not meta_rows:
            parts.append("No clustering-method results available -- skipping.\n\n")
            continue

        parts.append(to_latex_table(case['tag'], meta_rows, case['n_total']))
        parts.append('')
        parts.append(to_delta_table(case['tag'], meta_rows, case['n_total']))
        parts.append('')
        for cm, means, _n_used in meta_rows:
            pdf_path = plot_delta_bar(case['tag'], cm, means)
            parts.append(delta_figure_block(case['tag'], cm, pdf_path))
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

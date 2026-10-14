r"""
Generates LaTeX tables of Spearman's rho (per dataset, per index) from
partB_fixed_datasets_comparison.py's results pickles -- one table per
(external_metric, clustering_method) combination that is FULLY complete
(all 10 datasets present). Matches the exact structure of
Table~\ref{tab:nonconvex_results} in main.tex (same columns, same
bold-row-max styling) so a generated .tex file can directly
replace/refresh it via \input{}.

Also fixes a latent bug in the hand-written tab:nonconvex_results in
main.tex: its \begin{tabular}{ccccccccccc} declares 11 columns, but the
header row actually has 12 (Dataset, K* + 10 methods) -- this script's
output declares the correct 12 c's.

Usage: run from this folder after retrieving the results/*.pkl files
from the cluster (see cluster_setup_commands_partB.docx STEP 8):

    python generate_nonconvex_tables.py

Writes one .tex file per complete (metric, clustering_method) pair to
tables/nonconvex_{METRIC}_{ClusteringMethod}.tex, and prints a
completeness report for everything else (missing/partial combinations
are skipped, not silently included with gaps). Unlike Part A, there is
no N_REP here (each dataset is fixed, single seed) -- "complete" just
means the (dataset, method) scalar is present and finite.
"""

import os
import pickle

import numpy as np

DATA_DIR = 'data'
OUT_DIR = 'tables'
os.makedirs(OUT_DIR, exist_ok=True)

EXPERIMENT_TAG = 'paperkmax_buffer10_r130_nonself_oriented'
SCORE_DIRECTION_CONVENTION = 'larger_is_better_v1'

# Must match partB_fixed_datasets_comparison.py's own EXPECTED_TABLE3 exactly.
DATASET_K_STAR = {
    'pathbased':   3,
    'spiral':      3,
    'aggregation': 7,
    'atom':        2,
    'lsun':        3,
    'zelnik1':     3,
    'zelnik5':     4,
    'zelnik6':     3,
    'triangle1':   4,
    'longsquare':  6,
}
DATASET_DISPLAY_NAMES = {
    'pathbased':   'Pathbased',
    'spiral':      'Spiral',
    'aggregation': 'Aggregation',
    'atom':        'Atom',
    'lsun':        'Lsun',
    'zelnik1':     'Zelnik1',
    'zelnik5':     'Zelnik5',
    'zelnik6':     'Zelnik6',
    'triangle1':   'Triangle1',
    'longsquare':  'LongSquare',
}
DATASET_NAMES = list(DATASET_K_STAR.keys())

ALL_METHODS = [
    'VRC', 'Silhouette', 'WG', 'WB', 'DBCV', 'rNN_Hard', 'rNN_Soft',
    'CVDD', 'VIASCKDE', 'DCSI',
]
METHOD_LABELS = {
    'VRC': 'VRC', 'Silhouette': 'Silhouette', 'WG': 'WG', 'WB': 'WB',
    'DBCV': 'DBCV', 'rNN_Hard': r'$r$NN-Hard', 'rNN_Soft': r'$r$NN-Soft',
    'CVDD': 'CVDD', 'VIASCKDE': 'VIASCKDE', 'DCSI': 'DCSI',
}

METRICS = ['ACC', 'Jaccard', 'SS3', 'ARI', 'NMI', 'Powers', 'NID']

CM_LABELS = {
    'KMeans': 'K-means clustering',
    'Spectral': 'Spectral Clustering',
    'Agglomerative': 'Agglomerative Clustering',
}


def results_path_for(metric):
    '''Mirror of partB_fixed_datasets_comparison.py's own results_path_for:
    ACC gets the plain, no-suffix filename; every other external metric
    gets a `_{METRIC}` suffix. Must stay in sync with the writer side.'''
    suffix = '' if metric == 'ACC' else f'_{metric}'
    return os.path.join(DATA_DIR, f'nonconvex_fixed_results_{EXPERIMENT_TAG}{suffix}.pkl')


def load_results(metric):
    '''Load one external metric's main-results pickle (written by
    partB_fixed_datasets_comparison.py's save_results/save_all), or return
    None if that file doesn't exist yet (e.g. the SLURM job for that
    metric/dataset combination hasn't completed). Guards against silently
    mixing in results computed under a different score-direction
    convention (e.g. an older run that didn't reverse NID/WB before
    correlating) by checking the 'score_direction_convention' field
    stamped into the pickle and raising if it doesn't match what this
    script expects.'''
    path = results_path_for(metric)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    if payload.get('score_direction_convention') != SCORE_DIRECTION_CONVENTION:
        raise ValueError(
            f'{path} does not use the required larger-is-better score direction; '
            'do not mix it with corrected results.'
        )
    return payload['results']


def dataset_is_complete(results, cm, dname):
    """A dataset counts as complete once it's been run (present, with all
    10 method keys recorded) -- NOT that every value is finite. Individual
    NaN cells still render as '--' in the table (see to_latex) rather than
    hiding the whole row -- see Case4/generate_realdata_tables.py's version
    of this function for why (VIASCKDE/DBCV genuinely NaN on high-D real
    data)."""
    if cm not in results or dname not in results[cm]:
        return False
    return all(method in results[cm][dname] for method in ALL_METHODS)


def rho_row(results, cm, dname):
    '''Extract one dataset's Spearman rho for all 10 validity indices, in
    ALL_METHODS order, as a plain float list -- this fixed order is what
    lets to_latex and compute_rnn_performance_summary index into the row by
    position (idx_sil/idx_hard/idx_soft) rather than by dict lookup.'''
    return [float(results[cm][dname][method]) for method in ALL_METHODS]


def to_latex(metric, cm, rows_by_dataset):
    '''Render one (external_metric, clustering_method) table as a LaTeX
    `table` environment: one row per dataset (Dataset name, K*, then one rho
    column per index in ALL_METHODS order), formatted to match
    Table~\\ref{tab:nonconvex_results} in main.tex exactly (12 columns:
    Dataset + K* + 10 methods -- see the module docstring's note about the
    hand-written table's column-count bug this script avoids). Within each
    row, the single largest finite rho is bolded (ties within 1e-9 all get
    bolded, though a true tie is essentially never observed with 3-decimal
    values). Returns the table as a single string ready to be concatenated
    into the combined document by build_combined_document.'''
    cm_label = CM_LABELS.get(cm, cm)
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\small')
    lines.append(
        rf"\caption{{Spearman's $\rho$ with {metric} for the ten nonconvex "
        rf"real datasets, {cm_label}. Bold indicates the highest value in "
        rf"each row.}}"
    )
    lines.append(r'\centering')
    lines.append(r'\resizebox{\textwidth}{!}{')
    lines.append(r'\begin{tabular}{' + 'c' * (2 + len(ALL_METHODS)) + '}')
    lines.append(r'\hline')
    header = ['Dataset', '$K^*$'] + [METHOD_LABELS[m] for m in ALL_METHODS]
    lines.append(' & '.join(header) + r' \\')
    lines.append(r'\hline')
    for dname in DATASET_NAMES:
        vals = rows_by_dataset[dname]
        # NaN handling: a NaN entry (e.g. DBCV/VIASCKDE genuinely undefined
        # for a particular dataset/method combination) renders as '--'
        # rather than being dropped from the row or excluding the whole
        # dataset from the table -- the row still appears with every OTHER
        # index's value intact, and the row-max bolding below is computed
        # only over the finite values so a NaN can never be (mis)selected
        # as the "best" entry. Same convention as Case4/generate_realdata_tables.py.
        finite = [v for v in vals if not np.isnan(v)]
        maxv = max(finite) if finite else None
        cells = []
        for v in vals:
            if np.isnan(v):
                cells.append('--')
            elif maxv is not None and abs(v - maxv) < 1e-9:
                cells.append(rf'\textbf{{{v:.3f}}}')
            else:
                cells.append(f'{v:.3f}')
        lines.append(
            f'{DATASET_DISPLAY_NAMES[dname]} & {DATASET_K_STAR[dname]} & '
            + ' & '.join(cells) + r' \\'
        )
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append('}')
    lines.append(rf'\label{{tab:nonconvex_{metric}_{cm}}}')
    lines.append(r'\end{table}')
    return '\n'.join(lines) + '\n'


DOC_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}

\title{Benchmark\_data\_simulated\_nonconvex: Full Results Tables\\
\large (auto-generated by generate\_nonconvex\_tables.py -- do not edit by hand)}
\author{}
\date{}

\begin{document}
\maketitle

\section*{Experimental setup}

Ten real, fixed 2D/3D datasets with non-convex, non-spherical cluster
shapes (Pathbased, Spiral, Aggregation, Atom, Lsun, Zelnik1, Zelnik5,
Zelnik6, Triangle1, LongSquare -- the collection compiled for the CVDD
index) are each clustered independently under three algorithms: K-means,
Spectral Clustering (nearest-neighbors affinity), and Agglomerative
Clustering (Ward linkage). Because each dataset is fixed rather than
simulated, there is no replication -- one clustering per (dataset,
method) pair, with a fixed random seed.

For a dataset with true cluster count $K^*$, candidate $K$ is swept over
$K=2,\ldots,\max(25,1.75K^*)+10$. The evaluation scope for every reported
$\rho$ (all 10 indices, not just the $r$NN ones) is the narrower range
$K=2,\ldots,\max(25,1.75K^*)$ -- the reference paper's own $K$-range cap.
The wider $+10$-buffered range is used only to select $r^*$ for the
$r$NN-affinity Hard/Soft Silhouette variants
(\texttt{select\_r\_topK\_spearman}, top-$m$ monotonic-decrease
criterion, $m=35$), following the same convention as the simulated Case 2
study and the paper's Algorithm~1.

Each table reports the Spearman rank correlation $\rho$ between each of
the 10 validity indices' Silhouette-style score (evaluated across the $K$
sweep) and the named external metric (also evaluated across the same $K$
sweep), for one clustering method. Bold indicates the highest value in
each row. Tables are grouped by external metric, with one table per
clustering method (K-means, Spectral Clustering, Agglomerative
Clustering) within each group.

"""

DOC_CLOSING = r"""\end{document}
"""


def compute_rnn_performance_summary(all_raw_data):
    """all_raw_data: dict metric -> dict cm -> dict dataset -> list of 10
    floats (order = ALL_METHODS). Returns per-variant counts of: how often
    rNN_Hard/rNN_Soft is the single best index in a row, how often it's in
    the top 3, and how often it beats the original Silhouette score --
    tallied across every (metric, clustering_method, dataset) row in the
    tables actually generated (so this always matches what's in the
    document, whatever subset of metrics/methods happens to be complete).
    """
    idx_sil  = ALL_METHODS.index('Silhouette')
    idx_hard = ALL_METHODS.index('rNN_Hard')
    idx_soft = ALL_METHODS.index('rNN_Soft')

    stats = {}
    for variant, idx in [('rNN_Hard', idx_hard), ('rNN_Soft', idx_soft)]:
        n_total = n_best = n_top3 = n_beats_sil = 0
        for metric, by_cm in all_raw_data.items():
            for cm, by_dataset in by_cm.items():
                for dname, vals in by_dataset.items():
                    v = vals[idx]
                    if np.isnan(v):
                        continue
                    finite_vals = [x for x in vals if not np.isnan(x)]
                    rank = 1 + sum(1 for x in finite_vals if x > v)
                    n_total += 1
                    if rank == 1:
                        n_best += 1
                    if rank <= 3:
                        n_top3 += 1
                    v_sil = vals[idx_sil]
                    if not np.isnan(v_sil) and v > v_sil:
                        n_beats_sil += 1
        stats[variant] = dict(n_total=n_total, n_best=n_best, n_top3=n_top3,
                               n_beats_sil=n_beats_sil)
    return stats


def performance_summary_to_latex(stats):
    def pct(n, total):
        return 100.0 * n / total if total else float('nan')

    lines = [r'\section*{$r$NN-affinity Silhouette performance summary}', '']
    lines.append(
        r"Tallied across every (external metric, clustering method, dataset) "
        r"row in the tables below -- i.e. every row of every table currently "
        r"in this document, not a fixed hand-picked subset:"
    )
    lines.append(r'\begin{itemize}')
    for variant, label in [('rNN_Hard', r'$r$NN-Hard'), ('rNN_Soft', r'$r$NN-Soft')]:
        s = stats.get(variant)
        if not s or s['n_total'] == 0:
            continue
        lines.append(
            rf"\item \textbf{{{label}}}: out of {s['n_total']} rows, it is the "
            rf"single best-performing index in {s['n_best']} "
            rf"({pct(s['n_best'], s['n_total']):.0f}\%), ranks in the top three "
            rf"(of 10 indices) in {s['n_top3']} ({pct(s['n_top3'], s['n_total']):.0f}\%), "
            rf"and outperforms the original (global) Silhouette score in "
            rf"{s['n_beats_sil']} ({pct(s['n_beats_sil'], s['n_total']):.0f}\%)."
        )
    lines.append(r'\end{itemize}')
    lines.append('')
    return '\n'.join(lines) + '\n'


def build_combined_document(all_tables, all_raw_data):
    """all_tables: dict metric -> dict cm -> tex table string (only complete ones).
    all_raw_data: dict metric -> dict cm -> dict dataset -> list of 10 floats,
    same (metric, cm) coverage as all_tables -- used for the performance summary."""
    parts = [DOC_PREAMBLE]

    stats = compute_rnn_performance_summary(all_raw_data)
    parts.append(performance_summary_to_latex(stats))

    for metric in METRICS:
        if metric not in all_tables or not all_tables[metric]:
            continue
        parts.append(f'\\section{{External metric: {metric}}}\n')
        for cm in ['KMeans', 'Spectral', 'Agglomerative']:
            if cm not in all_tables[metric]:
                continue
            parts.append(all_tables[metric][cm])
            parts.append('')
        parts.append('\\clearpage\n')
    parts.append(DOC_CLOSING)
    return '\n'.join(parts)


def main():
    '''Entry point: for every (external_metric, clustering_method) pair
    across all 7 METRICS and 3 clustering methods, load the corresponding
    results pickle, check completeness (all 10 datasets present with all
    10 method keys -- see dataset_is_complete), and if complete, render a
    LaTeX table for it and fold it into the combined standalone document.
    Pairs that are missing or incomplete are printed but skipped, never
    included with gaps -- so tables/nonconvex_all_tables.tex only ever
    contains fully-populated tables. Prints a per-pair completeness report
    to stdout as it goes, then writes the combined .tex file.'''
    all_tables = {}    # metric -> cm -> tex string
    all_raw_data = {}  # metric -> cm -> dataset -> list of 10 floats

    for metric in METRICS:
        results = load_results(metric)
        if results is None:
            print(f'{metric}: no results file found ({results_path_for(metric)}) -- skipping')
            continue

        for cm in results:
            missing = [
                dname for dname in DATASET_NAMES
                if not dataset_is_complete(results, cm, dname)
            ]
            if missing:
                print(f'{metric} / {cm}: INCOMPLETE -- skipped '
                      f'({len(missing)}/{len(DATASET_NAMES)} datasets missing/non-finite): {missing}')
                continue

            rows_by_dataset = {
                dname: rho_row(results, cm, dname) for dname in DATASET_NAMES
            }
            tex = to_latex(metric, cm, rows_by_dataset)
            print(f'{metric} / {cm}: COMPLETE')

            all_tables.setdefault(metric, {})[cm] = tex
            all_raw_data.setdefault(metric, {})[cm] = rows_by_dataset

    combined = build_combined_document(all_tables, all_raw_data)
    combined_path = os.path.join(OUT_DIR, 'nonconvex_all_tables.tex')
    with open(combined_path, 'w') as f:
        f.write(combined)
    n_tables = sum(len(cms) for cms in all_tables.values())
    print(f'\nWrote combined standalone document: {combined_path} ({n_tables} tables)')
    print('Compile with: pdflatex nonconvex_all_tables.tex  (run from the tables/ folder)')


if __name__ == '__main__':
    main()

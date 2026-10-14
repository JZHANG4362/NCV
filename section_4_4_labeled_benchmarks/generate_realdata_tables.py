r"""
Generates LaTeX tables of Spearman's rho (per dataset, per index) from
Benchmark_data_real.py's results pickles -- one table per (external_metric,
clustering_method) combination PRESENT IN THE DATA SO FAR, populated with
whatever datasets/indices have actually finished; missing datasets show
'--' for every column, and individual indices that failed to produce a
finite value (see the VIASCKDE/DBCV numerical-stability notes) show '--'
for just that cell. Unlike generate_nonconvex_tables.py (Part B), which
only emits a table once all 10 datasets are complete, this script is
meant to be regenerated repeatedly as remote jobs finish -- each table's
caption reports how many of the 9 datasets are done so far. Unlike Part
A there is no N_REP here (each dataset is fixed, single seed per
(dataset, method)), so each cell is just a single value, no mean/median
split needed.

Usage: run from this folder after retrieving the results/*.pkl files from
the cluster and running merge_realdata_results.py to fold the per-job
tagged files into the canonical ones (see
cluster_setup_commands_realdata_parallel.docx):

    python merge_realdata_results.py
    python generate_realdata_tables.py

Writes ONE combined, standalone, compilable document to
tables/realdata_all_tables.tex, and prints a per-(metric,
clustering_method) coverage report (COMPLETE or PARTIAL n/9).
"""

import os
import pickle

import numpy as np

DATA_DIR = 'data'
OUT_DIR = 'tables'
os.makedirs(OUT_DIR, exist_ok=True)

EXPERIMENT_TAG = 'paperkmax_buffer10_r130_nonself_oriented'
SCORE_DIRECTION_CONVENTION = 'larger_is_better_v1'

# Must match Benchmark_data_real.py's own EXPECTED_TABLE_DATA exactly.
DATASET_K_STAR = {
    'Digits':             10,
    'MNIST-test':         10,
    'Ecoli':              8,
    'Letter':             26,
    'Iris':               3,
    'Fashion-MNIST-test': 10,
    'Wine':                3,
    'Libras':             15,
    'Vowel':               11,
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
    suffix = '' if metric == 'ACC' else f'_{metric}'
    return os.path.join(DATA_DIR, f'realdata_fixed_results_{EXPERIMENT_TAG}{suffix}.pkl')


def load_results(metric):
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


def rho_row(results, cm, dname):
    """Row of 10 values for this (cm, dataset) -- NaN for any dataset not
    yet run at all, or any method not yet recorded for a dataset that IS
    present (e.g. still running remotely). Renders as '--' in the table
    (see to_latex) rather than blocking the whole table until every
    dataset finishes -- this document is meant to be regenerated and
    refreshed incrementally as more results come in."""
    by_method = results.get(cm, {}).get(dname, {})
    return [float(by_method[m]) if m in by_method else np.nan for m in ALL_METHODS]


def to_latex(metric, cm, rows_by_dataset, n_present, n_expected):
    cm_label = CM_LABELS.get(cm, cm)
    partial_note = (
        '' if n_present == n_expected else
        rf" \textbf{{PARTIAL: {n_present}/{n_expected} datasets done so far, "
        rf"rest shown as --, refresh after more results are retrieved.}}"
    )
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\small')
    lines.append(
        rf"\caption{{Spearman's $\rho$ with {metric} for the nine real "
        rf"benchmark datasets, {cm_label}. Bold indicates the highest "
        rf"value in each row.{partial_note}}}"
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
            f'{dname} & {DATASET_K_STAR[dname]} & ' + ' & '.join(cells) + r' \\'
        )
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append('}')
    lines.append(rf'\label{{tab:realdata_{metric}_{cm}}}')
    lines.append(r'\end{table}')
    return '\n'.join(lines) + '\n'


DOC_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}

\title{Benchmark\_data\_real: Full Results Tables\\
\large (auto-generated by generate\_realdata\_tables.py -- do not edit by hand)}
\author{}
\date{}

\begin{document}
\maketitle

\section*{Experimental setup}

Nine widely used real benchmark datasets (Digits, MNIST-test, Ecoli,
Letter, Iris, Fashion-MNIST-test, Wine, Libras, Vowel -- summary
statistics in main.tex's Table~\texttt{tab:data}) are each clustered
independently under three algorithms: K-means (matching the paper's
sec:real protocol), Spectral Clustering (nearest-neighbors affinity), and
Agglomerative Clustering (Ward linkage). As with Part B, each dataset is
fixed rather than simulated, so there is no replication -- one clustering
per (dataset, method) pair, with a fixed random seed.

For a dataset with true cluster count $K^*$, candidate $K$ is swept over
$K=2,\ldots,\max(25,1.75K^*)+10$. The evaluation scope for every reported
$\rho$ (all 10 indices, not just the $r$NN ones) is the narrower range
$K=2,\ldots,\max(25,1.75K^*)$ -- the reference paper's own $K$-range cap.
The wider $+10$-buffered range is used only to select $r^*$ for the
$r$NN-affinity Hard/Soft Silhouette variants
(\texttt{select\_r\_topK\_spearman}, top-$m$ monotonic-decrease
criterion, $m=35$), following the same convention as Part A/B and the
paper's Algorithm~1.

Each table reports the Spearman rank correlation $\rho$ between each of
the 10 validity indices' Silhouette-style score (evaluated across the $K$
sweep) and the named external metric, for one clustering method. Bold
indicates the highest value in each row. Tables are grouped by external
metric, with one table per clustering method within each group. The
three large datasets (MNIST-test, Fashion-MNIST-test, Letter) are
computed via separate, isolated remote jobs (see SCALE WARNING in
\texttt{Benchmark\_data\_real.py}) and staged in incrementally as they
finish -- \textbf{tables are generated with whatever is currently
available}: a dataset not yet run at all shows \texttt{--} for every
column, and specific indices that fail to produce a finite value for a
completed dataset (see the VIASCKDE/DBCV numerical-stability notes for
the two largest real datasets) show \texttt{--} in just that cell. Each
table's caption flags how many of the 9 datasets are done so far;
rerun this script after retrieving more results to refresh it.

"""

DOC_CLOSING = r"""\end{document}
"""


def compute_rnn_performance_summary(all_raw_data):
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

    if not any(s.get('n_total', 0) for s in stats.values()):
        lines.append(
            r"No (external metric, clustering method) combination has all 9 "
            r"datasets complete yet -- rerun after more results are retrieved "
            r"and merged (see the module docstring)."
        )
        lines.append('')
        return '\n'.join(lines) + '\n'

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
    all_tables = {}    # metric -> cm -> tex string
    all_raw_data = {}  # metric -> cm -> dataset -> list of 10 floats

    for metric in METRICS:
        results = load_results(metric)
        if results is None:
            print(f'{metric}: no results file found ({results_path_for(metric)}) -- skipping')
            continue

        for cm in results:
            rows_by_dataset = {
                dname: rho_row(results, cm, dname) for dname in DATASET_NAMES
            }
            n_present = sum(
                1 for dname in DATASET_NAMES
                if dname in results.get(cm, {})
            )
            tex = to_latex(metric, cm, rows_by_dataset, n_present, len(DATASET_NAMES))
            status = 'COMPLETE' if n_present == len(DATASET_NAMES) else f'PARTIAL ({n_present}/{len(DATASET_NAMES)} datasets)'
            print(f'{metric} / {cm}: {status} -- table included with -- for anything missing')

            all_tables.setdefault(metric, {})[cm] = tex
            all_raw_data.setdefault(metric, {})[cm] = rows_by_dataset

    combined = build_combined_document(all_tables, all_raw_data)
    combined_path = os.path.join(OUT_DIR, 'realdata_all_tables.tex')
    with open(combined_path, 'w') as f:
        f.write(combined)
    n_tables = sum(len(cms) for cms in all_tables.values())
    print(f'\nWrote combined standalone document: {combined_path} ({n_tables} tables)')
    print('Compile with: pdflatex realdata_all_tables.tex  (run from the tables/ folder)')


if __name__ == '__main__':
    main()

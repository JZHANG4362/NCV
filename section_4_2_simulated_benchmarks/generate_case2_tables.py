r"""
Generates LaTeX tables of Spearman's rho (per case, per index) from
partA_type2_grid_comparison.py's results pickles -- one MEAN table and one
MEDIAN table (across the N_REP=100 replications) per (external_metric,
clustering_method) combination that is FULLY complete (all 12 current
cases, N_REP=100 each). Matches the exact structure of Table~\ref{tab:case2}
in main.tex (same columns, same bold-row-max styling), and mirrors
generate_nonconvex_tables.py's combined-document format (Part B) -- the
one difference from Part B is that Part A has replications, so each cell
is summarized two ways (mean and median) rather than being a single fixed
value.

Usage: run from this folder after retrieving the results/*.pkl files from
the cluster (see cluster_setup_commands_partA.docx STEP 8):

    python generate_case2_tables.py

Writes ONE combined, standalone, compilable document to
tables/case2_all_tables.tex, and prints a completeness report for every
(metric, clustering_method) combination that isn't fully complete yet
(missing/partial cases are skipped, not silently included with gaps).
"""

import os
import pickle

import numpy as np

DATA_DIR = 'data'
OUT_DIR = 'tables'
os.makedirs(OUT_DIR, exist_ok=True)

# Must match partA_type2_grid_comparison.py's own settings exactly.
K_STAR_VALUES = [6, 30]
D_VALUES = [2, 10, 200]
COMPACTNESS_VALUES = ['compact', 'sparse']
N_REP_EXPECTED = 100

CASES = [
    (k_star, D, comp)
    for k_star in K_STAR_VALUES
    for D in D_VALUES
    for comp in COMPACTNESS_VALUES
]

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
SCORE_DIRECTION_CONVENTION = 'larger_is_better_v1'

CM_LABELS = {
    'KMeans': 'K-means clustering',
    'Spectral': 'Spectral Clustering',
    'Agglomerative': 'Agglomerative Clustering',
}


def case_key(k_star, D, comp):
    return f'k{k_star}_D{D}_{comp}'


def results_path_for(metric):
    '''Canonical (merged, untagged) results pickle path for one external
    metric -- these are only produced by merge_case2_results.py, never
    written directly by this script. The "nonconvex_..." prefix is a
    historical filename quirk carried over from an earlier project
    structure; Case 2 is the Gaussian-mixture benchmark, not the
    nonconvex-shapes case (Case 4).'''
    fname = ('nonconvex_case_comparison_results_nonself_oriented.pkl' if metric == 'ACC'
             else f'nonconvex_case_comparison_results_nonself_oriented_{metric}.pkl')
    return os.path.join(DATA_DIR, fname)


def load_results(metric):
    '''Load one metric's merged results dict, guarding against accidentally
    mixing in a pickle written under a different (e.g. pre-correction)
    score-direction convention -- raises rather than silently producing
    tables with an inconsistent sign convention for NID/WB.'''
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


def case_is_complete(results, cm, ckey):
    '''True iff every one of the 10 validity indices has reached the full
    N_REP_EXPECTED=100 replications for this (clustering_method, case) --
    used to decide whether a (metric, clustering_method) pair is safe to
    tabulate at all (see main() below: partial data is skipped entirely,
    never shown with silent gaps).'''
    if cm not in results or ckey not in results[cm]:
        return False
    for method in ALL_METHODS:
        if len(results[cm][ckey].get(method, [])) < N_REP_EXPECTED:
            return False
    return True


def summary_rows(results, cm, ckey, stat_fn):
    """stat_fn: np.mean or np.median, applied over the N_REP replications
    (finite values only) for each of the 10 methods."""
    vals = []
    for method in ALL_METHODS:
        arr = np.array(results[cm][ckey].get(method, []), dtype=float)
        arr = arr[np.isfinite(arr)]
        vals.append(float(stat_fn(arr)) if len(arr) else np.nan)
    return vals


def to_latex(metric, cm, stat_label, rows_by_case):
    cm_label = CM_LABELS.get(cm, cm)
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\small')
    lines.append(
        rf"\caption{{{stat_label.capitalize()} Spearman's $\rho$ with "
        rf"{metric} across $N_{{\text{{rep}}}}={N_REP_EXPECTED}$ "
        rf"replications, {cm_label}, for the 12 cases in Case 2. Bold "
        rf"indicates the highest value in each row.}}"
    )
    lines.append(r'\centering')
    lines.append(r'\resizebox{\textwidth}{!}{')
    lines.append(r'\begin{tabular}{' + 'c' * (3 + len(ALL_METHODS)) + '}')
    lines.append(r'\hline')
    header = ['$K^*$', '$D$', 'Compactness'] + [METHOD_LABELS[m] for m in ALL_METHODS]
    lines.append(' & '.join(header) + r' \\')
    lines.append(r'\hline')
    for (k_star, D, comp) in CASES:
        # One table row per case; `vals` holds one summary rho per index,
        # in ALL_METHODS order. The row's best (largest, since every rho
        # here is already oriented larger-is-better) value is bolded --
        # ties within 1e-9 are all bolded together rather than picking an
        # arbitrary single winner.
        vals = rows_by_case[case_key(k_star, D, comp)]
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
            f'$K^*={k_star}$ & {D} & {comp} & ' + ' & '.join(cells) + r' \\'
        )
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append('}')
    lines.append(rf'\label{{tab:case2_{metric}_{cm}_{stat_label}}}')
    lines.append(r'\end{table}')
    return '\n'.join(lines) + '\n'


DOC_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}

\title{Simulated\_data\_comparison: Full Results Tables\\
\large (auto-generated by generate\_case2\_tables.py -- do not edit by hand)}
\author{}
\date{}

\begin{document}
\maketitle

\section*{Experimental setup}

Synthetic Type-2-style Gaussian-blob data are generated across a 12-case
grid: true cluster count $K^*\in\{6,30\}$, feature dimension
$D\in\{2,10,200\}$, and
compactness (cluster std) $\in\{\text{compact}=0.1,\ \text{sparse}=0.8\}$.
Each of the 12 cases is replicated $N_{\text{rep}}=100$ times with fresh
data redrawn each replication, under each clustering method considered
(K-means, Spectral Clustering, Agglomerative Clustering -- computed
independently, only cases with all 100 replications complete are
tabulated here).

For a case with true $K^*$, candidate $K$ is swept over
$K=2,\ldots,\max(25,1.75K^*)+10$. The evaluation scope for every reported
$\rho$ (all 10 indices, not just the $r$NN ones) is the narrower range
$K=2,\ldots,\max(25,1.75K^*)$ -- the reference paper's own $K$-range cap.
The wider $+10$-buffered range is used only to select $r^*$ for the
$r$NN-affinity Hard/Soft Silhouette variants
(\texttt{select\_r\_topK\_spearman}, top-$m$ monotonic-decrease
criterion, $m=35$), per the paper's Algorithm~1.

Because each case has $N_{\text{rep}}=100$ replications rather than a
single fixed value (unlike Part B's real, fixed datasets), each
(external metric, clustering method) combination below is summarized
TWICE: once using the \textbf{mean} $\rho$ across replications, and once
using the \textbf{median} $\rho$ -- shown as two separate tables, since
outlier-sensitive means and robust medians can highlight different
patterns (e.g. a heavy-tailed distribution of $\rho$ for a given index).
Bold indicates the highest value in each row, independently per table.

"""

DOC_CLOSING = r"""\end{document}
"""


def compute_rnn_performance_summary(all_raw_data):
    """all_raw_data: dict stat_label -> dict metric -> dict cm -> dict case
    -> list of 10 floats (order = ALL_METHODS). Returns, per stat_label,
    per-variant counts of: how often rNN_Hard/rNN_Soft is the single best
    index in a row, how often it's in the top 3, and how often it beats
    the original Silhouette score -- tallied across every (metric,
    clustering_method, case) row in the tables actually generated."""
    idx_sil  = ALL_METHODS.index('Silhouette')
    idx_hard = ALL_METHODS.index('rNN_Hard')
    idx_soft = ALL_METHODS.index('rNN_Soft')

    all_stats = {}
    for stat_label, by_metric in all_raw_data.items():
        stats = {}
        for variant, idx in [('rNN_Hard', idx_hard), ('rNN_Soft', idx_soft)]:
            n_total = n_best = n_top3 = n_beats_sil = 0
            for metric, by_cm in by_metric.items():
                for cm, by_case in by_cm.items():
                    for ckey, vals in by_case.items():
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
        all_stats[stat_label] = stats
    return all_stats


def performance_summary_to_latex(all_stats):
    def pct(n, total):
        return 100.0 * n / total if total else float('nan')

    lines = [r'\section*{$r$NN-affinity Silhouette performance summary}', '']
    lines.append(
        r"Tallied across every (external metric, clustering method, case) "
        r"row in the tables below -- i.e. every row of every table "
        r"currently in this document, not a fixed hand-picked subset:"
    )
    for stat_label in ['mean', 'median']:
        stats = all_stats.get(stat_label, {})
        if not any(s.get('n_total', 0) for s in stats.values()):
            continue
        lines.append(rf'\subsection*{{Using the {stat_label} across replications}}')
        lines.append(r'\begin{itemize}')
        for variant, label in [('rNN_Hard', r'$r$NN-Hard'), ('rNN_Soft', r'$r$NN-Soft')]:
            s = stats.get(variant)
            if not s or s['n_total'] == 0:
                continue
            lines.append(
                rf"\item \textbf{{{label}}}: out of {s['n_total']} rows, it is "
                rf"the single best-performing index in {s['n_best']} "
                rf"({pct(s['n_best'], s['n_total']):.0f}\%), ranks in the top "
                rf"three (of 10 indices) in {s['n_top3']} "
                rf"({pct(s['n_top3'], s['n_total']):.0f}\%), and outperforms "
                rf"the original (global) Silhouette score in "
                rf"{s['n_beats_sil']} ({pct(s['n_beats_sil'], s['n_total']):.0f}\%)."
            )
        lines.append(r'\end{itemize}')
    lines.append('')
    return '\n'.join(lines) + '\n'


def build_combined_document(all_tables, all_raw_data):
    """all_tables: dict stat_label -> dict metric -> dict cm -> tex table string.
    all_raw_data: dict stat_label -> dict metric -> dict cm -> dict case ->
    list of 10 floats -- used for the performance summary."""
    parts = [DOC_PREAMBLE]

    all_stats = compute_rnn_performance_summary(all_raw_data)
    parts.append(performance_summary_to_latex(all_stats))

    for metric in METRICS:
        has_any = any(
            metric in all_tables[stat_label] and all_tables[stat_label][metric]
            for stat_label in ['mean', 'median']
        )
        if not has_any:
            continue
        parts.append(f'\\section{{External metric: {metric}}}\n')
        for cm in ['KMeans', 'Spectral', 'Agglomerative']:
            for stat_label in ['mean', 'median']:
                tex = all_tables.get(stat_label, {}).get(metric, {}).get(cm)
                if tex is None:
                    continue
                parts.append(tex)
                parts.append('')
        parts.append('\\clearpage\n')
    parts.append(DOC_CLOSING)
    return '\n'.join(parts)


def main():
    all_tables = {'mean': {}, 'median': {}}      # stat_label -> metric -> cm -> tex string
    all_raw_data = {'mean': {}, 'median': {}}     # stat_label -> metric -> cm -> case -> list of 10 floats

    # Outer loop: one merged results pickle per external metric (ACC,
    # Jaccard, SS3, ARI, NMI, Powers, NID). Inner loop: each clustering
    # method found in that pickle (KMeans/Spectral/Agglomerative) is
    # tabulated independently -- a (metric, clustering_method) pair is only
    # ever included if ALL 12 cases have reached N_REP_EXPECTED=100 reps for
    # every index (case_is_complete), so partially-finished runs never
    # produce a table with silent gaps.
    for metric in METRICS:
        results = load_results(metric)
        if results is None:
            print(f'{metric}: no results file found ({results_path_for(metric)}) -- skipping')
            continue

        for cm in results:
            missing = [
                case_key(*case) for case in CASES
                if not case_is_complete(results, cm, case_key(*case))
            ]
            if missing:
                print(f'{metric} / {cm}: INCOMPLETE -- skipped '
                      f'({len(missing)}/{len(CASES)} cases not yet at '
                      f'N_REP={N_REP_EXPECTED}): {missing}')
                continue

            print(f'{metric} / {cm}: COMPLETE')
            # Each complete (metric, cm) combination gets TWO tables: one
            # summarizing each case's 100 replicate rho's by their mean,
            # one by their median (see DOC_PREAMBLE for why both are shown).
            for stat_label, stat_fn in [('mean', np.mean), ('median', np.median)]:
                rows_by_case = {
                    case_key(*case): summary_rows(results, cm, case_key(*case), stat_fn)
                    for case in CASES
                }
                tex = to_latex(metric, cm, stat_label, rows_by_case)
                all_tables[stat_label].setdefault(metric, {})[cm] = tex
                all_raw_data[stat_label].setdefault(metric, {})[cm] = rows_by_case

    # Assemble every generated table (plus the rNN performance summary) into
    # one standalone, pdflatex-compilable .tex document.
    combined = build_combined_document(all_tables, all_raw_data)
    combined_path = os.path.join(OUT_DIR, 'case2_all_tables.tex')
    with open(combined_path, 'w') as f:
        f.write(combined)
    n_tables = sum(
        len(cms) for stat_label in all_tables for cms in all_tables[stat_label].values()
    )
    print(f'\nWrote combined standalone document: {combined_path} ({n_tables} tables)')
    print('Compile with: pdflatex case2_all_tables.tex  (run from the tables/ folder)')


if __name__ == '__main__':
    main()

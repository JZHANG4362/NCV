r"""
Improvement summary: for each of the 3 cases, how many datasets show an
internal validity index -- applied to t-SNE- or UMAP-TRANSFORMED data --
improving on THAT SAME INDEX applied to the ORIGINAL (untransformed)
data. This is a SELF comparison (index-on-embedding vs. that index's own
raw-X value), NOT a comparison against a shared Silhouette baseline.

Silhouette is included as one of the candidates, on equal footing with
the 8 alternative indices (so "does embedding help Silhouette beat its
own raw-X self?" is answered the same way as for every other index).
Silhouette itself -- its Original/t-SNE/UMAP values -- is NOT computed
by this manifold package (only rNN_Hard and the 8 alternative indices
are). It is pulled in from the SIBLING package codes_new_manifold_tuned_
embedding_orig_kgrid, which runs the identical Algorithm-1 tuning design
for exactly this: 'Silhouette' (raw X), 'TSNE_Ordinary' (perplexity
tuned), 'UMAP_Ordinary' (n_neighbors tuned) -- see generate_manifold_
bar_summary_tex.py for the same source used to draw Silhouette as a full
bar-chart triplet. Silhouette is folded into the SAME per-index
machinery as the 8 alternative indices (ROW_NAMES = 'Silhouette' + the 8
indices), so it gets its own row in every per-index table and
participates in the case-level "any candidate improves" pool.

Two SCENARIOS, differing ONLY in where the t-SNE/UMAP (embedded) value
comes from -- the ORIGINAL/raw-X reference value is the SAME in both
(it never depends on embedding-parameter tuning at all):
  TUNED  -- manifold_fixed_results_*.pkl: Algorithm 1's ACTUAL per-dataset
            selected embedding parameter (data-driven, never looks at
            ACC) -- the achievable, deployable scenario.
  BEST SELECTION (oracle) -- manifold_param_sensitivity_results_*.pkl:
            the best possible rho for THIS dataset's own curve (ACC-
            informed oracle, an upper bound on what Algorithm 1 could
            ever find) -- same source as generate_best_params_detailed_
            tex.py.

Within each scenario, all 3 clustering methods (KMeans/Spectral/
Agglomerative) are combined into a SINGLE table -- no averaging across
methods, just placed side by side (case-level table: as column-groups)
or stacked as bolded row-group sections (per-index table). t-SNE and
UMAP representations are always shown as separate columns, never
pooled/stacked together. Each scenario has a case-level table (does AT
LEAST ONE of the 9 candidates improve on its own Original value?), and a
per-index breakdown table (one row per candidate) underneath it.

Every cell is the FRACTION of that case's datasets (as a decimal, e.g.
0.50, not a fraction string) where the candidate's embedded rho beats
that SAME candidate's own Original (raw-X) rho, for that (clustering
method, dataset).

Run this LOCALLY after retrieving data/ from the cluster and (for Case 4)
running merge_manifold_case4_results.py -- in BOTH this package and the
sibling section_4_6_local_embeddings/silhouette:

    python generate_improvement_summary_tex.py

Writes tables/improvement_summary.tex. Compile with:
    cd tables && pdflatex improvement_summary.tex
"""

import os
import pickle

import numpy as np

from generate_manifold_tsne_umap_tex import (
    DATA_DIR, EXPERIMENT_TAG, SCORE_DIRECTION_CONVENTION, CLUSTERING_METHODS,
    CM_LABELS, CASE2_ORDER, CASE3_ORDER, CASE4_ORDER,
)
from generate_oracle_landscape_plots import INDEX_NAMES

OUT_DIR = 'tables'
os.makedirs(OUT_DIR, exist_ok=True)

REPS = ['TSNE', 'UMAP']
SHORT_TITLES = {'case2': 'Case 2', 'case3': 'Case 3', 'case4': 'Case 4'}
ROW_NAMES = ['Silhouette'] + INDEX_NAMES

SIL_ROOT = os.path.join('..', 'silhouette')
SIL_TAG = 'tsne_umap_tuned_embedding_origkgrid_v1'
SIL_METHOD_KEYS = {'Original': 'Silhouette', 'TSNE': 'TSNE_Ordinary', 'UMAP': 'UMAP_Ordinary'}
SIL_TUNED_PATHS = {
    tag: os.path.join(SIL_ROOT, 'data', tag, f'manifold_fixed_results_{SIL_TAG}.pkl')
    for tag in ('case2', 'case3', 'case4')
}
SIL_ORACLE_PATHS = {
    tag: os.path.join(SIL_ROOT, 'data', tag, f'manifold_param_sensitivity_results_{SIL_TAG}.pkl')
    for tag in ('case2', 'case3', 'case4')
}

CASES = [
    dict(tag='case2', title='Case 2 -- Simulated Data', order=CASE2_ORDER,
         tuned_path=os.path.join(DATA_DIR, 'case2', f'manifold_fixed_results_{EXPERIMENT_TAG}.pkl'),
         oracle_path=os.path.join(DATA_DIR, 'case2', f'manifold_param_sensitivity_results_{EXPERIMENT_TAG}.pkl'),
         sil_tuned_path=SIL_TUNED_PATHS['case2'], sil_oracle_path=SIL_ORACLE_PATHS['case2']),
    dict(tag='case3', title='Case 3 -- Nonconvex Data', order=CASE3_ORDER,
         tuned_path=os.path.join(DATA_DIR, 'case3', f'manifold_fixed_results_{EXPERIMENT_TAG}.pkl'),
         oracle_path=os.path.join(DATA_DIR, 'case3', f'manifold_param_sensitivity_results_{EXPERIMENT_TAG}.pkl'),
         sil_tuned_path=SIL_TUNED_PATHS['case3'], sil_oracle_path=SIL_ORACLE_PATHS['case3']),
    dict(tag='case4', title='Case 4 -- Real Labeled Data', order=CASE4_ORDER,
         tuned_path=os.path.join(DATA_DIR, 'case4', f'manifold_fixed_results_{EXPERIMENT_TAG}.pkl'),
         oracle_path=os.path.join(DATA_DIR, 'case4', f'manifold_param_sensitivity_results_{EXPERIMENT_TAG}.pkl'),
         sil_tuned_path=SIL_TUNED_PATHS['case4'], sil_oracle_path=SIL_ORACLE_PATHS['case4']),
]


def load_pickle(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_manifold_results(path):
    payload = load_pickle(path)
    if payload is None:
        return None
    if payload.get('score_direction_convention') != SCORE_DIRECTION_CONVENTION:
        raise ValueError(f'{path} does not use the expected score-direction convention.')
    return payload['results']


def merge_silhouette_fixed(fixed_results, sil_fixed_results, order):
    """Injects 'Silhouette_Original'/'_TSNE'/'_UMAP' into fixed_results
    (in place), from the sibling package's own 'Silhouette'/
    'TSNE_Ordinary'/'UMAP_Ordinary' keys -- so rho_tuned() treats
    'Silhouette' exactly like any other index, no special-casing
    needed downstream."""
    for cm in CLUSTERING_METHODS:
        sil_by_dataset = sil_fixed_results.get(cm, {})
        for dname in order:
            sil_row = sil_by_dataset.get(dname, {})
            row = fixed_results.setdefault(cm, {}).setdefault(dname, {})
            for rep, sil_key in SIL_METHOD_KEYS.items():
                row[f'Silhouette_{rep}'] = sil_row.get(sil_key)


def merge_silhouette_sensitivity(sens_results, sil_sens_results, order):
    """Injects a 'Silhouette' curve into sens_results' 'perplexity'/
    'n_neighbors' dicts (in place). The sibling package has no index
    dimension of its own (one flat curve per cm/dataset), so that whole
    curve becomes the 'Silhouette' entry -- rho_oracle() then treats it
    exactly like any other index's curve."""
    for cm in CLUSTERING_METHODS:
        sil_by_dataset = sil_sens_results.get(cm, {})
        for dname in order:
            sil_entry = sil_by_dataset.get(dname, {})
            entry = sens_results.setdefault(cm, {}).setdefault(dname, {})
            entry.setdefault('perplexity', {})['Silhouette'] = sil_entry.get('perplexity', {})
            entry.setdefault('n_neighbors', {})['Silhouette'] = sil_entry.get('n_neighbors', {})


def get_original(fixed_results, cm, dname, idx):
    """That candidate's own Original (raw-X) rho -- the SAME value
    regardless of scenario, since Original never depends on embedding-
    parameter tuning. Works for 'Silhouette' too once merge_silhouette_
    fixed() has injected 'Silhouette_Original' into fixed_results."""
    v = fixed_results.get(cm, {}).get(dname, {}).get(f'{idx}_Original')
    return v if v is not None and np.isfinite(v) else None


def rho_tuned(fixed_results, cm, dname, idx, rep):
    """Algorithm 1's actual selected rho for this (cm, dataset, index,
    representation), or None if not present/finite."""
    v = fixed_results.get(cm, {}).get(dname, {}).get(f'{idx}_{rep}')
    return v if v is not None and np.isfinite(v) else None


def rho_oracle(sens_results, cm, dname, idx, rep):
    """Oracle-best rho (this dataset's own curve peak) for this (cm,
    dataset, index, representation), or None if not present/finite.
    rep='TSNE' reads the perplexity curve, rep='UMAP' the n_neighbors
    curve -- matching generate_best_params_detailed_tex.py exactly."""
    entry = sens_results.get(cm, {}).get(dname, {})
    curve_key = 'perplexity' if rep == 'TSNE' else 'n_neighbors'
    curve = entry.get(curve_key, {}).get(idx, {})
    finite = [r for r in curve.values() if r is not None and np.isfinite(r)]
    return max(finite) if finite else None


def summarize_index(case, embed_results, original_results, cm, idx, rep, rho_fn):
    """(n_total, n_improved, n_missing) for ONE candidate, ONE
    representation, ONE clustering method, across every dataset in this
    case -- comparing that candidate's embedded rho against its OWN
    Original (raw-X) rho, a pure self comparison."""
    n_total = len(case['order'])
    n_improved = 0
    n_missing = 0
    for dname in case['order']:
        embedded = rho_fn(embed_results, cm, dname, idx, rep)
        orig = get_original(original_results, cm, dname, idx)
        if embedded is not None and orig is not None:
            if embedded > orig:
                n_improved += 1
        else:
            n_missing += 1
    return n_total, n_improved, n_missing


def summarize_best_of_all(case, embed_results, original_results, cm, rep, rho_fn, row_names):
    """(n_total, n_improved, n_missing) for whether AT LEAST ONE candidate
    in row_names improves on its OWN Original value (within one
    representation, one clustering method), across every dataset in this
    case. Equivalent to: does the best (embedded - own Original) delta,
    over row_names, exceed zero?"""
    n_total = len(case['order'])
    n_improved = 0
    n_missing = 0
    for dname in case['order']:
        deltas = []
        for idx in row_names:
            embedded = rho_fn(embed_results, cm, dname, idx, rep)
            orig = get_original(original_results, cm, dname, idx)
            if embedded is not None and orig is not None:
                deltas.append(embedded - orig)
        if deltas:
            if max(deltas) > 0:
                n_improved += 1
        else:
            n_missing += 1
    return n_total, n_improved, n_missing


def decimal_cell(stat):
    """A dataset with no valid embedded/Original pair (e.g. DBCV on
    Fashion-MNIST-test, whose raw-X value cannot be computed) is fully
    EXCLUDED from consideration -- both numerator and denominator -- not
    just flagged while still counted against the full case total."""
    n_total, n_improved, n_missing = stat
    n_valid = n_total - n_missing
    if n_valid <= 0:
        return '--'
    note = r'$^\dagger$' if n_missing else ''
    return f'{n_improved / n_valid:.2f}{note}'


def to_case_table(label, tex_label, rows_by_cm):
    """rows_by_cm: {cm: [(case_title, n_total, tsne_stat, umap_stat), ...]}
    (same case order for every cm). ONE table combining all 3 clustering
    methods as column-groups: Case | KMeans(t-SNE,UMAP) |
    Spectral(t-SNE,UMAP) | Agglomerative(t-SNE,UMAP)."""
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(
        rf"\caption{{{label}: number of datasets where AT LEAST ONE of "
        rf"the 8 internal validity indices or Silhouette, applied with "
        rf"t-SNE- or UMAP-transformed data, results in an improvement "
        rf"over THAT SAME index/Silhouette applied to the original "
        rf"(untransformed) data -- shown as a decimal fraction of the "
        rf"total dataset count for that case (the number in "
        rf"parentheses). KMeans, Spectral, and Agglomerative shown side "
        rf"by side.}}"
    )
    lines.append(rf'\label{{{tex_label}}}')
    n_cm = len(CLUSTERING_METHODS)
    lines.append(r'\resizebox{\textwidth}{!}{%')
    lines.append(r'\begin{tabular}{l' + 'cc' * n_cm + '}')
    lines.append(r'\hline')
    group_header = [' '] + [
        rf'\multicolumn{{2}}{{c}}{{{CM_LABELS.get(cm, cm)}}}' for cm in CLUSTERING_METHODS
    ]
    lines.append(' & '.join(group_header) + r' \\')
    cmidrules = ' '.join(
        rf'\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(n_cm)
    )
    lines.append(cmidrules)
    header = ['Case (\\# datasets)'] + ['t-SNE', 'UMAP'] * n_cm
    lines.append(' & '.join(header) + r' \\')
    lines.append(r'\hline')
    any_missing = False
    n_rows = len(next(iter(rows_by_cm.values())))
    for i in range(n_rows):
        title, n_total = rows_by_cm[CLUSTERING_METHODS[0]][i][:2]
        cells = []
        for cm in CLUSTERING_METHODS:
            _, _, tsne_stat, umap_stat = rows_by_cm[cm][i]
            any_missing = any_missing or tsne_stat[2] or umap_stat[2]
            cells += [decimal_cell(tsne_stat), decimal_cell(umap_stat)]
        lines.append(rf'{title} ({n_total}) & ' + ' & '.join(cells) + r' \\')
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append('}')
    if any_missing:
        lines.append(r'\vspace{2pt}')
        lines.append(
            r'\par\footnotesize $^\dagger$ one or more datasets had no '
            r'candidate with both a finite embedded value AND a finite '
            r'own-Original value for that clustering method/'
            r'representation, and were excluded entirely (from both the '
            r'numerator and the denominator).'
        )
    lines.append(r'\end{table}')
    return '\n'.join(lines) + '\n'


def to_index_table(label, tex_label, per_index_by_cm, case_headers, row_names):
    """per_index_by_cm: {cm: {idx: [(tsne_stat, umap_stat), ...]}} (case
    order matching case_headers = [(short_title, n_total), ...]). ONE
    table combining all 3 clustering methods as bolded row-group
    sections (same 6 case x representation columns throughout); one row
    per name in row_names (the 8 alternative indices, plus
    'Silhouette') within each clustering-method section."""
    n_cases = len(case_headers)
    n_cols = 1 + 2 * n_cases
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(
        rf"\caption{{{label}, per-index breakdown (including "
        rf"Silhouette's own t-SNE/UMAP versions): number of datasets "
        rf"where THAT ROW's internal validity index, applied with "
        rf"t-SNE- or UMAP-transformed data, results in an improvement "
        rf"over THAT SAME index applied to the original (untransformed) "
        rf"data -- shown as a decimal fraction of the total dataset "
        rf"count for that case (the number in each column header). "
        rf"KMeans, Spectral, and Agglomerative stacked in one table. "
        rf"DBCV's raw-$X$ value is unavailable for Fashion-MNIST-test "
        rf"(Case 4, hdbscan infeasible on the 784-dim representation) "
        rf"-- this is NOT ignored or excluded here: DBCV-t-SNE/UMAP are "
        rf"computed normally for that dataset, so those cells report "
        rf"missing ($^\dagger$) since there is no raw-$X$ DBCV value to "
        rf"compare against, which is the correct, honest outcome rather "
        rf"than a silent omission.}}"
    )
    lines.append(rf'\label{{{tex_label}}}')
    lines.append(r'\resizebox{\textwidth}{!}{%')
    lines.append(r'\begin{tabular}{l' + 'c' * (2 * n_cases) + '}')
    lines.append(r'\hline')
    lines.append(
        r' & \multicolumn{' + str(n_cases) + r'}{c}{t-SNE} & '
        r'\multicolumn{' + str(n_cases) + r'}{c}{UMAP} \\'
    )
    lines.append(rf'\cmidrule(lr){{2-{1 + n_cases}}} \cmidrule(lr){{{2 + n_cases}-{1 + 2 * n_cases}}}')
    header = (['Index']
              + [rf'{title} ({n})' for title, n in case_headers]
              + [rf'{title} ({n})' for title, n in case_headers])
    lines.append(' & '.join(header) + r' \\')
    any_missing = False
    for cm in CLUSTERING_METHODS:
        lines.append(r'\hline')
        lines.append(rf'\multicolumn{{{n_cols}}}{{l}}{{\textbf{{{CM_LABELS.get(cm, cm)}}}}} \\')
        lines.append(r'\hline')
        per_index = per_index_by_cm[cm]
        for idx in row_names:
            stats = per_index[idx]
            tsne_cells = [decimal_cell(s[0]) for s in stats]
            umap_cells = [decimal_cell(s[1]) for s in stats]
            any_missing = any_missing or any(s[0][2] or s[1][2] for s in stats)
            lines.append(f'{idx} & ' + ' & '.join(tsne_cells + umap_cells) + r' \\')
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append('}')
    if any_missing:
        lines.append(r'\vspace{2pt}')
        lines.append(
            r'\par\footnotesize $^\dagger$ one or more datasets had no '
            r'candidate with both a finite embedded value AND a finite '
            r'own-Original value for that index/representation, and '
            r'were excluded entirely (from both the numerator and the '
            r'denominator) -- this is exactly the DBCV / Fashion-MNIST-'
            r'test case described above.'
        )
    lines.append(r'\end{table}')
    return '\n'.join(lines) + '\n'


DOC_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}

\title{Embedding Improvement Summary: t-SNE/UMAP vs. Own Original\\
\large (auto-generated by generate\_improvement\_summary\_tex.py -- do not edit by hand)}
\author{}
\date{}

\begin{document}
\maketitle

\section*{What this reports}

For each of the 3 cases, what fraction of its datasets show an internal
validity index, APPLIED TO t-SNE- OR UMAP-TRANSFORMED DATA, improving on
THAT SAME INDEX applied to the ORIGINAL (untransformed) data. This is a
SELF comparison -- index-on-embedding vs.\ that index's own raw-$X$
value -- NOT a comparison against a shared Silhouette baseline. Every
cell is a DECIMAL fraction of datasets improved (e.g.\ 0.50), never a
raw count or a fraction string.

\textbf{Silhouette is included as one of the candidates,} on equal
footing with the 8 alternative indices -- ``does embedding the data help
Silhouette beat its own raw-$X$ self?'' is answered the same way as for
every other index. This manifold package never computes Silhouette
itself (only rNN\_Hard and the 8 alternative indices), so its Original/
t-SNE/UMAP values are pulled in from the sibling package
\texttt{codes\_new\_manifold\_tuned\_embedding\_orig\_kgrid} (which runs
the identical Algorithm-1 tuning design for ordinary Euclidean
Silhouette on a t-SNE or UMAP embedding).

t-SNE and UMAP are always kept as SEPARATE columns (never pooled
together). The three clustering methods (KMeans, Spectral,
Agglomerative) are combined into a SINGLE table per scenario -- never
averaged together, just placed side by side or stacked -- so every
number is still a direct, unaveraged comparison for one specific (case,
clustering method, representation) combination.

Each scenario has two tables: a case-level table (does AT LEAST ONE of
the 8 alternative indices or Silhouette -- 9 candidates, within that
representation -- improve on its OWN Original value?), with the three
clustering methods placed side by side as column-groups; and a
per-index breakdown table underneath it (does THIS SPECIFIC candidate,
within that representation, improve on its OWN Original value?), with
the three clustering methods stacked as bolded row-group sections (same
6 columns throughout) -- the per-index table is the stricter, more
informative view, since the case-level ``any candidate'' bar is easy to
clear by chance.

\textbf{Tuned} (Section~\ref{sec:tuned}) uses Algorithm~1's ACTUAL
selected embedding parameter per dataset/index (data-driven, never looks
at ACC) -- the realistic, deployable scenario.

\textbf{Best selection} (Section~\ref{sec:oracle}) uses the ORACLE-best
parameter for that dataset's own curve instead (an upper bound on what
Algorithm~1 could ever select, computed with foreknowledge of ACC) --
the same per-dataset oracle computation as the sibling
\texttt{generate\_best\_params\_detailed\_tex.py} script. The gap
between the two scenarios is exactly what Algorithm~1 leaves on the
table. In BOTH scenarios, the Original (raw-$X$) reference value is the
SAME -- it never depends on embedding-parameter tuning.

\textbf{Known data gap -- DBCV / Case 4 / Fashion-MNIST-test.} DBCV's
raw-$X$ (Original) value is not available for this one dataset (hdbscan
is infeasible on the full 784-dimensional representation, for all 3
clustering methods) -- see \texttt{generate\_manifold\_bar\_summary\_
tex.py}'s bar charts, where that specific bar is marked \texttt{N/A}.
Because this report is a SELF comparison, that missing Original value
means DBCV genuinely CANNOT be evaluated for Fashion-MNIST-test in
either scenario -- Fashion-MNIST-test is EXCLUDED ENTIRELY from DBCV's
Case-4 columns (both numerator and denominator, marked $^\dagger$), not
silently dropped and not counted as an improvement or a non-improvement
by default. DBCV's Case-4 fraction is therefore out of the other 8
datasets, not 9.

"""

DOC_CLOSING = r"""\end{document}
"""


def main():
    case_headers = []
    scenarios = {
        'tuned': dict(rho_fn=rho_tuned, label='Tuned'),
        'oracle': dict(rho_fn=rho_oracle, label='Best selection'),
    }
    data = {
        skey: {cm: dict(case_rows=[], per_index={idx: [] for idx in ROW_NAMES}) for cm in CLUSTERING_METHODS}
        for skey in scenarios
    }

    for case in CASES:
        sil_fixed_results = load_manifold_results(case['sil_tuned_path'])
        if sil_fixed_results is None:
            print(f"{case['tag']}: NO SILHOUETTE RESULTS FILE at {case['sil_tuned_path']} -- skipping case entirely")
            continue
        sil_sens_results = load_manifold_results(case['sil_oracle_path'])
        if sil_sens_results is None:
            print(f"{case['tag']}: NO SILHOUETTE SENSITIVITY FILE at {case['sil_oracle_path']} -- skipping case entirely")
            continue
        fixed_results = load_manifold_results(case['tuned_path'])
        if fixed_results is None:
            print(f"{case['tag']}: NO TUNED RESULTS FILE at {case['tuned_path']} -- skipping case entirely (needed for the Original reference in BOTH scenarios)")
            continue
        # fixed_results serves double duty: the 'tuned' scenario's embedded
        # (t-SNE/UMAP) source, AND the Original reference for BOTH scenarios
        # (Original never depends on embedding-parameter tuning).
        merge_silhouette_fixed(fixed_results, sil_fixed_results, case['order'])
        case_headers.append((SHORT_TITLES[case['tag']], len(case['order'])))

        embed_sources = {'tuned': fixed_results}
        sens_results = load_manifold_results(case['oracle_path'])
        if sens_results is None:
            print(f"{case['tag']} / oracle: NO SENSITIVITY RESULTS FILE at {case['oracle_path']} -- skipping oracle scenario")
        else:
            merge_silhouette_sensitivity(sens_results, sil_sens_results, case['order'])
            embed_sources['oracle'] = sens_results

        for skey, sc in scenarios.items():
            embed_results = embed_sources.get(skey)
            if embed_results is None:
                continue
            for cm in CLUSTERING_METHODS:
                tsne_bo = summarize_best_of_all(case, embed_results, fixed_results, cm, 'TSNE', sc['rho_fn'], ROW_NAMES)
                umap_bo = summarize_best_of_all(case, embed_results, fixed_results, cm, 'UMAP', sc['rho_fn'], ROW_NAMES)
                data[skey][cm]['case_rows'].append((case['title'], len(case['order']), tsne_bo, umap_bo))
                for idx in ROW_NAMES:
                    tsne_stat = summarize_index(case, embed_results, fixed_results, cm, idx, 'TSNE', sc['rho_fn'])
                    umap_stat = summarize_index(case, embed_results, fixed_results, cm, idx, 'UMAP', sc['rho_fn'])
                    data[skey][cm]['per_index'][idx].append((tsne_stat, umap_stat))
                print(f"{case['tag']} / {skey} / {cm}: any-of-9 tsne={decimal_cell(tsne_bo)} umap={decimal_cell(umap_bo)}")

    parts = [DOC_PREAMBLE]
    parts.append(r'\section{Tuned (Algorithm 1 actual selection)}')
    parts.append(r'\label{sec:tuned}')
    rows_by_cm = {cm: data['tuned'][cm]['case_rows'] for cm in CLUSTERING_METHODS}
    per_index_by_cm = {cm: data['tuned'][cm]['per_index'] for cm in CLUSTERING_METHODS}
    parts.append(to_case_table('Tuned', 'tab:improvement_tuned', rows_by_cm))
    parts.append('')
    parts.append(to_index_table('Tuned', 'tab:improvement_tuned_by_index', per_index_by_cm, case_headers, ROW_NAMES))
    parts.append('')

    parts.append(r'\section{Best selection (oracle, per-dataset)}')
    parts.append(r'\label{sec:oracle}')
    rows_by_cm = {cm: data['oracle'][cm]['case_rows'] for cm in CLUSTERING_METHODS}
    per_index_by_cm = {cm: data['oracle'][cm]['per_index'] for cm in CLUSTERING_METHODS}
    parts.append(to_case_table('Best selection', 'tab:improvement_oracle', rows_by_cm))
    parts.append('')
    parts.append(to_index_table('Best selection', 'tab:improvement_oracle_by_index', per_index_by_cm, case_headers, ROW_NAMES))
    parts.append('')

    parts.append(DOC_CLOSING)

    combined = '\n'.join(parts)
    combined_path = os.path.join(OUT_DIR, 'improvement_summary.tex')
    with open(combined_path, 'w') as f:
        f.write(combined)
    print(f'\nWrote: {combined_path}')
    print('Compile with:')
    print('  cd tables && pdflatex improvement_summary.tex')


if __name__ == '__main__':
    main()

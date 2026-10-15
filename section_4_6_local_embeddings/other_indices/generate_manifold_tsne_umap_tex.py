"""Legacy detailed embedding report and shared dataset metadata.

This optional report contains per-algorithm tables and diagnostic plots;
it is not an exporter for the current main text Tables 4--5 or Supplement
Figures S7--S37. Those are produced by scripts/build_tables.py and
generate_manifold_bar_summary_tex.py through python reproduce.py cached.
Run this report from its own directory after computing the embedding results."""

import os
import pickle

import numpy as np

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

# ── per-case dataset metadata (K*, N, D) and display names ─────────────
_CASE2_K_STAR_VALUES = [6, 30]
_CASE2_D_VALUES = [2, 10, 200]
_CASE2_COMPACTNESS = ['compact', 'sparse']


def _case2_sample_size_for(k_star):
    """Use the Case 2 sample size N = min(1000, 50 K*)."""
    # Cap N at 1000 while targeting 50 observations per true component:
    # K*=6 gives N=300, and K*=30 gives N=1000 with nearly balanced component sizes.
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
    """Load saved correlations after checking their score-direction convention."""
    # Return an explicit missing-cache value so the report can skip unavailable
    # results without treating them as zero correlations.
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    # Reject caches with a different score orientation; otherwise signs could
    # reverse the meaning of an apparent ranking improvement.
    if payload.get('score_direction_convention') != SCORE_DIRECTION_CONVENTION:
        raise ValueError(f'{path} does not use the expected score-direction convention.')
    return payload['results']


def to_latex_table(case_tag, cm, meta, display, order, results):
    '''One WIDE results table (26 method columns) for (case, clustering_
    method), in a landscape page. Rows = datasets. Missing dataset/method
    combinations render as "--"; bold marks the row-wise max among the
    FINITE entries only.'''
    cm_label = CM_LABELS.get(cm, cm)
    # Assemble the caption, column headings, and LaTeX table structure first;
    # the following row loop inserts the previously computed summaries.
    lines = []
    lines.append(r'\begin{landscape}')
    lines.append(r'\begin{table}[h]')
    lines.append(r'\scriptsize')
    lines.append(r'\setlength{\tabcolsep}{2.2pt}')
    lines.append(
        rf"\caption{{Spearman's $\rho$ with ACC, {cm_label} ({case_tag}). "
        rf"Bold indicates the highest value in each row; `--' marks a "
        rf"dataset/method combination not yet retrieved. Column groups: "
        rf"Orig = computed on raw $X$ (not tuned, no embedding); t-SNE / "
        rf"UMAP = the SAME index computed on the 2D t-SNE / UMAP embedding "
        rf"of $X$ instead, with perplexity*/$n_{{neighbors}}^*$ tuned "
        rf"INDEPENDENTLY for this index via Algorithm~1 (mirroring $r^*$'s "
        rf"relationship to rNN\_Hard's own curve).}}"
    )
    lines.append(r'\centering')
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
        # This auxiliary report uses its own row-maximum highlighting convention.
        # Main text Tables 1--5 are formatted separately by scripts/build_tables.py.
        maxv = max(finite) if finite else None
        cells = []
        # Preserve missing entries as -- and format available correlations to two
        # decimals; the cached numerical values remain unchanged.
        for v in vals:
            if v is None or not np.isfinite(v):
                cells.append('--')
            elif maxv is not None and abs(v - maxv) < 1e-9:
                cells.append(rf'\textbf{{{v:.2f}}}')
            else:
                cells.append(f'{v:.2f}')
        lines.append(
            f"{display[dname]} & {m['K']} & {m['N']} & {m['D']} & "
            + ' & '.join(cells) + r' \\'
        )
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(rf'\label{{tab:manifold_{case_tag}_{cm}}}')
    lines.append(r'\end{table}')
    lines.append(r'\end{landscape}')
    return '\n'.join(lines) + '\n'


def to_delta_table(case_tag, cm, meta, display, order, results):
    '''Companion table to to_latex_table(): for each index, Delta_TSNE =
    TSNE - Original and Delta_UMAP = UMAP - Original (does the embedding
    help or hurt, relative to computing the same index on raw X?). Rows =
    datasets, same order as the main table. Sign shown explicitly (+/-);
    no bolding -- the sign/magnitude itself is the point, not a row max.'''
    # Assemble the caption, column headings, and LaTeX table structure first;
    # the following row loop inserts the previously computed summaries.
    lines = []
    lines.append(r'\begin{landscape}')
    lines.append(r'\begin{table}[h]')
    lines.append(r'\scriptsize')
    lines.append(r'\setlength{\tabcolsep}{2.2pt}')
    lines.append(
        rf"\caption{{$\Delta$ vs.\ Original ($=$ index$_{{\text{{t-SNE/UMAP}}}} - $ "
        rf"index$_{{\text{{Original}}}}$), {CM_LABELS.get(cm, cm)} ({case_tag}). "
        rf"Positive $=$ the embedding-based version correlates with ACC "
        rf"MORE than the same index computed on raw $X$; negative $=$ less. "
        rf"`--' marks a dataset/index combination not yet retrieved.}}"
    )
    lines.append(r'\centering')
    lines.append(r'\begin{tabular}{l' + 'c' * (3 + 2 * len(INDEX_NAMES)) + '}')
    lines.append(r'\hline')
    # Give every index two difference columns, in t-SNE then UMAP order,
    # so the row assembly below matches the displayed column headings.
    header = ['Dataset', '$K^*$', '$N$', '$D$']
    for idx in INDEX_NAMES:
        header += [rf'{idx} $\Delta$t-SNE', rf'{idx} $\Delta$UMAP']
    lines.append(' & '.join(header) + r' \\')
    lines.append(r'\hline')
    for dname in order:
        m = meta[dname]
        row = results.get(cm, {}).get(dname, {})
        cells = []
        for idx in INDEX_NAMES:
            # Use each index's own original-data correlation as the paired baseline.
            # Subtract only when both it and the embedding correlation are available.
            orig = row.get(f'{idx}_Original')
            for rep in ['TSNE', 'UMAP']:
                v = row.get(f'{idx}_{rep}')
                if v is None or orig is None or not np.isfinite(v) or not np.isfinite(orig):
                    cells.append('--')
                else:
                    cells.append(f'{v - orig:+.2f}')
        lines.append(
            f"{display[dname]} & {m['K']} & {m['N']} & {m['D']} & "
            + ' & '.join(cells) + r' \\'
        )
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(rf'\label{{tab:manifold_delta_{case_tag}_{cm}}}')
    lines.append(r'\end{table}')
    lines.append(r'\end{landscape}')
    return '\n'.join(lines) + '\n'


def figure_block(case_tag, cm, dname, display, fig_dir):
    '''One \figure block (in a landscape page, matching the wide 2-panel
    bar+r-landscape plot) for one (case, clustering_method, dataset).'''
    # Reference the already exported PDF and describe its panel encodings.
    # This builds a report fragment; it does not rerender or alter the figure data.
    rel_path = os.path.join('..', fig_dir, cm, EXPERIMENT_TAG, f'Combined_{dname}.pdf')
    lines = []
    lines.append(r'\begin{landscape}')
    lines.append(r'\begin{figure}[h]')
    lines.append(r'\centering')
    lines.append(rf'\includegraphics[width=\textwidth]{{{rel_path}}}')
    lines.append(
        rf'\caption{{{display[dname]} -- {CM_LABELS.get(cm, cm)} ({case_tag}). '
        r'Left: Spearman $\rho$ with ACC for all 25 methods (rNN\_Hard + '
        r'8 indices x 3 representations). Right: rNN\_Hard $\rho$ vs.\ '
        r'$r$, with the selected $r^*$ (dotted) and the diagnostic oracle '
        r'(dashed) -- the only method here with a tuned parameter.}'
    )
    lines.append(r'\end{figure}')
    lines.append(r'\end{landscape}')
    return '\n'.join(lines) + '\n'


DOC_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{pdflscape}

\title{Manifold Alternative Validity Indices (TUNED) vs.\ $r$NN Silhouette: Summary of Cases 2--4\\
\large (auto-generated by generate\_manifold\_tsne\_umap\_tex.py -- do not edit by hand)}
\author{}
\date{}

\begin{document}
\maketitle
\tableofcontents
\clearpage

\section*{Experimental setup}
\addcontentsline{toc}{section}{Experimental setup}

25 methods are compared against clustering accuracy (ACC) across three
case studies: the full 12-setting simulated Gaussian grid (Case 2), the
ten fixed non-globular SIPU benchmark datasets (Case 3), and nine real
labeled datasets (Case 4) -- the same three cases, same datasets, and same
ORIGINAL (non-bounded-window) $K$-grid convention as the sibling
\texttt{codes\_new\_manifold\_tuned\_embedding\_orig\_kgrid} /
\texttt{codes\_new\_manifold\_kernel\_distance\_orig\_kgrid} packages.

\textbf{Method (1): $r$NN-Hard} -- the existing $r$NN affinity silhouette
baseline, computed on raw $X$, with $r^*$ selected via Algorithm~1 exactly
as in every sibling package.

\textbf{Methods (2)--(25): 8 alternative internal validity indices, each
on 3 representations.} VRC (Variance Ratio Criterion, i.e.\ the
Calinski-Harabasz index), WG (Wemmert-Gan\c{c}arski), WB (Zhao \&
Fr\"anti, 2014), DBCV (Moulavi et al., 2014, via \texttt{hdbscan}), CVDD,
VIASCKDE, DCSI (Gauss, Scheipl \& Herrmann, 2025), and DB (Davies-Bouldin,
negated so larger is always better, matching WB's own sign convention).
(CH, the Calinski-Harabasz index, was dropped: it is numerically identical
to VRC by construction, so it was reported only once, under VRC.) All 8
formulas are ported VERBATIM from
\texttt{section\_4\_4\_labeled\_benchmarks/Benchmark\_data\_real.py} -- the same implementations
used for Table~3 in the main manuscript
(verified: this package's Iris/KMeans ``Orig'' column reproduces that
table's published VRC/WG/WB/DBCV/CVDD/VIASCKDE/DCSI values exactly).

Each of the 8 indices is computed on THREE representations, to directly
compare each index's behavior against its own raw-$X$ value:
\textbf{Orig} -- the index computed directly on raw $X$ (identical to the
\texttt{section\_4\_4\_labeled\_benchmarks} reference computation, no embedding, not tuned);
\textbf{t-SNE} -- the SAME index computed on the 2D t-SNE embedding of
$X$ instead of $X$ itself; \textbf{UMAP} -- the SAME index computed on
the 2D UMAP embedding of $X$ instead. UNLIKE the untuned sibling package
(\texttt{codes\_new\_manifold\_alt\_indices\_orig\_kgrid}, which fixes
perplexity=30/$n_{neighbors}$=15), perplexity and $n_{neighbors}$ ARE
tuned here via Algorithm~1 -- and crucially, EACH INDEX gets its OWN
independently-selected perplexity*/$n_{neighbors}^*$, chosen from THAT
index's own $\rho$-vs-$K$ curve, exactly mirroring $r^*$'s relationship
to rNN\_Hard's own curve (not a single shared/proxy criterion). This is
substantially more expensive than the untuned sibling: at every candidate
perplexity/$n_{neighbors}$ value, EVERY requested index's curve must be
computed (not just the eventual winner), so the $O(N^2)$ indices (DBCV,
CVDD, VIASCKDE, and to a lesser extent DCSI) have their already-expensive
per-representation cost multiplied by up to 18x relative to the untuned
sibling. To get partial results sooner and let independent SLURM jobs run
in parallel, the 8 indices are split into a FAST group (VRC, WG, WB, DB
-- no $O(N^2)$ precompute at all) and a SLOW group (DBCV, CVDD,
VIASCKDE, DCSI -- the expensive ones); each dataset/clustering-method
combination is computed by two independent jobs, one per group, and the
results merge automatically (see \texttt{merge\_manifold\_case4\_results.py}).

Only ACC is used as the external metric. Tables are in landscape
orientation (26 columns is too wide for portrait even at \verb|\scriptsize|).

"""

DOC_CLOSING = r"""\end{document}
"""


def main():
    """Build the auxiliary per-dataset embedding correlation and difference report."""
    # Assemble this auxiliary standalone report from saved correlations.
    # The main text/Supplement table fragments are built by scripts/build_tables.py.
    parts = [DOC_PREAMBLE]

    for case in CASES:
        results = load_results(case['results_path'])
        parts.append(f"\\section{{{case['title']}}}\n")
        if results is None:
            parts.append(f"No results file found at \\texttt{{{case['results_path'].replace('_', r'\_')}}} -- skipping.\n\n")
            print(f"{case['tag']}: NO RESULTS FILE -- skipped")
            continue

        # Group each case by clustering algorithm, pairing absolute correlations
        # with same-index embedding differences in two companion tables.
        for cm in CLUSTERING_METHODS:
            if cm not in results:
                print(f"{case['tag']} / {cm}: no results -- skipping table+figures")
                continue

            parts.append(f"\\subsection{{{CM_LABELS.get(cm, cm)}}}\n")
            parts.append(to_latex_table(case['tag'], cm, case['meta'], case['display'], case['order'], results))
            parts.append('')
            parts.append(to_delta_table(case['tag'], cm, case['meta'], case['display'], case['order'], results))
            parts.append('')

            n_present = 0
            for dname in case['order']:
                if dname not in results.get(cm, {}):
                    continue
                # Include a diagnostic PDF only if it has already been generated for
                # this dataset/algorithm pair; report missing figures without inventing data.
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
    print('Compile with (run pdflatex TWICE, in ONE command):')
    print('  cd tables && pdflatex manifold_tsne_umap_summary.tex && pdflatex manifold_tsne_umap_summary.tex')


if __name__ == '__main__':
    main()

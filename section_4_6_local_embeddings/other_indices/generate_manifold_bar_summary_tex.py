"""Draw Supplement Section S12, Figures S7--S37, from tuned ACC correlations.

Combine the Silhouette and other-index result stores. For each dataset,
stack the three clustering algorithms and group original, t-SNE, and UMAP
scores by internal index. Solid, ///, and xxx fills distinguish these
representations; embedding-bar labels give the change from that index's
own original-data correlation. These use Algorithm 1's selected parameters.

Figures S7--S18 cover Case 2, S19--S28 Case 3, and S29--S37 Case 4.
Run here: python generate_manifold_bar_summary_tex.py
python reproduce.py cached collects these PDFs from the repository root."""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from generate_manifold_tsne_umap_tex import (
    DATA_DIR, EXPERIMENT_TAG, CLUSTERING_METHODS, CM_LABELS,
    CASE2_META, CASE2_DISPLAY, CASE2_ORDER,
    CASE3_META, CASE3_DISPLAY, CASE3_ORDER,
    CASE4_META, CASE4_DISPLAY, CASE4_ORDER,
)
from generate_oracle_landscape_plots import INDEX_NAMES, INDEX_COLORS
from generate_improvement_summary_tex import load_manifold_results

OUT_DIR = 'tables'
FIG_DIR = 'figures/bar_summary'
os.makedirs(OUT_DIR, exist_ok=True)

REPS = ['Original', 'TSNE', 'UMAP']
REP_LABELS = {'Original': 'Orig', 'TSNE': 't-SNE', 'UMAP': 'UMAP'}
REP_HATCH = {'Original': '', 'TSNE': '///', 'UMAP': 'xxx'}

RNN_COLOR = 'steelblue'
SIL_COLOR = 'dimgray'
GROUP_GAP = 0.8

SIL_ROOT = os.path.join('..', 'silhouette')
SIL_TAG = 'tsne_umap_tuned_embedding_origkgrid_v1'
SIL_METHOD_KEYS = {'Original': 'Silhouette', 'TSNE': 'TSNE_Ordinary', 'UMAP': 'UMAP_Ordinary'}
SIL_PATHS = {
    tag: os.path.join(SIL_ROOT, 'data', tag, f'manifold_fixed_results_{SIL_TAG}.pkl')
    for tag in ('case2', 'case3', 'case4')
}

CASES = [
    dict(tag='case2', title='Case 2 -- Simulated Data', meta=CASE2_META,
         display=CASE2_DISPLAY, order=CASE2_ORDER,
         results_path=os.path.join(DATA_DIR, 'case2', f'manifold_fixed_results_{EXPERIMENT_TAG}.pkl'),
         sil_path=SIL_PATHS['case2']),
    dict(tag='case3', title='Case 3 -- Nonconvex Data', meta=CASE3_META,
         display=CASE3_DISPLAY, order=CASE3_ORDER,
         results_path=os.path.join(DATA_DIR, 'case3', f'manifold_fixed_results_{EXPERIMENT_TAG}.pkl'),
         sil_path=SIL_PATHS['case3']),
    dict(tag='case4', title='Case 4 -- Real Labeled Data', meta=CASE4_META,
         display=CASE4_DISPLAY, order=CASE4_ORDER,
         results_path=os.path.join(DATA_DIR, 'case4', f'manifold_fixed_results_{EXPERIMENT_TAG}.pkl'),
         sil_path=SIL_PATHS['case4']),
]


def build_bar_groups(row, sil_row):
    """Group original, t-SNE, and UMAP correlations for Figures S7--S37.
    
    Each bar specification carries its own index's original-data reference
    for the delta label. Hard rNN-Sil is a separate reference bar. Silhouette
    appears first, followed by rNN-Sil and the eight other internal indices."""
    # The Silhouette pipeline stores its three representations under separate keys.
    # Pair them with Sil's own original-data rho before joining the other indices.
    sil_orig_v = sil_row.get(SIL_METHOD_KEYS['Original'])
    sil_group = []
    for rep in REPS:
        v = sil_row.get(SIL_METHOD_KEYS[rep])
        orig_ref = sil_orig_v if rep != 'Original' else None
        sil_group.append((f'Silh.\n{REP_LABELS[rep]}', SIL_COLOR, REP_HATCH[rep], v, orig_ref))

    groups = [
        sil_group,
        [('rNN\nHard', RNN_COLOR, '', row.get('rNN_Hard'), None)],
    ]

    for idx in INDEX_NAMES:
        # For each remaining index, the reference is that index on original X.
        # Embedding gains are not measured against a shared Silhouette baseline.
        orig_v = row.get(f'{idx}_Original')
        group = []
        for rep in REPS:
            v = row.get(f'{idx}_{rep}')
            orig_ref = orig_v if rep != 'Original' else None
            group.append((f'{idx}\n{REP_LABELS[rep]}', INDEX_COLORS[idx], REP_HATCH[rep], v, orig_ref))
        groups.append(group)
    return groups


def draw_bars_on_ax(ax, cm, row, sil_row, show_xticklabels):
    """Draw one algorithm's panel in Supplement Figures S7--S37.
    
    Group each index's original, t-SNE, and UMAP correlations together.
    Color identifies the index; hatch identifies the representation.
    Annotate each embedding bar by its difference from that index's own
    original-data correlation. Missing values receive N/A, not a zero score."""
    groups = build_bar_groups(row, sil_row)

    # Flatten the index groups into parallel arrays, retaining a gap between groups.
    # This common layout aligns the three clustering-algorithm panels.
    xs, labels, colors, hatches, vals, orig_refs = [], [], [], [], [], []
    x = 0.0
    for group in groups:
        for label, color, hatch, val, orig_ref in group:
            xs.append(x)
            labels.append(label)
            colors.append(color)
            hatches.append(hatch)
            vals.append(val)
            orig_refs.append(orig_ref)
            x += 1.0
        x += GROUP_GAP

    # Matplotlib needs numeric bar heights. Use a zero placeholder for missing
    # values, then make those bars transparent and label them N/A below.
    finite_mask = [v is not None and np.isfinite(v) for v in vals]
    plot_vals = [v if m else 0.0 for v, m in zip(vals, finite_mask)]

    bars = ax.bar(xs, plot_vals, width=0.9, color=colors, edgecolor='black', linewidth=0.5)
    # Solid, ///, and xxx distinguish original, t-SNE, and UMAP representations.
    # Use the same color within an index group to keep that comparison visible.
    for bar, hatch, m in zip(bars, hatches, finite_mask):
        bar.set_alpha(0.9 if m else 0.0)
        if hatch:
            bar.set_hatch(hatch)

    for xpos, m in zip(xs, finite_mask):
        if not m:
            ax.text(xpos, 0.0, 'N/A', ha='center', va='center', fontsize=7.5,
                     color='black', rotation=90)

    for xpos, v, orig_ref, m in zip(xs, vals, orig_refs, finite_mask):
        if not m or orig_ref is None or not np.isfinite(orig_ref):
            continue
        # Delta rho = embedded rho - this index's original rho for the same dataset
        # and algorithm. Red means improvement; blue means deterioration.
        delta = v - orig_ref
        color = 'red' if delta > 0 else ('blue' if delta < 0 else 'black')
        # Place the difference label beyond the bar's endpoint, including negative bars,
        # so annotation placement follows the sign of the bar height.
        y = v + (0.03 if v >= 0 else -0.03)
        va = 'bottom' if v >= 0 else 'top'
        ax.text(xpos, y, f'{delta:+.2f}', ha='center', va=va, fontsize=8,
                 color=color, fontweight='bold')

    ax.axhline(0, color='grey', linewidth=1, linestyle='--')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels if show_xticklabels else [], fontsize=7.5)
    ax.set_xlim(xs[0] - 1, xs[-1] + 1)
    ax.set_ylabel('Spearman $\\rho$ with ACC', fontsize=11)
    ax.set_ylim(-1.15, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title(CM_LABELS.get(cm, cm), fontsize=12, fontweight='bold', loc='left')


def plot_bar_stacked(dname, rows_by_cm, sil_rows_by_cm, out_dir):
    """One figure per dataset: 3 stacked subplots, one per clustering
    method (KMeans/Spectral/Agglomerative, top to bottom), sharing the
    same x-axis layout so the three methods line up bar-for-bar."""
    # Use one row per algorithm in the fixed K-means/spectral/agglomerative order.
    # Sharing x ensures each index/representation occupies the same horizontal slot.
    fig, axes = plt.subplots(len(CLUSTERING_METHODS), 1, figsize=(22, 16), sharex=True)
    for i, cm in enumerate(CLUSTERING_METHODS):
        row = rows_by_cm.get(cm, {})
        sil_row = sil_rows_by_cm.get(cm, {})
        # Show x labels only on the bottom row to avoid repeating a crowded label set.
        draw_bars_on_ax(axes[i], cm, row, sil_row, show_xticklabels=(i == len(CLUSTERING_METHODS) - 1))

    fig.suptitle(dname, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(out_dir, exist_ok=True)
    # Export one vector PDF per dataset for Figures S7--S37 plus a PNG preview.
    # Return the PDF path for inclusion in the optional standalone report.
    pdf_path = os.path.join(out_dir, f'{dname}.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f'{dname}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return pdf_path


KNOWN_GAPS = {
    'Fashion-MNIST-test': (
        r" \textbf{Note:} DBCV's raw-$X$ (Original) value could not be "
        r"computed for this dataset (hdbscan is infeasible on the full "
        r"784-dim representation) -- its bar is omitted and marked "
        r"\texttt{N/A}; DBCV-t-SNE/UMAP are computed normally but show "
        r"no delta label since there is no Original value to compare "
        r"against."
    ),
}


def figure_block(case_tag, dname, display, pdf_path):
    """Wrap an exported dataset figure in a LaTeX figure block for the standalone report."""
    # Reference the already exported PDF and describe its panel encodings.
    # This builds a report fragment; it does not rerender or alter the figure data.
    lines = []
    lines.append(r'\begin{figure}[h]')
    lines.append(r'\centering')
    lines.append(rf'\includegraphics[width=\textwidth,height=0.92\textheight,keepaspectratio]{{{os.path.join("..", pdf_path)}}}')
    lines.append(
        rf'\caption{{{display[dname]} ({case_tag}) -- top to bottom: '
        r'KMeans, Spectral, Agglomerative, same bar layout in every '
        r'panel. Solid = Original (raw $X$); \texttt{///} hatch = '
        r't-SNE; \texttt{xxx} hatch = UMAP; same color within a triplet '
        r'(Silh.\ and each of the 8 indices). Number above a t-SNE/UMAP '
        r'bar = that bar minus its own Original value '
        r'(\textcolor{red}{red} = increase, \textcolor{blue}{blue} = '
        r'decrease). rNN\_Hard is the only single reference bar (no '
        r'embedding version exists for it). A rotated \texttt{N/A} label '
        r'at zero marks a value that could not be computed at all.'
        + KNOWN_GAPS.get(dname, '') + '}'
    )
    lines.append(r'\end{figure}')
    return '\n'.join(lines) + '\n'


DOC_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{pdflscape}

\title{Manifold Alternative Validity Indices: Bar-Chart Summary\\
\large (auto-generated by generate\_manifold\_bar\_summary\_tex.py -- do not edit by hand)}
\author{}
\date{}

\begin{document}
\maketitle
\tableofcontents
\clearpage

\section*{What this reports}
\addcontentsline{toc}{section}{What this reports}

This is the bar-chart HALF of \texttt{manifold\_tsne\_umap\_summary.tex}'s
\texttt{Combined\_\{dataset\}.pdf} figures, redrawn on its own (the
rNN\_Hard $\rho$-vs-$r$ landscape panel is dropped) using the TUNED
(Algorithm~1 actual selection) results -- the same data as that
document's own table. One figure per DATASET stacks all three
clustering methods (KMeans, Spectral, Agglomerative) as three panels,
top to bottom, sharing the same bar layout -- instead of a separate
figure per (dataset, clustering method) pair.

\textbf{Silhouette is now included, WITH its t-SNE/UMAP versions.} This
manifold package (\texttt{codes\_new\_manifold\_alt\_indices\_tuned\_
orig\_kgrid}) never computed Silhouette itself -- only rNN\_Hard and the
8 alternative indices. The sibling package \texttt{codes\_new\_manifold\_
tuned\_embedding\_orig\_kgrid} runs the identical experiment design for
exactly this purpose: ordinary Euclidean Silhouette on raw $X$
(\texttt{Silhouette}), on a t-SNE embedding (\texttt{TSNE\_Ordinary},
perplexity tuned via Algorithm~1), and on a UMAP embedding
(\texttt{UMAP\_Ordinary}, $n_{neighbors}$ tuned via Algorithm~1). All
three are pulled in here from that sibling's own result pickles under
\texttt{../silhouette/data/} and
drawn as a full Original/t-SNE/UMAP triplet, exactly like every
alternative index.

\textbf{Reading the bars.} Silhouette is drawn FIRST (leftmost group),
matching improvement\_summary.tex's row order, followed by rNN\_Hard,
then the 8 alternative indices. Each of the 8 alternative indices, AND
Silhouette, contributes THREE adjacent bars in the SAME color: solid
fill for Original (raw $X$), \texttt{///} hatch for t-SNE, \texttt{xxx}
hatch for UMAP -- same color, different fill, per the request.
rNN\_Hard is the only single reference bar (no embedding version exists
for it). The number printed above a t-SNE or UMAP bar is that bar's
value MINUS its own group's Original value -- \textcolor{red}{red}
means the embedding improved on the raw-$X$ version, \textcolor{blue}
{blue} means it made it worse.

\textbf{Known data gap.} A rotated \texttt{N/A} label printed at zero
marks a value that simply could not be computed (no bar is drawn there,
rather than a misleading zero-height bar). The one instance of this in
the whole dataset battery: DBCV's raw-$X$ (Original) value is not
available for Case 4's Fashion-MNIST-test (hdbscan is infeasible on the
full 784-dimensional representation, for all 3 clustering methods) --
DBCV-t-SNE/UMAP are still computed normally there, just without a delta
label since there is no Original value to compare against.

"""

DOC_CLOSING = r"""\end{document}
"""


def main():
    """Join both embedding pipelines and export the 31 dataset figures S7--S37."""
    parts = [DOC_PREAMBLE]

    for case in CASES:
        # Join Silhouette and the other-index caches for the same case. Each
        # dataset contributes one figure with all three algorithms stacked.
        sil_results = load_manifold_results(case['sil_path'])
        results = load_manifold_results(case['results_path'])
        parts.append(f"\\section{{{case['title']}}}\n")
        if results is None:
            parts.append(f"No results file found at \\texttt{{{case['results_path'].replace('_', r'\_')}}} -- skipping.\n\n")
            print(f"{case['tag']}: NO RESULTS FILE -- skipped")
            continue
        if sil_results is None:
            print(f"{case['tag']}: NO SILHOUETTE RESULTS FILE at {case['sil_path']} -- Silhouette bars will be blank")
            # Join Silhouette and the other-index caches for the same case. Each
            # dataset contributes one figure with all three algorithms stacked.
            sil_results = {}

        out_dir = os.path.join(FIG_DIR, case['tag'])
        n_figs = 0
        # Follow the configured dataset order used by Figures S7--S37.
        for dname in case['order']:
            rows_by_cm = {cm: results.get(cm, {}).get(dname, {}) for cm in CLUSTERING_METHODS}
            if not any(rows_by_cm.values()):
                continue
            sil_rows_by_cm = {cm: sil_results.get(cm, {}).get(dname, {}) for cm in CLUSTERING_METHODS}
            pdf_path = plot_bar_stacked(dname, rows_by_cm, sil_rows_by_cm, out_dir)
            parts.append(r'\begin{landscape}')
            parts.append(figure_block(case['tag'], dname, case['display'], pdf_path))
            parts.append(r'\end{landscape}')
            parts.append('')
            n_figs += 1
        print(f"{case['tag']}: {n_figs} stacked bar charts generated")
        parts.append(r'\clearpage')
        parts.append('')

    parts.append(DOC_CLOSING)
    combined = '\n'.join(parts)
    combined_path = os.path.join(OUT_DIR, 'manifold_bar_summary.tex')
    with open(combined_path, 'w') as f:
        f.write(combined)
    print(f'\nWrote: {combined_path}')
    print('Compile with (run pdflatex TWICE, in ONE command):')
    print('  cd tables && pdflatex manifold_bar_summary.tex && pdflatex manifold_bar_summary.tex')


if __name__ == '__main__':
    main()
